"""
Deep-learning time-series forecasters -- CNN, LSTM and Transformer, per the
literature survey (Nigar et al. 2024: CNN outperforms RF on classification;
Zeng et al. 2026: Transformer-LSTM outperforms plain LSTM/CNN-LSTM for
PM2.5; Mohammed et al. 2024: LSTM for long-range drought forecasting).

All three share a `SequenceForecaster` wrapper so the training script can
loop over architectures and compare them on identical train/test splits and
metrics (Stage 4/5 of the methodology), forecasting `horizon` steps ahead
for any core variable (PM2.5, NDVI, precipitation, temperature). 

`torch` (PyTorch) is imported lazily -- the module can be imported and the
data-preparation helpers used without PyTorch installed.  Only
`SequenceForecaster.fit()` and the `_make_model` factory require it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # only for type checkers, never at runtime
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

from geovision.models.evaluate import RegressionMetrics, regression_metrics


def _require_torch():
    """Import torch lazily and raise a helpful error if it is not installed."""
    try:
        import torch as _torch
        from torch import nn as _nn
        from torch.utils.data import DataLoader as _DataLoader, TensorDataset as _TensorDataset
        return _torch, _nn, _DataLoader, _TensorDataset
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for deep-learning forecasters. "
            "Install it with: pip install torch  "
            "(see https://pytorch.org/get-started/locally/ for your platform)."
        ) from exc

ArchType = Literal["lstm", "cnn", "transformer"]

DEFAULT_LOOKBACK = 30


def build_sequences(series: np.ndarray, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Slide a `lookback`-length window across `series` to build
    (X, y) pairs where y is the next `horizon` values -- the standard
    supervised framing for sequence forecasting."""
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback:i + lookback + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _make_model(arch: ArchType, lookback: int, horizon: int):
    """Instantiate the requested PyTorch model (requires torch to be installed)."""
    torch, nn, _, _ = _require_torch()

    class LSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=64,
                                num_layers=2, batch_first=True, dropout=0.1)
            self.head = nn.Linear(64, horizon)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x.unsqueeze(-1))
            return self.head(h_n[-1])

    class CNNForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            channels = 32
            self.net = nn.Sequential(
                nn.Conv1d(1, channels, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(channels, horizon)

        def forward(self, x):
            z = self.net(x.unsqueeze(1))  # (batch, channels, 1)
            return self.head(z.squeeze(-1))

    class TransformerForecaster(nn.Module):
        """A compact Transformer encoder, echoing the Transformer-LSTM hybrid
        that Zeng et al. (2026) find outperforms plain LSTM/CNN-LSTM for
        PM2.5 forecasting -- here as a standalone encoder for comparison."""

        def __init__(self):
            super().__init__()
            d_model, nhead, num_layers_t = 32, 4, 2
            self.input_proj = nn.Linear(1, d_model)
            self.pos_embedding = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                batch_first=True, dropout=0.1,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_t)
            self.head = nn.Linear(d_model, horizon)

        def forward(self, x):
            z = self.input_proj(x.unsqueeze(-1)) + self.pos_embedding
            z = self.encoder(z)
            return self.head(z.mean(dim=1))

    if arch == "lstm":
        return LSTMForecaster()
    if arch == "cnn":
        return CNNForecaster()
    if arch == "transformer":
        return TransformerForecaster()
    raise ValueError(f"Unknown architecture: {arch}")


@dataclass
class TrainedForecaster:
    arch: ArchType
    metrics: RegressionMetrics
    forecast: np.ndarray  # next `horizon` predicted values from the end of the series


class SequenceForecaster:
    """Trains one of {lstm, cnn, transformer} on a single-variable time
    series and produces a multi-horizon forecast, with z-score
    normalization applied/reversed internally."""

    def __init__(self, arch: ArchType, lookback: int = DEFAULT_LOOKBACK,
                 horizon: int = 14, epochs: int = 60, lr: float = 1e-3):
        self.arch = arch
        self.lookback = lookback
        self.horizon = horizon
        self.epochs = epochs
        self.lr = lr
        self.model = None  # type: ignore[assignment]  # nn.Module set by fit()
        self._mean = 0.0
        self._std = 1.0

    def _normalize(self, series: np.ndarray) -> np.ndarray:
        return (series - self._mean) / self._std

    def _denormalize(self, series: np.ndarray) -> np.ndarray:
        return series * self._std + self._mean

    def fit(self, series: np.ndarray) -> "SequenceForecaster":
        series = np.asarray(series, dtype=np.float32)
        self._mean, self._std = float(series.mean()), float(series.std() or 1.0)
        norm = self._normalize(series)

        X, y = build_sequences(norm, self.lookback, self.horizon)
        if len(X) < 8:
            raise ValueError(
                f"Not enough data to train a lookback={self.lookback}, "
                f"horizon={self.horizon} forecaster (need >= 8 sequences, "
                f"got {len(X)}). Provide a longer history."
            )

        split = max(1, int(len(X) * 0.85))
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]
        if len(X_val) == 0:  # guarantee a non-empty val set for tiny datasets
            X_val, y_val = X_train[-1:], y_train[-1:]

        torch, nn, DataLoader, TensorDataset = _require_torch()

        self.model = _make_model(self.arch, self.lookback, self.horizon)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=16, shuffle=True,
        )

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

        self._X_val, self._y_val = X_val, y_val
        self._last_window = norm[-self.lookback:]
        return self

    def evaluate(self) -> RegressionMetrics:
        torch, _, _, _ = _require_torch()
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.from_numpy(self._X_val)).numpy()
        y_true = self._denormalize(self._y_val).ravel()
        y_pred = self._denormalize(pred).ravel()
        return regression_metrics(y_true, y_pred)

    def forecast_next(self) -> np.ndarray:
        """Predict the next `horizon` values from the end of the series
        used in `fit`, in original (denormalized) units."""
        torch, _, _, _ = _require_torch()
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(self._last_window[None, :])
            pred = self.model(x).numpy().ravel()
        return self._denormalize(pred)


def train_and_compare_forecasters(
    series: np.ndarray,
    architectures: list[ArchType] | None = None,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = 14,
    epochs: int = 60,
) -> dict[ArchType, TrainedForecaster]:
    """Train each requested architecture on the same series and return
    comparable metrics + forecasts -- the DL analogue of
    `classifiers.train_and_compare`."""
    if architectures is None:
        architectures = ["lstm", "cnn", "transformer"]
    results: dict[ArchType, TrainedForecaster] = {}
    for arch in architectures:
        forecaster = SequenceForecaster(
            arch=arch, lookback=lookback, horizon=horizon, epochs=epochs
        ).fit(series)
        results[arch] = TrainedForecaster(
            arch=arch,
            metrics=forecaster.evaluate(),
            forecast=forecaster.forecast_next(),
        )
    return results
