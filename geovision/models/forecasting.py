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
BATCH_SIZE = 32  # benchmarked: bs16 -> bs64 is 4-6x faster for LSTM on both CPU and GPU


def get_device() -> str:
    """GPU when available (LSTM trains ~19x faster than on CPU); override with GEO_DEVICE=cpu."""
    import os
    torch, *_ = _require_torch()
    return os.getenv("GEO_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

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


def build_windows(data: np.ndarray, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """data (n, F) with the forecast target in column 0 -> X (N, lookback, F), y (N, horizon)."""
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + horizon, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _make_model(arch: ArchType, lookback: int, horizon: int, n_features: int = 1):
    """Instantiate the requested PyTorch model (requires torch to be installed)."""
    torch, nn, _, _ = _require_torch()

    class LSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_features, hidden_size=64,
                                num_layers=2, batch_first=True, dropout=0.1)
            self.head = nn.Linear(64, horizon)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return self.head(h_n[-1])

    class CNNForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            channels = 32
            self.net = nn.Sequential(
                nn.Conv1d(n_features, channels, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(channels, horizon)

        def forward(self, x):
            z = self.net(x.transpose(1, 2))  # (batch, channels, 1)
            return self.head(z.squeeze(-1))

    class TransformerForecaster(nn.Module):
        """A compact Transformer encoder, echoing the Transformer-LSTM hybrid
        that Zeng et al. (2026) find outperforms plain LSTM/CNN-LSTM for
        PM2.5 forecasting -- here as a standalone encoder for comparison."""

        def __init__(self):
            super().__init__()
            d_model, nhead, num_layers_t = 32, 4, 2
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_embedding = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                batch_first=True, dropout=0.1,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_t)
            self.head = nn.Linear(d_model, horizon)

        def forward(self, x):
            z = self.input_proj(x) + self.pos_embedding
            z = self.encoder(z)
            return self.head(z.mean(dim=1))

    models = {"lstm": LSTMForecaster, "cnn": CNNForecaster, "transformer": TransformerForecaster}
    if arch not in models:
        raise ValueError(f"Unknown architecture: {arch}")
    model = models[arch]()
    nn.init.zeros_(model.head.weight)  # start as "predict no change" (residual learning)
    nn.init.zeros_(model.head.bias)
    return model


def _train_network(arch, lookback, horizon, n_features, X, y, epochs, lr, residual, batch_size=BATCH_SIZE, seed=42):
    """Fit one network on windows X (N, L, F) -> y (N, H) (normalised units). Returns (model, device)."""
    torch, nn, _, _ = _require_torch()
    torch.manual_seed(seed)  # reproducible results
    dev = get_device()
    model = _make_model(arch, lookback, horizon, n_features).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    # the whole training set lives on the device; manual batching avoids DataLoader overhead
    Xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(y - (X[:, -1, 0:1] if residual else 0.0)).to(dev)
    n = len(Xt)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            optimizer.step()
    return model, dev


def _raw_predict(model, dev, X: np.ndarray, residual: bool) -> np.ndarray:
    """Un-shrunk network output in normalised units, (N, horizon)."""
    torch, _, _, _ = _require_torch()
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X).to(dev)).cpu().numpy()
    return out + X[:, -1, 0:1] if residual else out


def _fit_alpha(model, dev, residual, X_in, y_in, horizon) -> float:
    """Least-squares weight in [0, 1] on the model's correction to the naive forecast."""
    naive = np.repeat(X_in[:, -1, 0:1], horizon, axis=1)
    d = _raw_predict(model, dev, X_in, residual) - naive
    denom = float((d * d).sum())
    return float(np.clip((d * (y_in - naive)).sum() / denom, 0.0, 1.0)) if denom > 1e-12 else 0.0


class _Region:
    """Per-series state: normalisation, held-out windows, shrinkage weight."""

    def __init__(self, series, lookback, horizon):
        data = np.asarray(series, dtype=np.float32)
        data = data[:, None] if data.ndim == 1 else data
        # normalise with training-portion statistics only (no look-ahead into the val period)
        head = data[: int(len(data) * 0.85)]
        self.mean = head.mean(axis=0)
        self.std = np.where(head.std(axis=0) > 0, head.std(axis=0), 1.0)
        norm = (data - self.mean) / self.std
        X, y = build_windows(norm, lookback, horizon)
        if len(X) < 8:
            raise ValueError(
                f"Not enough data to train a lookback={lookback}, horizon={horizon} forecaster "
                f"(need >= 8 sequences, got {len(X)}). Provide a longer history.")
        split = max(1, int(len(X) * 0.85))
        self.X_train, self.y_train = X[:split], y[:split]
        self.X_val, self.y_val = X[split:], y[split:]
        if len(self.X_val) == 0:  # guarantee a non-empty val set for tiny datasets
            self.X_val, self.y_val = self.X_train[-1:], self.y_train[-1:]
        # inner split of the training windows: fit the network on the first part, alpha on the rest
        inner = int(len(self.X_train) * 0.85)
        self.X_in = self.y_in = None
        if inner >= 8 and len(self.X_train) - inner >= 4:
            self.X_in, self.y_in = self.X_train[inner:], self.y_train[inner:]
            self.X_train, self.y_train = self.X_train[:inner], self.y_train[:inner]
        self.last_window = norm[-lookback:]
        self.alpha = 1.0
        self.n_features = data.shape[1]


def _shrunk(model, dev, residual, r: _Region, X, horizon) -> np.ndarray:
    """Denormalised, shrunk predictions for windows X (N, L, F) -> (N, horizon)."""
    naive = np.repeat(X[:, -1, 0:1], horizon, axis=1)
    out = naive + r.alpha * (_raw_predict(model, dev, X, residual) - naive)
    return out * r.std[0] + r.mean[0]


def _evaluate(r: _Region, y_true, y_pred) -> RegressionMetrics:
    m = _metrics_with_baseline(y_true, y_pred, r.X_val, r.mean[0], r.std[0])
    m.shrink = round(r.alpha, 3)
    return m


class SequenceForecaster:
    """Trains one of {lstm, cnn, transformer} and produces a multi-horizon forecast.

    `series` is either 1-D (the target's own history) or 2-D (n, F) with the target in
    column 0 and extra columns as exogenous inputs (e.g. weather, season). Inputs are
    z-scored with training-portion statistics. With `residual=True` (default) the network
    predicts the CHANGE from the last observed target value, and its output layer starts
    at zero, so it begins as the naive forecast and only learns what improves on it.

    Shrinkage: the model is fitted on the first part of the training windows; a weight
    alpha in [0, 1] is then fitted (least squares) on a held-out slice of the TRAINING data
    and forecasts are `naive + alpha * (model - naive)`. Where the model adds nothing alpha
    falls to 0 (= the naive forecast), so it cannot do much worse than naive, and the final
    validation window stays untouched (unbiased metrics).
    """

    def __init__(self, arch: ArchType, lookback: int = DEFAULT_LOOKBACK,
                 horizon: int = 14, epochs: int = 60, lr: float = 1e-3, residual: bool = True):
        self.arch, self.lookback, self.horizon = arch, lookback, horizon
        self.epochs, self.lr, self.residual = epochs, lr, residual
        self.model = None  # nn.Module set by fit()

    def fit(self, series: np.ndarray) -> "SequenceForecaster":
        self.r = r = _Region(series, self.lookback, self.horizon)
        self.model, self._dev = _train_network(self.arch, self.lookback, self.horizon, r.n_features,
                                               r.X_train, r.y_train, self.epochs, self.lr, self.residual)
        if r.X_in is not None:
            r.alpha = _fit_alpha(self.model, self._dev, self.residual, r.X_in, r.y_in, self.horizon)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return _shrunk(self.model, self._dev, self.residual, self.r, X, self.horizon)

    def predict_val(self) -> np.ndarray:
        return self._predict(self.r.X_val)

    def y_val_true(self) -> np.ndarray:
        return self.r.y_val * self.r.std[0] + self.r.mean[0]

    def evaluate(self) -> RegressionMetrics:
        return _evaluate(self.r, self.y_val_true(), self.predict_val())

    def forecast_next(self) -> np.ndarray:
        """Predict the next `horizon` values from the end of the series used in `fit`."""
        return self._predict(self.r.last_window[None])[0]


class PooledForecaster:
    """ONE network trained on the training windows of ALL regions (about 13x the data of a
    per-region model), each region normalised with its own training statistics. Every
    region keeps its own held-out validation windows (its last 15 %, untouched by training,
    and all regions' training windows precede them) and its own shrinkage weight alpha."""

    def __init__(self, arch: ArchType, lookback: int = DEFAULT_LOOKBACK, horizon: int = 14,
                 epochs: int = 30, lr: float = 1e-3, residual: bool = True, batch_size: int = 128, seed: int = 42):
        self.arch, self.lookback, self.horizon, self.seed = arch, lookback, horizon, seed
        self.epochs, self.lr, self.residual, self.batch_size = epochs, lr, residual, batch_size

    def fit(self, series_by_region: dict[str, np.ndarray]) -> "PooledForecaster":
        self.regions = {rid: _Region(s, self.lookback, self.horizon) for rid, s in series_by_region.items()}
        X = np.concatenate([r.X_train for r in self.regions.values()])
        y = np.concatenate([r.y_train for r in self.regions.values()])
        n_features = next(iter(self.regions.values())).n_features
        self.model, self._dev = _train_network(self.arch, self.lookback, self.horizon, n_features, X, y,
                                               self.epochs, self.lr, self.residual, self.batch_size, self.seed)
        for r in self.regions.values():
            if r.X_in is not None:
                r.alpha = _fit_alpha(self.model, self._dev, self.residual, r.X_in, r.y_in, self.horizon)
        return self

    def predict_val(self, rid: str) -> np.ndarray:
        r = self.regions[rid]
        return _shrunk(self.model, self._dev, self.residual, r, r.X_val, self.horizon)

    def y_val_true(self, rid: str) -> np.ndarray:
        r = self.regions[rid]
        return r.y_val * r.std[0] + r.mean[0]

    def evaluate(self, rid: str) -> RegressionMetrics:
        return _evaluate(self.regions[rid], self.y_val_true(rid), self.predict_val(rid))

    def forecast_next(self, rid: str) -> np.ndarray:
        r = self.regions[rid]
        return _shrunk(self.model, self._dev, self.residual, r, r.last_window[None], self.horizon)[0]


@dataclass
class TrainedForecaster:
    arch: str
    metrics: RegressionMetrics
    forecast: np.ndarray  # next `horizon` predicted values from the end of the series


def _metrics_with_baseline(y_true, y_pred, X_val, mean0, std0) -> RegressionMetrics:
    m = regression_metrics(y_true.ravel(), y_pred.ravel())
    naive = np.repeat(X_val[:, -1:, 0], y_true.shape[1], axis=1) * std0 + mean0
    m.baseline_rmse = float(np.sqrt(np.mean((naive - y_true) ** 2)))
    return m


def train_and_compare_forecasters(
    series: np.ndarray,
    architectures: list[ArchType] | None = None,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = 14,
    epochs: int = 60,
    residual: bool = True,
    ensemble: bool = True,
) -> dict[str, TrainedForecaster]:
    """Train each architecture on the same series and return comparable metrics + forecasts,
    plus (optionally) an 'ensemble' that averages the architectures' predictions."""
    archs = architectures or ["lstm", "cnn", "transformer"]
    fitted = {a: SequenceForecaster(a, lookback, horizon, epochs, residual=residual).fit(series) for a in archs}
    results: dict[str, TrainedForecaster] = {
        a: TrainedForecaster(a, f.evaluate(), f.forecast_next()) for a, f in fitted.items()}
    if ensemble and len(fitted) > 1:
        fs = list(fitted.values())
        y_pred = np.mean([f.predict_val() for f in fs], axis=0)
        results["ensemble"] = TrainedForecaster(
            "ensemble",
            _metrics_with_baseline(fs[0].y_val_true(), y_pred, fs[0].r.X_val, fs[0].r.mean[0], fs[0].r.std[0]),
            np.mean([f.forecast_next() for f in fs], axis=0))
    return results


def train_and_compare_pooled(
    series_by_region: dict[str, np.ndarray],
    architectures: list[ArchType] | None = None,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = 14,
    epochs: int = 30,
    residual: bool = True,
    ensemble: bool = True,
    batch_size: int = 128,
    seeds: tuple[int, ...] = (42, 43, 44),
) -> dict[str, dict[str, TrainedForecaster]]:
    """Pooled analogue of `train_and_compare_forecasters`: {region: {arch: TrainedForecaster}}.
    Each architecture's prediction is the average over `seeds` (variance reduction); the
    'ensemble' then averages the architectures."""
    archs = architectures or ["lstm", "cnn", "transformer"]
    fitted = {a: [PooledForecaster(a, lookback, horizon, epochs, residual=residual, batch_size=batch_size, seed=sd)
                  .fit(series_by_region) for sd in seeds] for a in archs}

    def val_pred(a, rid):
        return np.mean([f.predict_val(rid) for f in fitted[a]], axis=0)

    def forecast(a, rid):
        return np.mean([f.forecast_next(rid) for f in fitted[a]], axis=0)

    out: dict[str, dict[str, TrainedForecaster]] = {}
    for rid in series_by_region:
        r0 = fitted[archs[0]][0].regions[rid]
        y_true = fitted[archs[0]][0].y_val_true(rid)

        def alpha(members):  # mean shrinkage weight over the (architecture, seed) members
            return round(float(np.mean([f.regions[rid].alpha for a in members for f in fitted[a]])), 3)

        def metrics(y_pred, members):
            m = _metrics_with_baseline(y_true, y_pred, r0.X_val, r0.mean[0], r0.std[0])
            m.shrink = alpha(members)
            return m

        res = {a: TrainedForecaster(a, metrics(val_pred(a, rid), [a]), forecast(a, rid)) for a in archs}
        if ensemble and len(archs) > 1:
            res["ensemble"] = TrainedForecaster(
                "ensemble", metrics(np.mean([val_pred(a, rid) for a in archs], axis=0), archs),
                np.mean([forecast(a, rid) for a in archs], axis=0))
        out[rid] = res
    return out
