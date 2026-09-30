"""
Validation metrics -- Stage 5 of the methodology.

Implements every metric named in the report's objectives (RMSE, MAE, R²,
Accuracy, Kappa, ROC-AUC) behind a small, consistent API so classifier and
forecaster evaluation report identically-shaped results.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


@dataclass
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClassificationMetrics:
    accuracy: float
    kappa: float
    roc_auc: float | None  # None when the model can't produce probabilities
                            # or the task isn't binary/one-vs-rest scorable

    def to_dict(self) -> dict:
        return asdict(self)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """RMSE, MAE and R² for forecasting models (LSTM/Transformer/CNN and
    the RF/XGBoost regressors used for continuous indicators)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    return RegressionMetrics(rmse=rmse, mae=mae, r2=r2)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> ClassificationMetrics:
    """Accuracy, Cohen's Kappa, and (when class probabilities are
    available) ROC-AUC for the RF/XGBoost environmental classifiers."""
    accuracy = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    roc_auc = None
    if y_proba is not None:
        try:
            n_classes = y_proba.shape[1] if y_proba.ndim > 1 else 2
            if n_classes == 2:
                proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                roc_auc = float(roc_auc_score(y_true, proba_pos))
            else:
                roc_auc = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except ValueError:
            roc_auc = None  # e.g. a class missing from this fold

    return ClassificationMetrics(accuracy=accuracy, kappa=kappa, roc_auc=roc_auc)
