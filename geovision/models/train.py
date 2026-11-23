"""
Training orchestrator -- runs Stage 4 (parallel ML/DL training) and Stage 5
(validation) together and returns one JSON-serializable report comparing
every model, mirroring the comparative framing used throughout the
literature survey (e.g. Nigar et al. 2024's CNN-vs-RF comparison).
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from geovision.models.classifiers import train_and_compare, train_and_compare_pooled as pooled_classifiers
from geovision.models.forecasting import train_and_compare_forecasters, train_and_compare_pooled as pooled_forecasters

# Per-variable forecaster setup, chosen by a 13-region backtest (report section 11):
# variable -> (use other variables + season as extra inputs, residual learning)
FORECAST_CONFIG = {
    "pm25": (False, False),
    "precipitation_total": (False, False),
    "temperature_2m": (False, False),
    "soil_moisture": (False, True),
    "ndvi": (True, True),   # slow 16-day composite signal: only beats naive with weather/season context
}
EXOG = ["ndvi", "soil_moisture", "temperature_2m", "precipitation_total", "pm25", "no2"]


def forecast_matrix(features: pd.DataFrame, variable: str, exog: bool) -> np.ndarray:
    """(n, F) array, forecast target in column 0; optionally + other variables + day-of-year sin/cos."""
    f = features.sort_values("date")
    d = f[[variable] + ([c for c in EXOG if c != variable and c in f] if exog else [])].copy()
    if exog:
        doy = pd.to_datetime(f["date"]).dt.dayofyear.to_numpy()
        d["season_sin"], d["season_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return d.ffill().bfill().to_numpy(dtype=np.float32)


def run_training(features: pd.DataFrame, forecast_horizon: int = 14,
                  epochs: int = 60) -> dict[str, Any]:
    """Train the RF/XGBoost classifiers and the CNN/LSTM/Transformer
    forecasters on `features` (output of
    `geovision.features.spectral_indices.build_feature_matrix`), returning
    a single report dict ready to serialize to JSON for the API/dashboard.
    """
    report: dict[str, Any] = {"classification": {}, "forecasting": {}}

    # -- Stage 4/5: classical classifiers --------------------------------
    classifier_results = train_and_compare(features)
    for model_type, result in classifier_results.items():
        report["classification"][model_type] = {
            "metrics": result.metrics.to_dict(),
            "top_features": dict(list(result.feature_importances.items())[:5]),
        }

    # -- Stage 4/5: deep-learning forecasters ------------------------------
    for variable, (exog, residual) in FORECAST_CONFIG.items():
        if variable not in features.columns:
            continue
        series = forecast_matrix(features, variable, exog)
        if len(series) < 60:
            continue  # not enough history for a meaningful lookback window
        try:
            forecaster_results = train_and_compare_forecasters(
                series, horizon=forecast_horizon, epochs=epochs, residual=residual
            )
        except (ValueError, ImportError):
            # ValueError: not enough data for lookback/horizon window
            # ImportError: PyTorch not installed — gracefully skip DL forecasters
            continue
        report["forecasting"][variable] = {
            arch: {
                "metrics": result.metrics.to_dict(),
                "forecast": result.forecast.tolist(),
            }
            for arch, result in forecaster_results.items()
        }

    return report


def _classification_entry(result) -> dict[str, Any]:
    return {"metrics": result.metrics.to_dict(),
            "top_features": dict(list(result.feature_importances.items())[:5])}


def _forecast_entry(result) -> dict[str, Any]:
    return {"metrics": result.metrics.to_dict(), "forecast": result.forecast.tolist()}


def run_pooled_training(features_by_region: dict[str, pd.DataFrame], forecast_horizon: int = 14,
                        epochs: int = 30) -> dict[str, dict[str, Any]]:
    """Pooled training: ONE classifier per type and ONE network per (variable, architecture, seed),
    each trained on the training windows of ALL regions (about 13x the data of per-region models),
    then evaluated per region on that region's own untouched chronological hold-out.
    Returns {region_id: report} with the same shape as `run_training`."""
    reports: dict[str, dict[str, Any]] = {rid: {"classification": {}, "forecasting": {}} for rid in features_by_region}

    for rid, by_model in pooled_classifiers(features_by_region).items():
        for mt, result in by_model.items():
            reports[rid]["classification"][mt] = _classification_entry(result)

    for variable, (exog, residual) in FORECAST_CONFIG.items():
        series = {rid: forecast_matrix(f, variable, exog) for rid, f in features_by_region.items()
                  if variable in f.columns and len(f) >= 60}
        if not series:
            continue
        for rid, by_arch in pooled_forecasters(series, horizon=forecast_horizon, epochs=epochs,
                                               residual=residual).items():
            reports[rid]["forecasting"][variable] = {a: _forecast_entry(r) for a, r in by_arch.items()}
    return reports


def best_model_summary(report: dict[str, Any]) -> dict[str, str]:
    """Pick the best-performing model per task by the validation metric the
    literature typically leads with (Kappa for classification, RMSE for
    forecasting) -- useful for a one-line dashboard summary."""
    summary: dict[str, str] = {}

    classification = report.get("classification", {})
    if classification:
        best = max(classification.items(), key=lambda kv: kv[1]["metrics"]["kappa"])
        summary["classification"] = best[0]

    for variable, archs in report.get("forecasting", {}).items():
        best = min(archs.items(), key=lambda kv: kv[1]["metrics"]["rmse"])
        summary[f"forecasting:{variable}"] = best[0]

    return summary
