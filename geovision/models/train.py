"""
Training orchestrator -- runs Stage 4 (parallel ML/DL training) and Stage 5
(validation) together and returns one JSON-serializable report comparing
every model, mirroring the comparative framing used throughout the
literature survey (e.g. Nigar et al. 2024's CNN-vs-RF comparison).
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from geovision.models.classifiers import train_and_compare
from geovision.models.forecasting import train_and_compare_forecasters

FORECAST_VARIABLES = ["pm25", "ndvi", "precipitation_total"]


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
    for variable in FORECAST_VARIABLES:
        if variable not in features.columns:
            continue
        series = features.sort_values("date")[variable].dropna().to_numpy()
        if len(series) < 60:
            continue  # not enough history for a meaningful lookback window
        try:
            forecaster_results = train_and_compare_forecasters(
                series, horizon=forecast_horizon, epochs=epochs
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
