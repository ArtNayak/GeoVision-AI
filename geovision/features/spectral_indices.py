"""
Feature engineering -- Stage 3 of the methodology.

Turns the tidy, preprocessed observation DataFrame (one row per
region/date/source/variable) into a wide, model-ready feature matrix: one
row per region/date, one column per variable, plus rolling-window
statistics (short-term trend/anomaly context for the LSTM/Transformer
forecasters) and a PCA-compressed representation used by the classical ML
classifiers (Random Forest, XGBoost) to reduce collinearity between the
spectral indices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CORE_VARIABLES = [
    "ndvi", "ndwi", "temperature_2m", "precipitation_total", "pm25", "no2",
]
ROLLING_WINDOWS = (7, 30)


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot tidy observations into one row per (region_id, date)."""
    if df.empty:
        return pd.DataFrame(columns=["region_id", "date", *CORE_VARIABLES])
    wide = df.pivot_table(
        index=["region_id", "date"], columns="variable", values="value", aggfunc="mean"
    ).reset_index()
    wide.columns.name = None
    return wide.sort_values(["region_id", "date"])


def add_rolling_features(wide: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean/std features per region for each core variable,
    giving the forecasting models short- and medium-term context and
    surfacing anomalies (e.g. a PM2.5 spike vs its 30-day baseline)."""
    wide = wide.copy()
    out_frames = []
    for _, group in wide.groupby("region_id"):
        group = group.sort_values("date").set_index("date")
        for var in CORE_VARIABLES:
            if var not in group.columns:
                continue
            for window in ROLLING_WINDOWS:
                group[f"{var}_roll{window}_mean"] = (
                    group[var].rolling(window, min_periods=1).mean()
                )
                group[f"{var}_roll{window}_std"] = (
                    group[var].rolling(window, min_periods=2).std().fillna(0.0)
                )
            group[f"{var}_anomaly"] = group[var] - group[f"{var}_roll30_mean"]
        out_frames.append(group.reset_index())
    return pd.concat(out_frames, ignore_index=True)


def apply_pca(features: pd.DataFrame, n_components: int = 3,
              columns: list[str] | None = None) -> tuple[pd.DataFrame, PCA, StandardScaler]:
    """Standardize and PCA-compress the core variables, returning the
    component scores appended to the frame plus the fitted PCA/scaler so
    the same transform can be applied to future batches (important for
    near real-time scoring, where refitting PCA on every new day would
    make component meaning drift)."""
    columns = columns or [c for c in CORE_VARIABLES if c in features.columns]
    clean = features[columns].fillna(features[columns].mean())

    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean)

    n_components = min(n_components, len(columns))
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(scaled)

    out = features.copy()
    for i in range(n_components):
        out[f"pca_{i+1}"] = components[:, i]
    return out, pca, scaler


def build_feature_matrix(df: pd.DataFrame, n_pca_components: int = 3) -> pd.DataFrame:
    """Full Stage-3 pipeline: pivot -> rolling features -> PCA."""
    wide = to_wide(df)
    if wide.empty:
        return wide
    enriched = add_rolling_features(wide)
    enriched, _, _ = apply_pca(enriched, n_components=n_pca_components)
    return enriched
