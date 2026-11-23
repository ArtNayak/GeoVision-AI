"""
Environmental Health Index (EHI) fusion -- Stage 6 of the methodology.

Combines four domain sub-indices (air quality, vegetation, water, climate),
each rescaled to a common 0 (poor) - 100 (excellent) scale, into a single
weighted composite score, echoing the "aggregated environmental index"
precedent surveyed in Aggregated AQI (2025) but extended from air quality
alone to a multi-domain fusion as called for in the report's objectives.
"""
from __future__ import annotations

import pandas as pd

from geovision.config import settings
from geovision.preprocessing.normalize import min_max_normalize

# Physically/statistically reasonable bounds used to rescale each raw
# variable onto the 0-100 sub-index scale. `invert=True` means higher raw
# values are worse (e.g. more PM2.5 is worse air quality).
SUBINDEX_RECIPE = {
    "air_quality": [
        {"variable": "pm25", "lo": 0, "hi": 150, "invert": True, "weight": 0.6},
        {"variable": "no2", "lo": 0, "hi": 200, "invert": True, "weight": 0.4},
    ],
    "vegetation": [
        {"variable": "ndvi", "lo": -0.1, "hi": 0.9, "invert": False, "weight": 1.0},
    ],
    "water": [
        {"variable": "soil_moisture", "lo": 0.05, "hi": 0.45, "invert": False, "weight": 1.0},
    ],
    "climate": [
        # 30-day mean rainfall: one dry day is not a drought
        {"variable": "precipitation_total_roll30_mean", "lo": 0, "hi": 15, "invert": False, "weight": 0.5},
        # Climate comfort penalizes distance from ~295K (~22C), not raw heat.
        {"variable": "temperature_2m", "lo": 0, "hi": 15, "invert": True, "weight": 0.5,
         "transform": "distance_from_295k"},
    ],
}


def _prepare_variable(features: pd.DataFrame, spec: dict) -> pd.Series:
    series = features[spec["variable"]]
    if spec.get("transform") == "distance_from_295k":
        series = (series - 295.0).abs()
    return min_max_normalize(series, spec["lo"], spec["hi"], invert=spec["invert"])


def compute_subindices(features: pd.DataFrame) -> pd.DataFrame:
    """Add one 0-100 column per domain sub-index to `features` (which must
    already contain the raw/engineered variables from
    `geovision.features.spectral_indices.build_feature_matrix`)."""
    out = features.copy()
    for subindex_name, components in SUBINDEX_RECIPE.items():
        available = [c for c in components if c["variable"] in out.columns]
        if not available:
            out[subindex_name] = float("nan")
            continue
        total_weight = sum(c["weight"] for c in available)
        weighted_sum = sum(
            _prepare_variable(out, c) * (c["weight"] / total_weight) for c in available
        )
        out[subindex_name] = weighted_sum
    return out


def compute_composite(features_with_subindices: pd.DataFrame) -> pd.DataFrame:
    """Add the weighted composite `ehi` column from the four sub-indices,
    using `settings.ehi_weights`. Rows where some sub-indices are NaN
    (e.g. cloudy satellite overpass dates) still get an EHI computed from
    whatever sub-indices ARE available, with weights renormalized to 1.0."""
    out = features_with_subindices.copy()
    weights = settings.ehi_weights
    subindex_cols = [k for k in SUBINDEX_RECIPE if k in out.columns]

    ehi_values = []
    for _, row in out.iterrows():
        available = {
            k: weights.get(k, 0) for k in subindex_cols
            if pd.notna(row[k]) and weights.get(k, 0) > 0
        }
        if not available:
            ehi_values.append(float("nan"))
        else:
            total = sum(available.values())
            ehi_values.append(
                float(sum(row[k] * w / total for k, w in available.items()))
            )
    out["ehi"] = ehi_values
    return out



def compute_ehi(features: pd.DataFrame) -> pd.DataFrame:
    """Full Stage-6 pipeline: sub-indices -> composite EHI."""
    return compute_composite(compute_subindices(features))


def latest_snapshot(ehi_df: pd.DataFrame) -> pd.DataFrame:
    """Most recent EHI + sub-index row per region -- what the dashboard's
    comparison view and map need."""
    return (
        ehi_df.sort_values("date")
        .groupby("region_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def compare_regions(ehi_by_region: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a single comparison table (one row per region) from a dict of
    per-region EHI DataFrames -- powers the report's country/state/
    city/continent comparison objective."""
    snapshots = []
    for region_id, df in ehi_by_region.items():
        if df.empty:
            continue
        snap = latest_snapshot(df)
        snap["region_id"] = region_id
        snapshots.append(snap)
    if not snapshots:
        return pd.DataFrame()
    return pd.concat(snapshots, ignore_index=True).sort_values("ehi", ascending=False)
