"""
Quality handling: physical-range validation and gap-filling.

Readings outside physically valid ranges are set to missing, then every
(region, source, variable) series is resampled to a daily grid and interpolated
(16-day MODIS composites become daily values; the most recent weeks carry the last
value forward). Cloud screening itself happens in the MODIS client, which keeps only
clear-pixel composites (pixel-reliability flag).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Physically valid ranges used to flag corrupted / sensor-error readings.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "ndvi": (-1.0, 1.0),
    "soil_moisture": (0.0, 1.0),  # m3/m3
    "temperature_2m": (200.0, 330.0),   # Kelvin
    "precipitation_total": (0.0, 500.0),  # mm/window
    "pm25": (0.0, 1000.0),
    "no2": (0.0, 500.0),
}


def flag_out_of_range(df: pd.DataFrame) -> pd.DataFrame:
    """Mark physically-implausible readings as missing so they get
    interpolated rather than corrupting downstream statistics."""
    df = df.copy()
    for variable, (lo, hi) in VALID_RANGES.items():
        mask = (df["variable"] == variable) & ~df["value"].between(lo, hi)
        df.loc[mask, "quality_flag"] = "invalid"
        df.loc[mask, "value"] = np.nan
    return df


def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate and gap-fill missing/cloud-masked readings per
    (region_id, source, variable) time series across the full date range to
    daily frequency, matching the report's 'cloud masking and correction'
    preprocessing stage.

    Satellite overpasses occur at 5-16 day cadences (or can be cloud-covered),
    so sparse observation dates are resampled to daily frequency and continuous
    profiles are produced via time-interpolation with forward/backward fill.
    """
    if df.empty:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    min_date = df["date"].min()
    max_date = df["date"].max()
    full_dates = pd.date_range(min_date, max_date, freq="D", name="date")

    filled_frames = []
    group_cols = ["region_id", "source", "variable"]
    for (region_id, source, variable), group in df.groupby(group_cols):
        # Drop any intra-group date duplicates before reindexing
        group = group.drop_duplicates(subset=["date"]).set_index("date")
        
        # Reindex to complete daily date grid
        group = group.reindex(full_dates)
        group["region_id"] = region_id
        group["source"] = source
        group["variable"] = variable
        
        # Fill missing unit
        unit_val = group["unit"].dropna().iloc[0] if not group["unit"].dropna().empty else "index"
        group["unit"] = unit_val

        was_missing = group["value"].isna()
        # Interpolate and pad edge dates with backfill / forward fill
        group["value"] = (
            group["value"]
            .interpolate(method="time", limit_direction="both")
            .bfill()
            .ffill()
        )
        group["quality_flag"] = group["quality_flag"].fillna("interpolated")
        group.loc[~was_missing & (group["quality_flag"] == "interpolated"), "quality_flag"] = "ok"

        filled_frames.append(group.reset_index())

    if not filled_frames:
        return df

    return pd.concat(filled_frames, ignore_index=True)


def run_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Full Stage-2 pipeline: range validation followed by gap filling."""
    df = flag_out_of_range(df)
    df = fill_gaps(df)
    return df.sort_values(["variable", "date"]).reset_index(drop=True)
