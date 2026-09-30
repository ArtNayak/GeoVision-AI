"""Unit harmonization and normalization helpers (Stage 2, continued)."""
from __future__ import annotations

import pandas as pd


def kelvin_to_celsius(df: pd.DataFrame) -> pd.DataFrame:
    """Add a Celsius-unit copy of any Kelvin temperature readings so
    downstream consumers (dashboard, health-index thresholds) can work in
    human-readable units without repeating the conversion."""
    df = df.copy()
    mask = (df["variable"] == "temperature_2m") & (df["unit"] == "K")
    converted = df.loc[mask].copy()
    converted["value"] = converted["value"] - 273.15
    converted["unit"] = "C"
    converted["variable"] = "temperature_2m_c"
    return pd.concat([df, converted], ignore_index=True)


def min_max_normalize(series: pd.Series, lo: float, hi: float, invert: bool = False) -> pd.Series:
    """Scale a series to 0-100 given known physical/statistical bounds.

    Used by the Environmental Health Index fusion stage, where every raw
    variable (NDVI, PM2.5, precipitation deficit, ...) must be placed on a
    common 0 ("poor") - 100 ("excellent") scale before weighted averaging.
    `invert=True` for variables where higher raw values mean *worse*
    environmental condition (e.g. PM2.5).
    """
    clipped = series.clip(lower=lo, upper=hi)
    scaled = (clipped - lo) / (hi - lo) * 100.0
    return (100.0 - scaled) if invert else scaled
