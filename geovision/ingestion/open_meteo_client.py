"""
Open-Meteo ingestion client -- free, no API key, real data:

* ERA5 reanalysis (archive API):  daily mean 2 m temperature, daily precipitation
* ERA5-Land (archive API):        root-zone soil moisture 0-100 cm (water sub-index; the 0-7 cm
                                   surface layer swings wildly day to day, e.g. Mumbai 0.26 -> 0.07
                                   in a week while the root zone stayed at 0.26)
* CAMS reanalysis (air-quality API): PM2.5 and NO2 (hourly -> daily mean)
* Live feed (forecast API, `past_days`): fills the last ~2 days the ERA5 archive has not
  published yet, so the series ends today instead of carrying values forward.

Values are the mean over the LAND points of a grid inside the region's bbox (plus the
centroid); ocean points are recognised by ERA5-Land soil moisture == 0 and excluded.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

import numpy as np

from geovision.config import REGION_REGISTRY
from geovision.ingestion.base import BaseIngestionClient, Observation, get_json, sample_points

SOIL = "soil_moisture_0_to_100cm"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
LIVE_URL = "https://api.open-meteo.com/v1/forecast"  # near-real-time analysis, no archive lag


def _daily_mean(times: list[str], values: list) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for t, v in zip(times, values):
        if v is not None:
            acc[t[:10]].append(v)
    return {d: sum(v) / len(v) for d, v in acc.items()}


class OpenMeteoClient(BaseIngestionClient):
    source_name = "open_meteo"

    def is_configured(self) -> bool:
        return True  # public API, no credentials

    def fetch(self, region_id: str, start: date, end: date) -> list[Observation]:
        region = REGION_REGISTRY.get(region_id)
        if region is None:
            raise ValueError(f"Unknown region_id: {region_id}")
        pts = sample_points(region)
        loc = {"latitude": ",".join(str(p[0]) for p in pts), "longitude": ",".join(str(p[1]) for p in pts),
               "timezone": "UTC"}
        p = {**loc, "start_date": start.isoformat(), "end_date": end.isoformat()}

        def as_list(j):  # one point -> dict, several -> list
            return j if isinstance(j, list) else [j]

        # the ERA5 archive lags ~2 days; the live feed below fills the most recent days
        archive_end = min(end, date.today() - timedelta(days=2)).isoformat()
        w = as_list(get_json(ARCHIVE_URL, {**p, "end_date": archive_end,
                                           "daily": "temperature_2m_mean,precipitation_sum", "hourly": SOIL}, ttl=6 * 3600))
        a = as_list(get_json(AIR_URL, {**p, "hourly": "pm2_5,nitrogen_dioxide"}))
        live = None
        if end > date.fromisoformat(archive_end):
            live = as_list(get_json(LIVE_URL, {**loc, "past_days": 5, "forecast_days": 1,
                                               "daily": "temperature_2m_mean,precipitation_sum", "hourly": SOIL}))

        # land mask: ERA5-Land soil moisture is exactly 0 over the ocean
        land = [i for i, r in enumerate(w) if np.nanmean([v if v is not None else np.nan
                                                          for v in r["hourly"][SOIL]]) > 0.001] or [0]

        def mean_over_land(per_point: list[dict[str, float]]) -> dict[str, float]:
            acc: dict[str, list[float]] = defaultdict(list)
            for i in land:
                for d, v in per_point[i].items():
                    acc[d].append(v)
            return {d: sum(v) / len(v) for d, v in acc.items()}

        def daily(r, name, conv=lambda v: v):
            return {d: conv(v) for d, v in zip(r["daily"]["time"], r["daily"][name]) if v is not None}

        series = {
            "temperature_2m": ("K", mean_over_land([daily(r, "temperature_2m_mean", lambda v: v + 273.15) for r in w])),
            "precipitation_total": ("mm", mean_over_land([daily(r, "precipitation_sum") for r in w])),
            "soil_moisture": ("m3/m3", mean_over_land([_daily_mean(r["hourly"]["time"], r["hourly"][SOIL]) for r in w])),
            "pm25": ("µg/m³", mean_over_land([_daily_mean(r["hourly"]["time"], r["hourly"]["pm2_5"]) for r in a])),
            "no2": ("µg/m³", mean_over_land([_daily_mean(r["hourly"]["time"], r["hourly"]["nitrogen_dioxide"]) for r in a])),
        }
        if live:  # days beyond the archive (never overwrite archive values)
            fresh = {
                "temperature_2m": mean_over_land([daily(r, "temperature_2m_mean", lambda v: v + 273.15) for r in live]),
                "precipitation_total": mean_over_land([daily(r, "precipitation_sum") for r in live]),
                "soil_moisture": mean_over_land([_daily_mean(r["hourly"]["time"], r["hourly"][SOIL]) for r in live]),
            }
            for var, extra in fresh.items():
                series[var][1].update({d: v for d, v in extra.items() if archive_end < d <= end.isoformat()})
        return [
            Observation(region_id, date.fromisoformat(d), self.source_name, var, float(v), unit)
            for var, (unit, by_day) in series.items() for d, v in by_day.items()
        ]
