"""
MODIS NDVI ingestion client -- real satellite NDVI, free, no API key.

Uses NASA ORNL DAAC's MODIS web-service (MOD13Q1, 250 m, 16-day composites),
which returns pixel values around a point -- no raster handling needed. For each region we
take a 5x5 km window at up to 5 LAND points (the centroid plus a grid inside the region's
bbox; ocean points are recognised by the no-data fill value), keep only pixels whose
pixel-reliability flag is good/marginal (0/1), and average the points. This matters: under monsoon/persistent
cloud the composite pixels are flagged cloudy (3) and their NDVI is garbage (India
July-Aug: 100 % cloudy, NDVI 0.09 instead of ~0.6). Composites with <25 % clear pixels
are skipped and the last good value is carried forward by the gap-fill.
Composites are published with ~1-2 months latency.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np

from geovision.config import REGION_REGISTRY
from geovision.ingestion.base import (BaseIngestionClient, IngestionUnavailableError, Observation, get_json,
                                      sample_points)

BASE_URL = "https://modis.ornl.gov/rst/api/v1/MOD13Q1"
BAND = "250m_16_days_NDVI"
RELIABILITY = "250m_16_days_pixel_reliability"
HALF_WIDTH_KM = 2          # 5 x 5 km window
MIN_CLEAR_FRACTION = 0.25
MAX_POINTS = 5             # land points averaged per region (centroid first, then a bbox grid)
MAX_TILES = 10  # API limit per subset request


class ModisNdviClient(BaseIngestionClient):
    source_name = "modis_ndvi"

    def is_configured(self) -> bool:
        return True

    @staticmethod
    def _is_land(loc: dict, modis_date: str) -> bool:
        try:
            rows = get_json(f"{BASE_URL}/subset", {**loc, "band": BAND, "startDate": modis_date,
                                                  "endDate": modis_date, "kmAboveBelow": 0, "kmLeftRight": 0},
                            ttl=12 * 3600)["subset"]
        except IngestionUnavailableError:  # the service has no tile here (open ocean)
            return False
        return bool(rows) and rows[0]["data"][0] > -3000  # water / no-data pixels carry the fill value

    @staticmethod
    def _point_series(loc: dict, wanted: list[str]) -> dict[str, float]:
        """{calendar_date: clear-pixel mean NDVI} for one point; cloudy composites are omitted."""
        out: dict[str, float] = {}
        for i in range(0, len(wanted), MAX_TILES):
            chunk = wanted[i:i + MAX_TILES]
            q = {**loc, "startDate": chunk[0], "endDate": chunk[-1],
                 "kmAboveBelow": HALF_WIDTH_KM, "kmLeftRight": HALF_WIDTH_KM}
            ndvi = get_json(f"{BASE_URL}/subset", {**q, "band": BAND}, ttl=12 * 3600)["subset"]
            rel = {r["calendar_date"]: r["data"] for r in
                   get_json(f"{BASE_URL}/subset", {**q, "band": RELIABILITY}, ttl=12 * 3600)["subset"]}
            for row in ndvi:
                vals = np.array(row["data"], dtype=float)
                flags = np.array(rel[row["calendar_date"]])
                clear = vals[((flags == 0) | (flags == 1)) & (vals > -3000)]
                if len(clear) >= MIN_CLEAR_FRACTION * len(vals):  # else: cloudy composite
                    out[row["calendar_date"]] = float(clear.mean()) * 0.0001
        return out

    def fetch(self, region_id: str, start: date, end: date) -> list[Observation]:
        region = REGION_REGISTRY.get(region_id)
        if region is None:
            raise ValueError(f"Unknown region_id: {region_id}")
        pts = [{"latitude": la, "longitude": lo} for la, lo in sample_points(region)]

        dates = get_json(f"{BASE_URL}/dates", pts[0], ttl=12 * 3600)["dates"]  # same tile calendar for the whole region
        # look back up to 200 days before `start`: if the window itself is all cloudy (monsoon,
        # publishing lag) the latest clear composite before it is carried into the window
        wanted = [d["modis_date"] for d in dates
                  if start - timedelta(days=200) <= date.fromisoformat(d["calendar_date"]) <= end]
        if not wanted:
            raise IngestionUnavailableError(f"No MODIS composites for {region_id} in window")

        with ThreadPoolExecutor(len(pts)) as ex:  # keep land points only, at most MAX_POINTS
            land = [p for p, ok in zip(pts, ex.map(lambda q: self._is_land(q, wanted[-1]), pts)) if ok]
        land = land[:MAX_POINTS] or pts[:1]
        with ThreadPoolExecutor(len(land)) as ex:
            per_point = list(ex.map(lambda q: self._point_series(q, wanted), land))

        by_date: dict[str, list[float]] = {}
        for series in per_point:
            for d, v in series.items():
                by_date.setdefault(d, []).append(v)
        obs = [Observation(region_id, date.fromisoformat(d), "modis", "ndvi", sum(v) / len(v), "index",
                           meta={"points": len(v), "composite": d}) for d, v in sorted(by_date.items())]
        if not obs:
            raise IngestionUnavailableError(f"No clear MODIS NDVI for {region_id}")
        # composites before the window: keep only the latest one, dated at the window start
        before = [o for o in obs if o.date < start]
        obs = [o for o in obs if o.date >= start]
        if before:
            last = max(before, key=lambda o: o.date)
            obs.append(Observation(region_id, start, "modis", "ndvi", last.value, "index", meta=last.meta))
        return obs
