"""
Synthetic observation generator -- TEST FIXTURE ONLY (not used by the pipeline).

Seeded, physically-plausible daily observations for every core variable, used only by
unit tests that must run offline. The pipeline never falls back to this generator.
"""
from __future__ import annotations

import hashlib
import math
from datetime import date, timedelta

from geovision.config import REGION_REGISTRY
from geovision.ingestion.base import BaseIngestionClient, Observation


def _region_seed(region_id: str) -> int:
    return int(hashlib.sha256(region_id.encode()).hexdigest(), 16) % (2**32)


def _seasonal(day_of_year: int, phase: float, amplitude: float, base: float) -> float:
    return base + amplitude * math.sin(2 * math.pi * (day_of_year / 365.0) + phase)


class SyntheticClient(BaseIngestionClient):
    """Drop-in stand-in for any single live client, or generate all
    variables at once via `fetch_all`."""

    source_name = "synthetic"

    def is_configured(self) -> bool:
        return True

    def fetch(self, region_id: str, start: date, end: date) -> list[Observation]:
        return self.fetch_all(region_id, start, end)

    def fetch_all(self, region_id: str, start: date, end: date) -> list[Observation]:
        region = REGION_REGISTRY.get(region_id)
        if region is None:
            raise ValueError(f"Unknown region_id: {region_id}")

        seed = _region_seed(region_id)
        lat = region["lat"]
        # Rough climate archetype from latitude: tropical / temperate / polar-ish.
        tropical = abs(lat) < 15
        arid_bias = 0.4 if region_id in {"AU", "KE"} else 0.0

        observations: list[Observation] = []
        n_days = (end - start).days + 1
        for i in range(n_days):
            d = start + timedelta(days=i)
            doy = d.timetuple().tm_yday
            noise = lambda k, scale=1.0: (
                ((seed >> (k % 24)) % 1000) / 1000.0 - 0.5
            ) * 2 * scale

            # -- vegetation (NDVI) --
            ndvi_base = 0.55 if tropical else 0.40
            ndvi = max(0.0, min(0.95, _seasonal(doy, 0.3, 0.12, ndvi_base) + noise(i) * 0.03))
            observations.append(Observation(region_id, d, "sentinel2", "ndvi", ndvi, "index"))

            # -- water (soil moisture) --
            sm_base = 0.30 if tropical else 0.22
            sm = max(0.02, min(0.5, _seasonal(doy, 1.8, 0.08, sm_base) + noise(i + 1) * 0.02))
            observations.append(Observation(region_id, d, "sentinel2", "soil_moisture", sm, "m3/m3"))

            # -- climate: temperature (ERA5) --
            temp_base = 300 if tropical else 288  # Kelvin
            temp = _seasonal(doy, math.pi if lat < 0 else 0, 8, temp_base) + noise(i + 2, 1.5)
            observations.append(Observation(region_id, d, "era5", "temperature_2m", temp, "K"))

            # -- climate: precipitation --
            precip = max(0.0, _seasonal(doy, 2.5, 6, 4 - arid_bias * 3) + noise(i + 3, 3))
            observations.append(Observation(region_id, d, "chirps", "precipitation_total", precip, "mm"))

            # -- air quality: PM2.5 --
            pm25_base = 55 if region_id in {"IN", "MH-IN", "MUM-IN", "ID"} else 18
            pm25 = max(2.0, _seasonal(doy, 4.0, 15, pm25_base) + noise(i + 4, 6))
            observations.append(Observation(region_id, d, "openaq", "pm25", pm25, "µg/m³"))

            # -- air quality: NO2 --
            no2 = max(1.0, _seasonal(doy, 1.2, 8, 22) + noise(i + 5, 4))
            observations.append(Observation(region_id, d, "openaq", "no2", no2, "µg/m³"))

        return observations
