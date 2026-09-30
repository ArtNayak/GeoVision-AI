"""
Synthetic observation generator.

The live clients (Planetary Computer, ERA5, NASA POWER, OpenAQ) either
require credentials or outbound network access that this sandboxed
environment does not have. This module produces physically-plausible,
seeded-random daily observations for every variable those clients would
otherwise return, so the rest of the platform (preprocessing -> features
-> models -> index -> API -> dashboard) can be built, tested, and demoed
end-to-end without live credentials or network access.

Swap this out for the real clients in `geovision.ingestion.pipeline`
automatically once `pystac-client`/`planetary-computer`/`rioxarray` are
installed and/or CDS_API_KEY / OPENAQ_API_KEY are configured -- the
`Observation` contract is identical either way.
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

            # -- vegetation (GEE: sentinel2/landsat8/modis NDVI) --
            ndvi_base = 0.55 if tropical else 0.40
            ndvi = max(0.0, min(0.95, _seasonal(doy, 0.3, 0.12, ndvi_base) + noise(i) * 0.03))
            observations.append(Observation(region_id, d, "sentinel2", "ndvi", ndvi, "index"))

            # -- water (GEE: sentinel2 NDWI) --
            ndwi_base = 0.10 if tropical else -0.05
            ndwi = max(-0.6, min(0.6, _seasonal(doy, 1.8, 0.10, ndwi_base) + noise(i + 1) * 0.04))
            observations.append(Observation(region_id, d, "sentinel2", "ndwi", ndwi, "index"))

            # -- climate: temperature (ERA5) --
            temp_base = 300 if tropical else 288  # Kelvin
            temp = _seasonal(doy, math.pi if lat < 0 else 0, 8, temp_base) + noise(i + 2, 1.5)
            observations.append(Observation(region_id, d, "era5", "temperature_2m", temp, "K"))

            # -- climate: precipitation (CHIRPS) --
            precip = max(0.0, _seasonal(doy, 2.5, 6, 4 - arid_bias * 3) + noise(i + 3, 3))
            observations.append(Observation(region_id, d, "chirps", "precipitation_total", precip, "mm"))

            # -- air quality: PM2.5 (OpenAQ) --
            pm25_base = 55 if region_id in {"IN", "MH-IN", "MUM-IN", "ID"} else 18
            pm25 = max(2.0, _seasonal(doy, 4.0, 15, pm25_base) + noise(i + 4, 6))
            observations.append(Observation(region_id, d, "openaq", "pm25", pm25, "µg/m³"))

            # -- air quality: NO2 --
            no2 = max(1.0, _seasonal(doy, 1.2, 8, 22) + noise(i + 5, 4))
            observations.append(Observation(region_id, d, "openaq", "no2", no2, "µg/m³"))

        return observations
