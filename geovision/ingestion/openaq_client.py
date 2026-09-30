"""
OpenAQ ingestion client -- ground-station air-quality measurements
(PM2.5, PM10, NO2, O3) that feed the air-quality sub-index and the
Transformer-LSTM PM2.5 forecaster.

Uses OpenAQ's public REST API (v3). An API key is recommended (higher rate
limits) but not strictly required for read access to most endpoints.
"""
from __future__ import annotations

from datetime import date

import requests

from geovision.config import settings, REGION_REGISTRY
from geovision.ingestion.base import BaseIngestionClient, IngestionUnavailableError, Observation

BASE_URL = "https://api.openaq.org/v3"
POLLUTANTS = ["pm25", "pm10", "no2", "o3"]


class OpenAQClient(BaseIngestionClient):
    source_name = "openaq"

    def is_configured(self) -> bool:
        # OpenAQ v3 requires an API key for all requests.
        return bool(settings.credentials.openaq_api_key)

    def fetch(self, region_id: str, start: date, end: date) -> list[Observation]:
        if not self.is_configured():
            raise IngestionUnavailableError(
                "OPENAQ_API_KEY not configured. See README.md for setup."
            )
        region = REGION_REGISTRY.get(region_id)
        if region is None:
            raise ValueError(f"Unknown region_id: {region_id}")

        lon_min, lat_min, lon_max, lat_max = region["bbox"]
        headers = {"X-API-Key": settings.credentials.openaq_api_key}
        params = {
            "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
            "date_from": start.isoformat(),
            "date_to": end.isoformat(),
            "parameters_id": ",".join(POLLUTANTS),
            "limit": 1000,
        }
        try:
            resp = requests.get(f"{BASE_URL}/measurements", headers=headers,
                                 params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise IngestionUnavailableError(f"OpenAQ request failed: {exc}") from exc

        results = resp.json().get("results", [])
        observations: list[Observation] = []
        for row in results:
            observations.append(
                Observation(
                    region_id=region_id,
                    date=date.fromisoformat(row["date"]["local"][:10]),
                    source="openaq",
                    variable=row["parameter"],
                    value=float(row["value"]),
                    unit=row.get("unit", "µg/m³"),
                    quality_flag="ok",
                    meta={"station": row.get("location")},
                )
            )
        return observations
