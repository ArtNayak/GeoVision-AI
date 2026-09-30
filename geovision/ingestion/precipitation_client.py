"""
Precipitation ingestion client -- NASA POWER (free, no API key, no auth).

The old `CHIRPSClient` piggybacked on GEEClient's session because CHIRPS is
most conveniently accessed through GEE's catalog. CHIRPS itself is **not**
hosted on Microsoft Planetary Computer, so rather than keep a GEE
dependency alive for this one variable, we source
`precipitation_total` (used by the climate sub-index and the LSTM
drought-forecasting model) from NASA POWER's public REST API instead --
zero credentials, zero rate-limit headaches, free forever.
"""
from __future__ import annotations

from datetime import date, timedelta

import requests

from geovision.config import REGION_REGISTRY
from geovision.ingestion.base import BaseIngestionClient, IngestionUnavailableError, Observation

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


class PrecipitationClient(BaseIngestionClient):
    source_name = "nasa_power"

    def is_configured(self) -> bool:
        # Public, no-key REST API -- always "configured" in the sense that
        # no credentials are needed. If the live request fails (e.g. no
        # outbound network, as in this sandbox), fetch() raises
        # IngestionUnavailableError and the pipeline falls back to
        # synthetic data, same as every other source.
        return True

    def fetch(self, region_id: str, start: date, end: date) -> list[Observation]:
        region = REGION_REGISTRY.get(region_id)
        if region is None:
            raise ValueError(f"Unknown region_id: {region_id}")

        params = {
            "parameters": "PRECTOTCORR",
            "community": "AG",
            "longitude": region["lon"],
            "latitude": region["lat"],
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "format": "JSON",
        }
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise IngestionUnavailableError(f"NASA POWER request failed: {exc}") from exc

        daily = (
            resp.json().get("properties", {}).get("parameter", {}).get("PRECTOTCORR", {})
        )
        # Return one Observation per day so the feature matrix gets daily
        # precipitation values (not a single aggregated midpoint point).
        # NASA POWER uses -999 as a fill value for missing days.
        observations = []
        for date_str, v in daily.items():
            if v is None or v < 0:
                continue  # skip fill values
            try:
                obs_date = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
            except (ValueError, IndexError):
                continue
            observations.append(
                Observation(
                    region_id=region_id,
                    date=obs_date,
                    source="nasa_power",
                    variable="precipitation_total",
                    value=float(v),
                    unit="mm",
                    quality_flag="ok",
                )
            )
        if not observations:
            raise IngestionUnavailableError("NASA POWER returned no precipitation data.")
        return observations
