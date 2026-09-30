"""
ERA5 (Copernicus Climate Data Store) ingestion client.

Provides 2m air temperature, total precipitation and 10m wind speed used by
the climate sub-index and the LSTM/Transformer forecasting models. Requires
`cdsapi` and a CDS_API_KEY. Falls back cleanly when unconfigured so the
pipeline can substitute synthetic data during development.
"""
from __future__ import annotations

from datetime import date

from geovision.config import settings, REGION_REGISTRY
from geovision.ingestion.base import BaseIngestionClient, IngestionUnavailableError, Observation

VARIABLES = {
    "2m_temperature": ("temperature_2m", "K"),
    "total_precipitation": ("precipitation", "m"),
    "10m_wind_speed": ("wind_speed_10m", "m/s"),
}


class ERA5Client(BaseIngestionClient):
    source_name = "era5"

    def is_configured(self) -> bool:
        return bool(settings.credentials.cds_api_key)

    def fetch(self, region_id: str, start: date, end: date) -> list[Observation]:
        if not self.is_configured():
            raise IngestionUnavailableError(
                "CDS_API_KEY not configured. See README.md for setup."
            )
        try:
            import cdsapi  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise IngestionUnavailableError(
                "cdsapi is not installed. `pip install cdsapi`."
            ) from exc

        region = REGION_REGISTRY.get(region_id)
        if region is None:
            raise ValueError(f"Unknown region_id: {region_id}")

        # A real implementation submits a CDS retrieval request (grib/netcdf),
        # downloads it, and reduces it with xarray over the region bbox.
        # That request/poll/download cycle is asynchronous and can take
        # minutes, so it is intentionally isolated here behind the same
        # `Observation` contract the rest of the pipeline expects -- see
        # `geovision.ingestion.pipeline.IngestionPipeline._reduce_era5_grib`
        # for the reduction step once the file lands.
        raise NotImplementedError(
            "Live ERA5 retrieval requires an async CDS request/download "
            "cycle -- wire this up to cdsapi.Client().retrieve(...) once "
            "credentials are configured. Use geovision.ingestion.synthetic "
            "for local development in the meantime."
        )
