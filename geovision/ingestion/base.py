"""Common interface every ingestion client implements, so the pipeline
(geovision.ingestion.pipeline) can treat Planetary Computer, ERA5, NASA
POWER and OpenAQ uniformly."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class Observation:
    """One normalized observation record produced by any ingestion client.

    This is the common currency the preprocessing and feature-engineering
    stages operate on, regardless of which upstream API produced it.
    """

    region_id: str
    date: date
    source: str            # e.g. "sentinel2", "era5", "chirps", "openaq"
    variable: str           # e.g. "ndvi", "temperature_2m", "precipitation", "pm25"
    value: float
    unit: str
    quality_flag: str = "ok"   # "ok" | "cloud_masked" | "interpolated" | "missing"
    meta: dict[str, Any] | None = None


class BaseIngestionClient(abc.ABC):
    """Abstract base class for a data-source client."""

    source_name: str = "base"

    @abc.abstractmethod
    def is_configured(self) -> bool:
        """Whether credentials/config needed to hit the live API are present."""
        raise NotImplementedError

    @abc.abstractmethod
    def fetch(self, region_id: str, start: date, end: date) -> list[Observation]:
        """Fetch observations for a region between start and end (inclusive).

        Implementations should raise `IngestionUnavailableError` if the
        live API cannot be reached rather than silently returning nothing,
        so the pipeline can decide whether to fall back to synthetic data.
        """
        raise NotImplementedError


class IngestionUnavailableError(RuntimeError):
    """Raised when a live upstream API is unreachable or unconfigured."""
