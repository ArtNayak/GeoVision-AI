"""
Ingestion pipeline orchestrator -- Stage 1 of the methodology.

Fans out to the four data-source clients (Planetary Computer, ERA5, NASA
POWER precipitation, OpenAQ) for a region/date-range, falling back to the
synthetic generator per-source when a live API is unconfigured or
unreachable, and returns one combined, de-duplicated list of `Observation`
records ready for preprocessing.
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from geovision.ingestion.base import IngestionUnavailableError, Observation
from geovision.ingestion.era5_client import ERA5Client
from geovision.ingestion.openaq_client import OpenAQClient
from geovision.ingestion.planetary_computer_client import PlanetaryComputerClient
from geovision.ingestion.precipitation_client import PrecipitationClient
from geovision.ingestion.synthetic import SyntheticClient

logger = logging.getLogger(__name__)

# Each live client's `source_name` is a category, but the observations it
# (or its synthetic stand-in) produces are tagged with the specific
# collection/dataset they came from. This maps category -> tags so the
# per-source synthetic fallback pulls the right rows.
#
# NOTE: the synthetic generator still tags its rows "sentinel2" / "chirps"
# etc (see synthetic.py) since those tags describe *what the variable is*,
# not which live API produced it -- that mapping doesn't need to change
# just because the live source moved from GEE to Planetary Computer.
SOURCE_GROUPS: dict[str, set[str]] = {
    "planetary_computer": {"sentinel2", "landsat8", "modis"},
    "era5": {"era5"},
    "nasa_power": {"chirps"},
    "openaq": {"openaq"},
}


class IngestionPipeline:
    """Orchestrates near real-time ingestion across all data sources."""

    def __init__(self, use_synthetic_fallback: bool = True) -> None:
        self.use_synthetic_fallback = use_synthetic_fallback
        self._planetary_computer = PlanetaryComputerClient()
        self._era5 = ERA5Client()
        self._precipitation = PrecipitationClient()
        self._openaq = OpenAQClient()
        self._synthetic = SyntheticClient()

    def run(self, region_id: str, start: date, end: date) -> pd.DataFrame:
        """Fetch and combine all sources for a region/window into a tidy
        DataFrame with columns: region_id, date, source, variable, value,
        unit, quality_flag.
        """
        observations: list[Observation] = []

        for client in (self._planetary_computer, self._era5, self._precipitation, self._openaq):
            try:
                if not client.is_configured():
                    raise IngestionUnavailableError(f"{client.source_name} not configured")
                observations.extend(client.fetch(region_id, start, end))
            except (IngestionUnavailableError, NotImplementedError) as exc:
                logger.info("Falling back to synthetic data for %s: %s",
                            client.source_name, exc)
                if self.use_synthetic_fallback:
                    allowed_sources = SOURCE_GROUPS.get(client.source_name, {client.source_name})
                    observations.extend(
                        [o for o in self._synthetic.fetch_all(region_id, start, end)
                         if o.source in allowed_sources]
                    )

        # If nothing at all was live-configured, the per-source fallback
        # above still only fires per attempted source, so this catches the
        # (common, in this environment) fully-offline case in one shot.
        if not observations and self.use_synthetic_fallback:
            observations = self._synthetic.fetch_all(region_id, start, end)

        return self._to_dataframe(observations)

    @staticmethod
    def _to_dataframe(observations: list[Observation]) -> pd.DataFrame:
        if not observations:
            return pd.DataFrame(
                columns=["region_id", "date", "source", "variable", "value",
                         "unit", "quality_flag"]
            )
        df = pd.DataFrame([o.__dict__ for o in observations])
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset=["region_id", "date", "source", "variable"])
        return df.sort_values(["variable", "date"]).reset_index(drop=True)
