"""
Ingestion pipeline -- Stage 1. Fetches real data for a region/date-range from
the free sources (Open-Meteo: temperature, precipitation, soil moisture, PM2.5,
NO2; NASA ORNL MODIS: NDVI) and returns one tidy, de-duplicated DataFrame.

There is deliberately NO synthetic fallback: if a source is down the error
propagates, so the dashboard can never show fabricated numbers as real.
(`synthetic.py` is kept only as a test fixture.)
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from geovision.ingestion.base import Observation
from geovision.ingestion.modis_ndvi_client import ModisNdviClient
from geovision.ingestion.open_meteo_client import OpenMeteoClient

COLUMNS = ["region_id", "date", "source", "variable", "value", "unit", "quality_flag"]


class IngestionPipeline:
    def __init__(self) -> None:
        self._clients = (OpenMeteoClient(), ModisNdviClient())

    def run(self, region_id: str, start: date, end: date) -> pd.DataFrame:
        observations: list[Observation] = []
        for client in self._clients:
            observations.extend(client.fetch(region_id, start, end))
        return self._to_dataframe(observations)

    @staticmethod
    def _to_dataframe(observations: list[Observation]) -> pd.DataFrame:
        if not observations:
            return pd.DataFrame(columns=COLUMNS)
        df = pd.DataFrame([o.__dict__ for o in observations])
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset=["region_id", "date", "source", "variable"])
        return df.sort_values(["variable", "date"]).reset_index(drop=True)
