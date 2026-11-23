"""Make the whole test-suite hermetic: no network, and no writes to the real data/cache.

The production pipeline deliberately has no synthetic fallback, so API tests that go through
it would hit Open-Meteo / MODIS. Here the service's pipeline is swapped for the synthetic
test fixture, and the cache is redirected to a temp dir so fake data never lands in the
cache the real dashboard reads.
"""
from datetime import date

import pytest

import geovision.api.service as service
import geovision.config as config
from geovision.ingestion.pipeline import IngestionPipeline
from geovision.ingestion.synthetic import SyntheticClient


class _OfflinePipeline:
    def run(self, region_id: str, start: date, end: date):
        observations = SyntheticClient().fetch_all(region_id, start, end)
        return IngestionPipeline._to_dataframe(observations)


@pytest.fixture(autouse=True)
def offline_pipeline_and_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "_pipeline", _OfflinePipeline())
    monkeypatch.setattr(service, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    service._CACHE.clear()
    yield
    service._CACHE.clear()
