from datetime import date, timedelta

import pandas as pd
import pytest

from geovision.features.spectral_indices import build_feature_matrix, to_wide
from geovision.index.health_index import compute_ehi, compare_regions
from geovision.ingestion.pipeline import IngestionPipeline
from geovision.ingestion.synthetic import SyntheticClient
from geovision.preprocessing.cloud_mask import flag_out_of_range, run_preprocessing


@pytest.fixture(scope="module")
def raw_observations():
    client = SyntheticClient()
    end = date.today()
    start = end - timedelta(days=90)
    obs = client.fetch_all("IN", start, end)
    return pd.DataFrame([o.__dict__ for o in obs])


def test_synthetic_client_covers_all_core_variables(raw_observations):
    expected = {"ndvi", "soil_moisture", "temperature_2m", "precipitation_total", "pm25", "no2"}
    assert expected.issubset(set(raw_observations["variable"].unique()))


def test_pipeline_fetches_real_sources():
    pipeline = IngestionPipeline()
    end = date.today()
    start = end - timedelta(days=30)
    df = pipeline.run("US", start, end)
    assert not df.empty
    assert {"region_id", "date", "source", "variable", "value", "unit", "quality_flag"}.issubset(df.columns)
    assert (df["region_id"] == "US").all()


def test_unknown_region_raises():
    pipeline = IngestionPipeline()
    with pytest.raises(ValueError):
        pipeline.run("NOT-A-REGION", date.today() - timedelta(days=5), date.today())


def test_preprocessing_flags_out_of_range_values(raw_observations):
    df = raw_observations.copy()
    bad_idx = df[df["variable"] == "ndvi"].index[0]
    df.loc[bad_idx, "value"] = 5.0  # NDVI can't exceed 1.0
    flagged = flag_out_of_range(df)
    assert flagged.loc[bad_idx, "quality_flag"] == "invalid"
    assert pd.isna(flagged.loc[bad_idx, "value"])


def test_preprocessing_fills_gaps(raw_observations):
    df = raw_observations.copy()
    na_idx = df[df["variable"] == "pm25"].index[3]
    df.loc[na_idx, "value"] = None
    cleaned = run_preprocessing(df)
    assert cleaned["value"].isna().sum() == 0


def test_feature_matrix_has_core_columns_and_pca(raw_observations):
    cleaned = run_preprocessing(raw_observations)
    features = build_feature_matrix(cleaned)
    assert not features.empty
    for col in ["ndvi", "pm25", "precipitation_total", "pca_1", "ndvi_anomaly"]:
        assert col in features.columns
    # one row per date (no duplicated region/date pairs)
    wide = to_wide(cleaned)
    assert wide.groupby(["region_id", "date"]).size().max() == 1


def test_ehi_fusion_produces_bounded_composite(raw_observations):
    cleaned = run_preprocessing(raw_observations)
    features = build_feature_matrix(cleaned)
    ehi = compute_ehi(features)
    assert "ehi" in ehi.columns
    valid = ehi["ehi"].dropna()
    assert not valid.empty
    assert (valid >= 0).all() and (valid <= 100).all()


def test_compare_regions_ranks_by_ehi_desc():
    pipeline = IngestionPipeline()
    end = date.today()
    start = end - timedelta(days=60)
    frames = {}
    for region in ["IN", "AU"]:
        raw = pipeline.run(region, start, end)
        clean = run_preprocessing(raw)
        features = build_feature_matrix(clean)
        frames[region] = compute_ehi(features)
    table = compare_regions(frames)
    assert list(table["region_id"]) == list(table.sort_values("ehi", ascending=False)["region_id"])


def test_sample_points_are_inside_the_region_bbox():
    from geovision.config import REGION_REGISTRY
    from geovision.ingestion.base import sample_points
    for region in REGION_REGISTRY.values():
        pts = sample_points(region)
        assert len(pts) == 5 and pts[0] == (region["lat"], region["lon"])
        x0, y0, x1, y1 = region["bbox"]
        assert all(y0 <= la <= y1 and x0 <= lo <= x1 for la, lo in pts[1:])
