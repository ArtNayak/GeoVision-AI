import pytest
from fastapi.testclient import TestClient

from geovision.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_regions():
    r = client.get("/api/regions")
    assert r.status_code == 200
    body = r.json()
    assert len(body) > 0
    assert {"region_id", "name", "level"}.issubset(body[0].keys())


def test_ehi_series_for_known_region():
    r = client.get("/api/ehi", params={"region_id": "IN", "days": 60})
    assert r.status_code == 200
    body = r.json()
    assert body["region_id"] == "IN"
    assert len(body["series"]) > 0
    assert "ehi" in body["series"][0]


def test_ehi_series_unknown_region_404():
    r = client.get("/api/ehi", params={"region_id": "NOPE", "days": 60})
    assert r.status_code == 404


def test_compare_endpoint():
    r = client.get("/api/compare", params={"region_ids": "IN,US,BR", "days": 60})
    assert r.status_code == 200
    body = r.json()
    assert len(body["table"]) == 3


def test_compare_unknown_region_404():
    r = client.get("/api/compare", params={"region_ids": "IN,NOPE", "days": 60})
    assert r.status_code == 404


def test_alerts_endpoint():
    r = client.get("/api/alerts", params={"days": 60})
    assert r.status_code == 200
    assert "alerts" in r.json()


@pytest.mark.slow
def test_train_endpoint_runs_full_pipeline():
    r = client.get("/api/train", params={"region_id": "IN", "days": 150, "horizon": 5, "epochs": 3})
    assert r.status_code == 200
    report = r.json()["report"]
    assert "classification" in report and "forecasting" in report
