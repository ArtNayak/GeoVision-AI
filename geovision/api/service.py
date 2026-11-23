"""
Service layer tying every architecture layer together for the API:
ingestion -> preprocessing -> feature engineering -> EHI fusion -> alerting,
plus on-demand model training/forecasting. A tiny in-memory TTL cache keeps
the interactive dashboard responsive without hammering the (synthetic, in
this environment) ingestion clients on every request.
"""
from __future__ import annotations

import math
import time
from datetime import date, timedelta
from typing import Any

import pandas as pd

import pickle
import re
from pathlib import Path

from geovision.alerting.alerts import Alert, generate_alerts
from geovision.config import CACHE_DIR, REGION_REGISTRY, settings
from geovision.features.spectral_indices import build_feature_matrix
from geovision.index.health_index import compare_regions, compute_ehi, latest_snapshot
from geovision.ingestion.pipeline import IngestionPipeline

_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL_SECONDS = settings.ingestion_interval_minutes * 60
_pipeline = IngestionPipeline()


def _disk_cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", key)
    return CACHE_DIR / f"{safe_name}.pkl"


def _cached(key: str, builder):
    now = time.time()
    # 1. Check in-memory cache
    hit = _CACHE.get(key)
    if hit and now - hit[0] < _CACHE_TTL_SECONDS:
        return hit[1]

    # 2. Check persistent disk cache (survives server restarts)
    disk_file = _disk_cache_path(key)
    if disk_file.exists():
        try:
            mtime = disk_file.stat().st_mtime
            if now - mtime < _CACHE_TTL_SECONDS:
                with open(disk_file, "rb") as f:
                    disk_val = pickle.load(f)
                _CACHE[key] = (mtime, disk_val)
                return disk_val
        except Exception:
            pass  # Fall back to builder on corrupted file

    # 3. Compute and store in both memory and disk
    value = builder()
    _CACHE[key] = (now, value)
    try:
        tmp_file = disk_file.with_suffix(".tmp")
        with open(tmp_file, "wb") as f:
            pickle.dump(value, f)
        tmp_file.replace(disk_file)
    except Exception:
        pass

    return value


def clear_all_caches() -> int:
    """Clears both memory and disk caches."""
    count = len(_CACHE)
    _CACHE.clear()
    if CACHE_DIR.exists():
        for p in [*CACHE_DIR.glob("*.pkl"), *(CACHE_DIR / "http").glob("*.json")]:  # results + raw API responses
            try:
                p.unlink()
                count += 1
            except Exception:
                pass
    return count


def _sanitize(obj):
    """Recursively replace float NaN / Inf with None so FastAPI's JSON
    encoder never hits 'Out of range float values are not JSON compliant'.
    pandas to_dict() always emits float('nan') from float64 columns even
    after DataFrame.where(), so we fix it at the Python-dict level instead."""
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def list_regions() -> list[dict]:
    return [{"region_id": rid, **meta} for rid, meta in REGION_REGISTRY.items()]


def region_ehi_dataframe(region_id: str, days: int = 180) -> pd.DataFrame:
    """Full per-region pipeline: ingest -> features -> EHI, cached per
    (region_id, days, date-of-request) so repeated dashboard polling
    within the ingestion interval is served from memory."""
    cache_key = f"ehi:{region_id}:{days}:{date.today().isoformat()}"

    def build() -> pd.DataFrame:
        end = date.today()
        start = end - timedelta(days=days)
        raw = _pipeline.run(region_id, start, end)
        from geovision.preprocessing.cloud_mask import run_preprocessing
        clean = run_preprocessing(raw)
        features = build_feature_matrix(clean)
        if features.empty:
            return features
        return compute_ehi(features)

    return _cached(cache_key, build)


def ehi_series(region_id: str, days: int = 180) -> list[dict]:
    df = region_ehi_dataframe(region_id, days=days)
    if df.empty:
        return []
    cols = ["date", "region_id", "ehi", "air_quality", "vegetation", "water", "climate"]
    out = df[[c for c in cols if c in df.columns]].copy()
    # Guard: date may already be a string-typed column after a cache hit
    if not pd.api.types.is_string_dtype(out["date"]):
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return _sanitize(out.to_dict(orient="records"))


def comparison_table(region_ids: list[str], days: int = 180) -> list[dict]:
    by_region = {rid: region_ehi_dataframe(rid, days=days) for rid in region_ids}
    table = compare_regions(by_region)
    if table.empty:
        return []
    cols = ["region_id", "date", "ehi", "air_quality", "vegetation", "water", "climate"]
    out = table[[c for c in cols if c in table.columns]].copy()
    # Guard: date may already be a string-typed column after a cache hit
    if not pd.api.types.is_string_dtype(out["date"]):
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["name"] = out["region_id"].map(lambda r: REGION_REGISTRY.get(r, {}).get("name", r))
    return _sanitize(out.to_dict(orient="records"))


def alerts_for(region_ids: list[str] | None = None, days: int = 90) -> list[dict]:
    ids = region_ids or list(REGION_REGISTRY.keys())
    frames = [region_ehi_dataframe(rid, days=days) for rid in ids]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return []
    combined = pd.concat(frames, ignore_index=True)
    alerts: list[Alert] = generate_alerts(combined)
    return [
        {
            "region_id": a.region_id,
            "region_name": REGION_REGISTRY.get(a.region_id, {}).get("name", a.region_id),
            "date": a.date.isoformat() if hasattr(a.date, "isoformat") else str(a.date),
            "severity": a.severity,
            "trigger": a.trigger,
            "indicator": a.indicator,
            "value": round(a.value, 2),
            "message": a.message,
        }
        for a in alerts
    ]


def training_report(region_id: str, days: int = 730, horizon: int | None = None,
                     epochs: int = 40) -> dict:
    """Runs Stage 4/5 (parallel ML/DL training + validation) for one
    region. Uses a longer default history than the EHI series endpoints
    since the forecasters need enough data for train/val splitting."""
    horizon = horizon or settings.forecast_horizon_days
    cache_key = f"train:{region_id}:{days}:{horizon}:{epochs}:{date.today().isoformat()}"

    def build() -> dict:
        end = date.today()
        start = end - timedelta(days=days)
        raw = _pipeline.run(region_id, start, end)
        from geovision.preprocessing.cloud_mask import run_preprocessing
        from geovision.models.train import run_training  # lazy: torch may not be installed
        clean = run_preprocessing(raw)
        features = build_feature_matrix(clean)
        if features.empty:
            return {"classification": {}, "forecasting": {}}
        return run_training(features, forecast_horizon=horizon, epochs=epochs)

    return _cached(cache_key, build)
