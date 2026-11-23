#!/usr/bin/env python3
"""
End-to-end demo: runs every stage of the GeoVision AI methodology for a
handful of regions and writes a single JSON snapshot the static dashboard
can load without a running API server (`dashboard/demo_data.json`).

    python scripts/run_demo.py
    python scripts/run_demo.py --regions IN,US,BR,MUM-IN --days 240

Stages exercised: (1) ingestion [real data: Open-Meteo + NASA MODIS --
see geovision/ingestion/] -> (2) preprocessing -> (3) feature
engineering -> (4)/(5) parallel ML/DL training + validation ->
(6) Environmental Health Index fusion -> (8) alerting. Stage 7
(presentation) is the JSON this script writes plus dashboard/index.html.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geovision.alerting.alerts import generate_alerts
from geovision.config import REGION_REGISTRY
from geovision.features.spectral_indices import build_feature_matrix
from geovision.index.health_index import compare_regions, compute_ehi
from geovision.ingestion.pipeline import IngestionPipeline
from geovision.models.train import best_model_summary, run_pooled_training
from geovision.preprocessing.cloud_mask import run_preprocessing

DEFAULT_REGIONS = list(REGION_REGISTRY)
TRAIN_DAYS = 730  # ~2 years so the forecasters see full seasonal cycles


def build_region(pipeline: IngestionPipeline, region_id: str, days: int):
    end = date.today()
    start = end - timedelta(days=days)
    raw = pipeline.run(region_id, start, end)
    clean = run_preprocessing(raw)
    features = build_feature_matrix(clean)
    ehi = compute_ehi(features) if not features.empty else features
    modis = raw[(raw["source"] == "modis")]
    ehi.attrs["ndvi_asof"] = max(m["composite"] for m in modis["meta"]) if len(modis) else None
    return ehi


def _work(region_id: str, train_days: int):
    """One process per region: fetch ~2 years of real data and build the EHI/feature frame."""
    time.sleep(random.uniform(0, 20))  # stagger requests: the free APIs rate-limit bursts (HTTP 429)
    t0 = time.time()
    ehi = build_region(IngestionPipeline(), region_id, train_days)
    return region_id, ehi, time.time() - t0


def _clean(obj):
    """NaN/Inf -> None so the snapshot is valid JSON for the browser."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", default=",".join(DEFAULT_REGIONS),
                        help="Comma-separated region ids from geovision.config.REGION_REGISTRY")
    parser.add_argument("--days", type=int, default=180, help="History window shown in the dashboard trend")
    parser.add_argument("--train-days", type=int, default=TRAIN_DAYS, help="History used to train the models")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for the deep-learning forecasters")
    parser.add_argument("--horizon", type=int, default=14, help="Forecast horizon, days")
    parser.add_argument("--workers", type=int, default=4, help="Regions fetched in parallel")
    parser.add_argument(
        "--out", default=str(Path(__file__).resolve().parent.parent / "dashboard" / "demo_data.json")
    )
    args = parser.parse_args()

    region_ids = [r.strip() for r in args.regions.split(",") if r.strip()]
    unknown = [r for r in region_ids if r not in REGION_REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown region ids: {unknown}")

    os.environ["GEO_N_JOBS"] = "1"  # inherited by the worker processes
    from geovision.models.forecasting import get_device
    device = get_device()
    print(f"Fetching {len(region_ids)} regions ({args.workers} parallel workers), then POOLED training on "
          f"device={device}: {args.train_days} days, {args.epochs} epochs, horizon {args.horizon}", flush=True)

    t_start = time.time()
    frames, timings = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_work, rid, args.train_days): rid for rid in region_ids}
        for f in as_completed(futures):
            rid = futures[f]
            try:
                _, ehi, t_fetch = f.result()
            except Exception as exc:  # one failing region must not lose the rest
                print(f"  [{rid}] FAILED: {type(exc).__name__}: {exc}", flush=True)
                continue
            frames[rid] = ehi
            timings[rid] = {"fetch_s": round(t_fetch, 1)}
            print(f"  [{rid}] fetched in {t_fetch:.0f}s, {len(ehi)} rows", flush=True)
    t_fetched = time.time()

    usable = {rid: f for rid, f in frames.items() if not f.empty}
    print(f"Pooled training on {len(usable)} regions...", flush=True)
    reports = run_pooled_training(usable, forecast_horizon=args.horizon, epochs=args.epochs)
    reports = {rid: reports.get(rid, {}) for rid in frames}
    wall = time.time() - t_start
    print(f"Fetch {t_fetched - t_start:.0f}s + pooled training {time.time() - t_fetched:.0f}s = {wall:.0f}s", flush=True)

    region_ids = [r for r in region_ids if r in frames]  # keep regions that succeeded
    ehi_by_region = {rid: frames[rid].tail(args.days + 1) for rid in region_ids}

    comparison = compare_regions(ehi_by_region)
    import pandas as pd
    combined = pd.concat([df for df in ehi_by_region.values() if not df.empty], ignore_index=True)
    alerts = generate_alerts(combined)
    print(f"{len(alerts)} alert(s) generated")

    # -- assemble the dashboard snapshot -----------------------------------
    def series_records(df):
        cols = ["date", "region_id", "ehi", "air_quality", "vegetation", "water", "climate"]
        out = df[[c for c in cols if c in df.columns]].copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        return out.to_dict(orient="records")

    snapshot = {
        "generated_at": date.today().isoformat(),
        "regions": [{"region_id": rid, **REGION_REGISTRY[rid], "ndvi_asof": frames[rid].attrs.get("ndvi_asof")}
                    for rid in region_ids],
        "ehi_series": {rid: series_records(df) for rid, df in ehi_by_region.items() if not df.empty},
        "comparison": json.loads(comparison.assign(
            date=comparison["date"].dt.strftime("%Y-%m-%d")
        )[["region_id", "date", "ehi", "air_quality", "vegetation", "water", "climate"]]
          .to_json(orient="records")) if not comparison.empty else [],
        "alerts": [
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
        ],
        "training_by_region": {
            rid: {"report": reports[rid], "best_models": best_model_summary(reports[rid]) if reports[rid] else {},
                  **timings[rid]} for rid in region_ids},
        "training_meta": {"device": device, "train_days": args.train_days, "epochs": args.epochs,
                          "horizon": args.horizon, "mode": "pooled", "seeds": 3,
                          "fetch_seconds": round(t_fetched - t_start), "train_seconds": round(time.time() - t_fetched),
                          "wall_seconds": round(wall), "trained_on": date.today().isoformat()},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_clean(snapshot), indent=2, default=str, allow_nan=False))
    print(f"Wrote dashboard snapshot -> {out_path}")


if __name__ == "__main__":
    main()
