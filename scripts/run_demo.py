#!/usr/bin/env python3
"""
End-to-end demo: runs every stage of the GeoVision AI methodology for a
handful of regions and writes a single JSON snapshot the static dashboard
can load without a running API server (`dashboard/demo_data.json`).

    python scripts/run_demo.py
    python scripts/run_demo.py --regions IN,US,BR,MUM-IN --days 240

Stages exercised: (1) ingestion [synthetic fallback -- see
geovision/ingestion/synthetic.py] -> (2) preprocessing -> (3) feature
engineering -> (4)/(5) parallel ML/DL training + validation ->
(6) Environmental Health Index fusion -> (8) alerting. Stage 7
(presentation) is the JSON this script writes plus dashboard/index.html.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geovision.alerting.alerts import generate_alerts
from geovision.config import REGION_REGISTRY
from geovision.features.spectral_indices import build_feature_matrix
from geovision.index.health_index import compare_regions, compute_ehi
from geovision.ingestion.pipeline import IngestionPipeline
from geovision.models.train import best_model_summary, run_training
from geovision.preprocessing.cloud_mask import run_preprocessing

DEFAULT_REGIONS = ["IN", "US", "BR", "AU", "KE", "MH-IN", "MUM-IN", "AS", "AF"]


def build_region(pipeline: IngestionPipeline, region_id: str, days: int):
    end = date.today()
    start = end - timedelta(days=days)
    raw = pipeline.run(region_id, start, end)
    clean = run_preprocessing(raw)
    features = build_feature_matrix(clean)
    ehi = compute_ehi(features) if not features.empty else features
    return ehi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", default=",".join(DEFAULT_REGIONS),
                         help="Comma-separated region ids from geovision.config.REGION_REGISTRY")
    parser.add_argument("--days", type=int, default=240, help="History window length")
    parser.add_argument("--train-region", default="IN",
                         help="Region to run the full ML/DL training comparison for")
    parser.add_argument("--epochs", type=int, default=25,
                         help="Training epochs for the deep-learning forecasters (kept low for a quick demo)")
    parser.add_argument("--horizon", type=int, default=14, help="Forecast horizon, days")
    parser.add_argument(
        "--out", default=str(Path(__file__).resolve().parent.parent / "dashboard" / "demo_data.json")
    )
    args = parser.parse_args()

    region_ids = [r.strip() for r in args.regions.split(",") if r.strip()]
    unknown = [r for r in region_ids if r not in REGION_REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown region ids: {unknown}")

    pipeline = IngestionPipeline()

    print(f"[1/4] Ingesting + preprocessing + feature engineering + EHI fusion "
          f"for {len(region_ids)} regions over {args.days} days...")
    ehi_by_region = {}
    for region_id in region_ids:
        ehi_by_region[region_id] = build_region(pipeline, region_id, args.days)
        print(f"      {region_id}: {len(ehi_by_region[region_id])} daily rows")

    print("[2/4] Building region comparison table...")
    comparison = compare_regions(ehi_by_region)

    print("[3/4] Generating alerts across all regions...")
    import pandas as pd
    combined = pd.concat([df for df in ehi_by_region.values() if not df.empty], ignore_index=True)
    alerts = generate_alerts(combined)
    print(f"      {len(alerts)} alert(s) generated")

    print(f"[4/4] Training ML/DL models for '{args.train_region}' "
          f"(RF, XGBoost, CNN, LSTM, Transformer)...")
    training_report = run_training(
        ehi_by_region[args.train_region], forecast_horizon=args.horizon, epochs=args.epochs
    )
    summary = best_model_summary(training_report)
    print(f"      Best models by task: {summary}")

    # -- assemble the dashboard snapshot -----------------------------------
    def series_records(df):
        cols = ["date", "region_id", "ehi", "air_quality", "vegetation", "water", "climate"]
        out = df[[c for c in cols if c in df.columns]].copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        return out.to_dict(orient="records")

    snapshot = {
        "generated_at": date.today().isoformat(),
        "regions": [{"region_id": rid, **REGION_REGISTRY[rid]} for rid in region_ids],
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
        "training": {"region_id": args.train_region, "report": training_report,
                     "best_models": summary},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2, default=str))
    print(f"\nWrote dashboard snapshot -> {out_path}")


if __name__ == "__main__":
    main()
