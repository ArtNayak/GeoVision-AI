from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from geovision.api import service
from geovision.config import REGION_REGISTRY, settings

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/train")
def get_training_report(
    region_id: str = Query(..., description="Region id, e.g. IN, MUM-IN"),
    days: int = Query(730, ge=90, le=1000),
    horizon: int = Query(settings.forecast_horizon_days, ge=3, le=60),
    epochs: int = Query(40, ge=3, le=200),
):
    """Runs the RF/XGBoost classifiers and the CNN/LSTM/Transformer
    forecasters for `region_id` and returns comparative validation
    metrics (Stage 4/5). This is intentionally synchronous and cached --
    for a production deployment, move this behind a task queue and poll
    a job-status endpoint instead of blocking the request."""
    if region_id not in REGION_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown region_id: {region_id}")
    report = service.training_report(region_id, days=days, horizon=horizon, epochs=epochs)
    return {"region_id": region_id, "report": report}
