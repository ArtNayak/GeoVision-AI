from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from geovision.api import service
from geovision.config import REGION_REGISTRY

router = APIRouter(prefix="/api", tags=["alerts"])


@router.get("/alerts")
def get_alerts(
    region_ids: str | None = Query(
        None, description="Comma-separated region ids; omit for all regions"
    ),
    days: int = Query(180, ge=14, le=365),
):
    ids = None
    if region_ids:
        ids = [r.strip() for r in region_ids.split(",") if r.strip()]
        unknown = [r for r in ids if r not in REGION_REGISTRY]
        if unknown:
            raise HTTPException(status_code=404, detail=f"Unknown region_id(s): {unknown}")
    return {"alerts": service.alerts_for(ids, days=days)}
