from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from geovision.api import service
from geovision.config import REGION_REGISTRY

router = APIRouter(prefix="/api", tags=["environment"])


@router.get("/regions")
def get_regions():
    return service.list_regions()


@router.get("/ehi")
def get_ehi(
    region_id: str = Query(..., description="Region id, e.g. IN, MUM-IN, AS"),
    days: int = Query(180, ge=14, le=730),
):
    if region_id not in REGION_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown region_id: {region_id}")
    return {"region_id": region_id, "series": service.ehi_series(region_id, days=days)}


@router.get("/compare")
def get_comparison(
    region_ids: str = Query(..., description="Comma-separated region ids, e.g. IN,US,BR"),
    days: int = Query(180, ge=14, le=730),
):
    ids = [r.strip() for r in region_ids.split(",") if r.strip()]
    unknown = [r for r in ids if r not in REGION_REGISTRY]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Unknown region_id(s): {unknown}")
    return {"regions": ids, "table": service.comparison_table(ids, days=days)}
