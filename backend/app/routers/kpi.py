from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.deps import get_twin_engine
from backend.app.schemas import KpiSummary
from backend.app.services.twin_engine import TwinEngine

router = APIRouter(prefix="/api/kpi", tags=["kpi"])


@router.get("/summary", response_model=KpiSummary)
def get_kpi_summary(engine: TwinEngine = Depends(get_twin_engine)) -> KpiSummary:
    return KpiSummary(**engine.kpi_summary())
