from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.deps import get_twin_engine
from backend.app.schemas import FoulingStatus
from backend.app.services.twin_engine import TwinEngine

router = APIRouter(prefix="/api/fouling", tags=["fouling"])


@router.get("/status", response_model=FoulingStatus)
def get_fouling_status(engine: TwinEngine = Depends(get_twin_engine)) -> FoulingStatus:
    return FoulingStatus(**engine.fouling_status())
