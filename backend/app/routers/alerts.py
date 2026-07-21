from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.deps import get_twin_engine
from backend.app.schemas import Alert
from backend.app.services.alert_engine import get_active_alerts, get_alert_log
from backend.app.services.twin_engine import TwinEngine

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("", response_model=list[Alert])
def list_alerts(limit: int = 50, engine: TwinEngine = Depends(get_twin_engine)) -> list[Alert]:
    return [Alert(**a) for a in get_alert_log(engine, limit)]


@router.get("/active", response_model=list[Alert])
def list_active_alerts(engine: TwinEngine = Depends(get_twin_engine)) -> list[Alert]:
    return [Alert(**a) for a in get_active_alerts(engine)]
