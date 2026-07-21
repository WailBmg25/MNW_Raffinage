"""Accès aux alertes depuis les routers (le moteur de règles lui-même vit dans
`src/alert_system.py` et est piloté à chaque tick par `TwinEngine`)."""
from __future__ import annotations

from backend.app.services.twin_engine import TwinEngine


def get_active_alerts(engine: TwinEngine) -> list[dict]:
    return engine.alert_engine.active_alerts()


def get_alert_log(engine: TwinEngine, limit: int = 50) -> list[dict]:
    return engine.alert_engine.log_alerts(limit)
