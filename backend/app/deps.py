"""Dépendances FastAPI partagées par les routers."""
from __future__ import annotations

from fastapi import Request

from backend.app.services.twin_engine import TwinEngine


def get_twin_engine(request: Request) -> TwinEngine:
    return request.app.state.twin_engine
