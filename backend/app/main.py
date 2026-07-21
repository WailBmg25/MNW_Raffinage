"""FastAPI — Jumeau numérique CDU & Vapocraqueur.

Charge les modèles au démarrage (lifespan), configure CORS, mesure la latence de chaque
requête (header `X-Process-Time-Ms`), et expose tous les endpoints REST + le WebSocket
`/ws/realtime` qui pousse un `TwinState` complet en continu."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import get_settings
from backend.app.routers import alerts, energy, fouling, kpi, realtime, twin, yields
from backend.app.schemas import HealthResponse
from backend.app.services.twin_engine import TwinEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("refinery.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Démarrage — chargement des artefacts depuis %s", settings.artifacts_dir)
    engine = TwinEngine(settings)
    app.state.twin_engine = engine
    app.state.settings = settings
    await engine.start()
    logger.info("Jumeau numérique démarré (tick=%ss)", settings.realtime_tick_seconds)
    yield
    await engine.stop()


app = FastAPI(title="Jumeau Numérique — Raffinage CDU & Vapocraqueur", lifespan=lifespan)

settings_for_cors = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings_for_cors.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


@app.get("/api/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    engine: TwinEngine = request.app.state.twin_engine
    summary = engine.registry.models_loaded_summary()
    status = "ok" if all(summary.values()) else "degraded"
    return HealthResponse(status=status, device=str(engine.registry.device),
                           models_loaded=summary, timestamp=datetime.now(timezone.utc))


app.include_router(kpi.router)
app.include_router(yields.router)
app.include_router(fouling.router)
app.include_router(energy.router)
app.include_router(alerts.router)
app.include_router(twin.router)
app.include_router(realtime.router)
