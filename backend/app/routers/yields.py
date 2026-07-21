from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.app.deps import get_twin_engine
from backend.app.schemas import OperatingConditions, YieldHistory, YieldPrediction
from backend.app.services.twin_engine import TwinEngine

router = APIRouter(prefix="/api/yields", tags=["yields"])


@router.get("/history", response_model=YieldHistory)
def get_yields_history(hours: int = 168, engine: TwinEngine = Depends(get_twin_engine)) -> YieldHistory:
    return YieldHistory(**engine.yields_history(hours))


@router.post("/predict", response_model=YieldPrediction)
def predict_yields(conditions: OperatingConditions, engine: TwinEngine = Depends(get_twin_engine)) -> YieldPrediction:
    if not engine.yield_model.available:
        raise HTTPException(status_code=503, detail="Modèle de rendements non disponible (mode dégradé)")

    feat = engine.yield_model.registry.yields.feature_names
    window_y = engine.cfg["preprocessing"]["yield_window_hours"]
    base_window = engine.replay_df[feat].iloc[-window_y:].values[None, ...]

    pred = engine.yield_model.predict_whatif(base_window, conditions.model_dump())
    meta = engine.registry.yields_meta or {}
    return YieldPrediction(
        naphtha_yield=float(pred[0]), kerosene_yield=float(pred[1]),
        gasoil_yield=float(pred[2]), residue_yield=float(pred[3]),
        model_name=meta.get("model_type", "inconnu"), mape_test=float(meta.get("mape_global", 0.0)),
    )
