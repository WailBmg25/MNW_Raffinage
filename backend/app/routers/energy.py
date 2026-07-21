from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.app.deps import get_twin_engine
from backend.app.schemas import OperatingConditions, OptimizationResult
from backend.app.services.twin_engine import TwinEngine

router = APIRouter(prefix="/api/energy", tags=["energy"])


@router.post("/optimize", response_model=OptimizationResult)
def optimize_energy(conditions: OperatingConditions, engine: TwinEngine = Depends(get_twin_engine)) -> OptimizationResult:
    if not engine.energy_optimizer.available:
        raise HTTPException(status_code=503, detail="Surrogate énergétique non disponible (mode dégradé)")

    feature_names = engine.registry.surrogate.feature_names
    base_row = engine.replay_df.iloc[-1]
    X_row = engine.energy_optimizer.build_row(base_row, conditions.model_dump())

    cot_idx = feature_names.index("furnace_cot")
    reflux_idx = feature_names.index("reflux_ratio")
    result = engine.energy_optimizer.optimize(X_row, cot_idx, reflux_idx)
    return OptimizationResult(**result)
