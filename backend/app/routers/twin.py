from __future__ import annotations

import torch
from fastapi import APIRouter, Depends, HTTPException

from backend.app.deps import get_twin_engine
from backend.app.schemas import SensorDetail, SensorLimits, SensorTrend, TwinState
from backend.app.services.twin_engine import TwinEngine
from src.realtime_monitor import SENSOR_DEFS

router = APIRouter(prefix="/api/twin", tags=["twin"])


@router.get("/state", response_model=TwinState)
def get_twin_state(engine: TwinEngine = Depends(get_twin_engine)) -> TwinState:
    state = engine.latest_state or engine.monitor.step()
    return TwinState(
        timestamp=state["timestamp"], latency_ms=state["latency_ms"], sensors=state["sensors"],
        equipment=state["equipment"], yields_stream=state["yields_stream"],
        active_alerts_count=state["active_alerts_count"],
    )


@router.get("/sensor/{sensor_id}", response_model=SensorDetail)
def get_sensor_detail(sensor_id: str, engine: TwinEngine = Depends(get_twin_engine)) -> SensorDetail:
    sdef = next((s for s in SENSOR_DEFS if s["id"] == sensor_id), None)
    if sdef is None:
        raise HTTPException(status_code=404, detail="Capteur inconnu")

    col = sdef["column"]
    series = engine.replay_df[col].iloc[-24 * 7:]
    measured = series.tolist()

    if col == "preheat_outlet_temp" and engine.fouling_detector.available:
        feat_f = engine.fouling_detector.registry.fouling.feature_names
        window_f = engine.cfg["preprocessing"]["fouling_window_hours"]
        idx = feat_f.index("preheat_outlet_temp")
        scaler = engine.fouling_detector.registry.fouling.scaler
        device = engine.fouling_detector.registry.device
        predicted = []
        vals = engine.replay_df[feat_f].values.astype("float32")
        start = len(vals) - len(series)
        with torch.no_grad():
            for i in range(start, len(vals)):
                if i < window_f:
                    predicted.append(measured[i - start])
                    continue
                w = vals[i - window_f:i][None, ...]
                ws = scaler.transform(w.reshape(-1, w.shape[-1])).reshape(w.shape)
                t = torch.tensor(ws, dtype=torch.float32, device=device)
                predicted.append(float(engine.fouling_detector.registry.fouling.model(t).cpu().numpy()[0, 0]))
    else:
        predicted = list(measured)

    lo, hi = engine.replay_df[col].quantile(0.01), engine.replay_df[col].quantile(0.99)
    alerts = [a for a in engine.alert_engine.log_alerts(50) if a["equipment"] == sdef["equipment"]]

    return SensorDetail(
        id=sensor_id, name=sdef["name"], unit=sdef["unit"],
        trend=SensorTrend(timestamps=list(series.index), measured=measured, predicted=predicted),
        limits=SensorLimits(low=float(lo), high=float(hi)), alerts=alerts,
    )
