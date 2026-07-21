from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.schemas import TwinState

logger = logging.getLogger("refinery.ws")

router = APIRouter(tags=["realtime"])


@router.websocket("/ws/realtime")
async def realtime_ws(websocket: WebSocket) -> None:
    engine = websocket.app.state.twin_engine
    await websocket.accept()
    queue = engine.subscribe()
    try:
        if engine.latest_state is not None:
            state = engine.latest_state
            payload = TwinState(
                timestamp=state["timestamp"], latency_ms=state["latency_ms"], sensors=state["sensors"],
                equipment=state["equipment"], yields_stream=state["yields_stream"],
                active_alerts_count=state["active_alerts_count"],
            )
            await websocket.send_text(payload.model_dump_json())
        while True:
            state = await queue.get()
            payload = TwinState(
                timestamp=state["timestamp"], latency_ms=state["latency_ms"], sensors=state["sensors"],
                equipment=state["equipment"], yields_stream=state["yields_stream"],
                active_alerts_count=state["active_alerts_count"],
            )
            await websocket.send_text(payload.model_dump_json())
    except WebSocketDisconnect:
        logger.info("Client WebSocket déconnecté")
    finally:
        engine.unsubscribe(queue)
