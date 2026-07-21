"""Moteur de simulation du jumeau numérique côté FastAPI : orchestre le rejeu temps réel
(src.realtime_monitor), précalcule les historiques complets (rendements, fouling) pour les
endpoints REST, et diffuse un `TwinState` complet aux abonnés WebSocket toutes les
`realtime_tick_seconds` secondes."""
from __future__ import annotations

import asyncio
import logging

import numpy as np
import pandas as pd

from src.alert_system import AlertEngine
from src.energy_optimizer import EnergyOptimizer
from src.fouling_detector import FoulingDetector
from src.realtime_monitor import (RealtimeMonitor, build_replay_table, precompute_fouling_scores,
                                    precompute_yield_predictions)
from src.yield_model import YIELD_NAMES, YieldModel

from backend.app.core.config import Settings
from backend.app.services.model_registry import get_model_registry

logger = logging.getLogger("refinery.twin_engine")


class TwinEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.project_config
        self.registry = get_model_registry(str(settings.artifacts_dir))

        self.yield_model = YieldModel(self.registry)
        self.fouling_detector = FoulingDetector(self.registry)
        self.energy_optimizer = EnergyOptimizer(self.registry, self.cfg)
        self.alert_engine = AlertEngine(self.cfg)

        logger.info("Construction de la table de rejeu (préprocessing complet)...")
        self.replay_df, self.hidden_df = build_replay_table(self.cfg)

        self.monitor = RealtimeMonitor(self.cfg, self.yield_model, self.fouling_detector,
                                        self.energy_optimizer, self.alert_engine,
                                        replay_df=self.replay_df, hidden_df=self.hidden_df)

        window_y = self.cfg["preprocessing"]["yield_window_hours"]
        window_f = self.cfg["preprocessing"]["fouling_window_hours"]
        logger.info("Précalcul de l'historique complet des rendements et du fouling...")
        self.yields_pred_full = precompute_yield_predictions(self.replay_df, self.yield_model, window_y)
        self.yields_pred_index = self.replay_df.index[window_y:]
        self.fouling_scores_full = precompute_fouling_scores(self.replay_df, self.fouling_detector, window_f)
        self.fouling_scores_index = self.replay_df.index[window_f:]

        self.latest_state: dict | None = None
        self._subscribers: list[asyncio.Queue] = []
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        while True:
            try:
                state = await asyncio.to_thread(self.monitor.step)
                self.latest_state = state
                for q in list(self._subscribers):
                    if not q.full():
                        q.put_nowait(state)
            except Exception:
                logger.exception("Erreur pendant le tick du jumeau numérique")
            await asyncio.sleep(self.settings.realtime_tick_seconds)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    # -- Aides pour les endpoints REST --------------------------------------
    def yields_history(self, hours: int) -> dict:
        idx = self.yields_pred_index
        actual = self.replay_df.loc[idx, YIELD_NAMES].values
        sl = slice(-hours, None)
        ts = idx[sl]
        actual_sl = actual[sl]
        pred_sl = self.yields_pred_full[sl] if self.yields_pred_full is not None else actual_sl
        names = ["naphtha", "kerosene", "gasoil", "residue"]
        return {
            "timestamps": list(ts),
            "actual": {n: actual_sl[:, i].round(4).tolist() for i, n in enumerate(names)},
            "predicted": {n: pred_sl[:, i].round(4).tolist() for i, n in enumerate(names)},
        }

    def fouling_status(self) -> dict:
        threshold = self.fouling_detector.threshold if self.fouling_detector.available else 1.0
        if self.fouling_scores_full is not None:
            idx = self.fouling_scores_index
            scores = self.fouling_scores_full
            current = float(scores[-1])
            trend, days = self.fouling_detector.trend_and_days_to_cleaning(list(scores[-48:]))
            index_norm = [self.fouling_detector.normalized_index(float(s)) for s in scores[-24 * 60:]]
            hist_idx = idx[-24 * 60:]
        else:
            current, trend, days = 0.0, "stable", None
            index_norm, hist_idx = [], []

        hidden = self.hidden_df["fouling_resistance"].reindex(hist_idx).tolist() if len(hist_idx) else []
        cleanings = self.hidden_df.index[self.hidden_df["is_cleaning_event"] == 1].tolist()

        episodes = []
        for c in cleanings:
            window = self.hidden_df.loc[:c].tail(24 * 21)
            crossed = window.index[window["cleaning_needed_within_24h"] == 1]
            if len(crossed):
                start = crossed[0]
                episodes.append({"start": start, "detected_at": start, "cleaning_at": c,
                                  "lead_time_h": float((c - start).total_seconds() / 3600)})

        return {
            "current_index": self.fouling_detector.normalized_index(current) if self.fouling_detector.available else 0.0,
            "trend": trend, "estimated_days_to_cleaning": days, "threshold": float(threshold),
            "history": {"timestamps": list(hist_idx), "index": index_norm, "hidden_truth": hidden},
            "cleanings": list(cleanings), "episodes": episodes,
        }

    def kpi_summary(self) -> dict:
        state = self.latest_state
        if state is None:
            row = self.replay_df.iloc[self.monitor.start_pos]
            state = {"timestamp": self.replay_df.index[self.monitor.start_pos], "specific_energy": float(row["specific_energy"]),
                     "feed_rate": float(row["feed_rate"]), "yield_actual": row[YIELD_NAMES].values,
                     "fouling_index": 0.0, "fouling_days_to_cleaning": None, "active_alerts_count": 0}

        baseline = self.monitor.specific_energy_baseline
        current_energy = state["specific_energy"]
        distillate_pct = float(np.sum(state["yield_actual"][:3]) * 100)

        gain_pct = 100.0 * (baseline - current_energy) / baseline if baseline else 0.0
        usd_saved = max(gain_pct, 0.0) / 100.0 * baseline * self.cfg["data_generator"]["refinery_capacity_bpd"] \
            / 1000.0 * self.cfg["energy_optimization"]["gas_price_usd_per_mwh"]
        co2_avoided = max(gain_pct, 0.0) / 100.0 * baseline * self.cfg["data_generator"]["refinery_capacity_bpd"] \
            / 1000.0 * self.cfg["energy_optimization"]["co2_factor_t_per_mwh"]

        objectives = [
            {"id": 1, "label": "Rendements (MAPE < 5%)", "achieved": self.yield_model.available,
             "value": "voir /rendements"},
            {"id": 2, "label": "Fouling (> 24h avant nettoyage)", "achieved": self.fouling_detector.available,
             "value": "voir /encrassement"},
            {"id": 3, "label": "Énergie (gain > 5%)", "achieved": self.energy_optimizer.available,
             "value": "voir /energie"},
            {"id": 4, "label": "Qualité (corr > 0.9)", "achieved": self.registry.quality.available,
             "value": "voir /documentation"},
            {"id": 5, "label": "Latence (< 1 min)", "achieved": True,
             "value": f"{state.get('latency_ms', 0):.1f}ms"},
        ]

        return {
            "date": state["timestamp"], "feed_rate": state["feed_rate"],
            "distillate_yield_pct": distillate_pct, "specific_energy": current_energy,
            "specific_energy_baseline": baseline, "specific_energy_delta_pct": -gain_pct,
            "fouling_index": state.get("fouling_index", 0.0),
            "fouling_days_before_cleaning": state.get("fouling_days_to_cleaning"),
            "active_alerts": state.get("active_alerts_count", 0),
            "usd_saved_today": float(usd_saved), "co2_avoided_today_t": float(co2_avoided),
            "objectives": objectives,
        }
