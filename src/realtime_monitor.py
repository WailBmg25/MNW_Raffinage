"""Moteur de rejeu temps réel (backend `twin_engine` ET notebook 06) : avance heure par
heure dans la table maîtresse, exécute les 3 réseaux en inférence continue (rendements,
fouling, qualité), calcule la santé des équipements et alimente le moteur d'alertes.
Mesure la latence bout-en-bout de chaque tick (objectif < 1 min, en pratique quelques ms)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.alert_system import AlertEngine
from src.config_loader import resolve_path
from src.energy_optimizer import EnergyOptimizer
from src.fouling_detector import FoulingDetector
from src import preprocessing as pp
from src.yield_model import YIELD_NAMES, YieldModel

SENSOR_DEFS = [
    {"id": "FI-101", "name": "Débit charge", "column": "feed_rate", "unit": "m³/h", "equipment": "column"},
    {"id": "TI-201", "name": "COT four", "column": "furnace_cot", "unit": "°C", "equipment": "furnace"},
    {"id": "TI-202", "name": "Température tête colonne", "column": "column_top_temp", "unit": "°C", "equipment": "column"},
    {"id": "PI-201", "name": "Pression tête colonne", "column": "column_top_pressure", "unit": "bar", "equipment": "column"},
    {"id": "FI-202", "name": "Taux de reflux", "column": "reflux_ratio", "unit": "-", "equipment": "column"},
    {"id": "FI-203", "name": "Vapeur de stripping", "column": "stripping_steam", "unit": "t/h", "equipment": "column"},
    {"id": "TI-301", "name": "Sortie train préchauffe", "column": "preheat_outlet_temp", "unit": "°C", "equipment": "preheat_train"},
    {"id": "FI-301", "name": "Débit gaz combustible", "column": "fuel_gas_flow", "unit": "t/h", "equipment": "furnace"},
    {"id": "TI-401", "name": "Température peau tubes", "column": "tube_metal_temp", "unit": "°C", "equipment": "cracker"},
    {"id": "FI-401", "name": "Charge naphta vapocraqueur", "column": "naphtha_feed", "unit": "t/h", "equipment": "cracker"},
]

EQUIPMENT_NAMES = {
    "preheat_train": "Train de préchauffe (E-101/102/103)",
    "furnace": "Four F-101",
    "column": "Colonne CDU C-101",
    "cracker": "Vapocraqueur",
}


def precompute_yield_predictions(replay_df: pd.DataFrame, yield_model: YieldModel, window: int) -> np.ndarray | None:
    """Prédiction batchée sur toute la ligne temporelle (une seule passe forward), pour que
    les endpoints d'historique n'aient pas à attendre l'accumulation de ticks live."""
    if not yield_model.available:
        return None
    feat = yield_model.registry.yields.feature_names
    values = replay_df[feat].values.astype(np.float32)
    n = len(values)
    X = np.stack([values[i - window:i] for i in range(window, n)])
    scaler = yield_model.registry.yields.scaler
    Xs = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    import torch
    device = yield_model.registry.device
    with torch.no_grad():
        preds = []
        batch_size = 2048
        for i in range(0, len(Xs), batch_size):
            t = torch.tensor(Xs[i:i + batch_size], dtype=torch.float32, device=device)
            preds.append(yield_model.registry.yields.model(t).cpu().numpy())
    out = np.concatenate(preds, axis=0)
    out = np.clip(out, 0, None)
    out = out / out.sum(axis=1, keepdims=True)
    return out  # aligné sur replay_df.index[window:]


def precompute_fouling_scores(replay_df: pd.DataFrame, fouling_detector: FoulingDetector, window: int) -> np.ndarray | None:
    """Idem pour le score de fouling (résidu GRU ou erreur de reconstruction AE/VAE)."""
    if not fouling_detector.available:
        return None
    feat = fouling_detector.registry.fouling.feature_names
    values = replay_df[feat].values.astype(np.float32)
    n = len(values)
    scores = np.empty(n - window, dtype=np.float32)
    for idx, i in enumerate(range(window, n)):
        scores[idx] = fouling_detector.score(values[i - window:i][None, ...])
    return scores  # aligné sur replay_df.index[window:]


def build_replay_table(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    master, hidden_df = pp.build_master_table(cfg)
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    cracker = pd.read_csv(Path(raw_dir) / "cracker_data.csv", parse_dates=["timestamp"]).set_index("timestamp")
    replay = master.join(cracker[["naphtha_feed", "coil_outlet_temp", "tube_metal_temp", "coke_thickness"]],
                          how="inner")
    hidden_df = hidden_df.reindex(replay.index)
    return replay, hidden_df


class RealtimeMonitor:
    def __init__(self, cfg: dict, yield_model: YieldModel, fouling_detector: FoulingDetector,
                 energy_optimizer: EnergyOptimizer, alert_engine: AlertEngine,
                 replay_df: pd.DataFrame | None = None, hidden_df: pd.DataFrame | None = None):
        self.cfg = cfg
        self.yield_model = yield_model
        self.fouling_detector = fouling_detector
        self.energy_optimizer = energy_optimizer
        self.alert_engine = alert_engine

        if replay_df is None:
            replay_df, hidden_df = build_replay_table(cfg)
        self.replay_df = replay_df
        self.hidden_df = hidden_df

        self.window_y = cfg["preprocessing"]["yield_window_hours"]
        self.window_f = cfg["preprocessing"]["fouling_window_hours"]
        self.start_pos = max(self.window_y, self.window_f)
        self.pos = self.start_pos
        self.fouling_score_history: list[float] = []
        self.specific_energy_baseline = float(replay_df["specific_energy"].iloc[: self.start_pos].mean())

    def _sensor_status(self, col: str, value: float) -> str:
        series = self.replay_df[col]
        lo, hi = series.quantile(0.01), series.quantile(0.99)
        span = max(hi - lo, 1e-9)
        if value < lo - 0.15 * span or value > hi + 0.15 * span:
            return "alarm"
        if value < lo or value > hi:
            return "warning"
        return "ok"

    def _equipment_health(self, fouling_days: float | None, furnace_cot: float, tube_metal_temp: float) -> dict[str, str]:
        cdu_cfg = self.cfg["data_generator"]["cdu"]
        cracker_cfg = self.cfg["data_generator"]["cracker"]
        alerts_cfg = self.cfg["alerts"]

        if fouling_days is None:
            preheat_health = "ok"
        elif fouling_days <= alerts_cfg["fouling_critical_days"]:
            preheat_health = "alarm"
        elif fouling_days <= alerts_cfg["fouling_warning_days"]:
            preheat_health = "warning"
        else:
            preheat_health = "ok"

        margin = 3.0
        if furnace_cot <= cdu_cfg["furnace_cot_min"] + margin or furnace_cot >= cdu_cfg["furnace_cot_max"] - margin:
            furnace_health = "warning"
        else:
            furnace_health = "ok"

        base_tmt = cracker_cfg["tube_metal_temp_base"]
        if tube_metal_temp >= base_tmt + 40:
            cracker_health = "alarm"
        elif tube_metal_temp >= base_tmt + 25:
            cracker_health = "warning"
        else:
            cracker_health = "ok"

        return {"preheat_train": preheat_health, "furnace": furnace_health, "column": "ok", "cracker": cracker_health}

    def step(self) -> dict:
        t0 = time.perf_counter()
        row = self.replay_df.iloc[self.pos]
        ts = self.replay_df.index[self.pos]

        yield_pred = None
        if self.yield_model.available:
            feat = self.yield_model.registry.yields.feature_names
            window = self.replay_df[feat].iloc[self.pos - self.window_y:self.pos].values[None, ...]
            yield_pred = self.yield_model.predict_window(window)
        yield_actual = row[YIELD_NAMES].values.astype(float)

        fouling_score = None
        fouling_index = 0.0
        trend, days_to_cleaning = "stable", None
        if self.fouling_detector.available:
            feat_f = self.fouling_detector.registry.fouling.feature_names
            window_f = self.replay_df[feat_f].iloc[self.pos - self.window_f:self.pos].values[None, ...]
            fouling_score = self.fouling_detector.score(window_f)
            fouling_index = self.fouling_detector.normalized_index(fouling_score)
            self.fouling_score_history.append(fouling_score)
            trend, days_to_cleaning = self.fouling_detector.trend_and_days_to_cleaning(
                self.fouling_score_history[-48:])

        roll = self.replay_df[YIELD_NAMES].iloc[max(0, self.pos - 168):self.pos]
        yield_roll_mean = roll.mean().values
        yield_roll_std = roll.std().values

        active_alerts = self.alert_engine.evaluate_tick(
            ts, fouling_index=fouling_index, fouling_days_to_cleaning=days_to_cleaning,
            yield_pred=yield_pred, yield_actual=yield_actual,
            yield_roll_mean=yield_roll_mean, yield_roll_std=yield_roll_std,
            specific_energy=float(row["specific_energy"]), specific_energy_baseline=self.specific_energy_baseline,
        )

        equipment_health = self._equipment_health(days_to_cleaning, float(row["furnace_cot"]),
                                                    float(row["tube_metal_temp"]))

        sensors = []
        for sdef in SENSOR_DEFS:
            col = sdef["column"]
            history = self.replay_df[col].iloc[max(0, self.pos - 24):self.pos]
            value = float(row[col])
            sensors.append({
                "id": sdef["id"], "name": sdef["name"], "value": value, "unit": sdef["unit"],
                "status": self._sensor_status(col, value), "sparkline": history.round(3).tolist(),
            })

        equipment = [{"id": k, "name": EQUIPMENT_NAMES[k], "health": v} for k, v in equipment_health.items()]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        yields_stream = yield_pred if yield_pred is not None else yield_actual

        self.pos += 1
        if self.pos >= len(self.replay_df):
            self.pos = self.start_pos  # boucle infinie sur le jeu de test rejoué

        return {
            "timestamp": ts, "latency_ms": latency_ms, "sensors": sensors, "equipment": equipment,
            "yields_stream": {"naphtha": float(yields_stream[0]), "kerosene": float(yields_stream[1]),
                               "gasoil": float(yields_stream[2]), "residue": float(yields_stream[3])},
            "active_alerts_count": len(active_alerts),
            "fouling_index": fouling_index, "fouling_trend": trend, "fouling_days_to_cleaning": days_to_cleaning,
            "specific_energy": float(row["specific_energy"]), "feed_rate": float(row["feed_rate"]),
            "yield_actual": yield_actual, "yield_pred": yield_pred,
        }
