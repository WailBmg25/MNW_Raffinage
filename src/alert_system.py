"""Moteur d'alertes à base de règles (backend + notebook 06) : fouling, dérive de rendement
(> 2σ), qualité hors spécification, anomalie énergétique. Niveaux info/warning/critical,
anti-rebond (une condition doit persister `debounce_ticks` pas consécutifs avant de
déclencher OU de se lever), journal en mémoire."""
from __future__ import annotations

import itertools
from datetime import datetime
from typing import Literal

AlertLevel = Literal["info", "warning", "critical"]
AlertType = Literal["fouling", "yield_drift", "quality", "energy"]


class AlertEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg["alerts"]
        self._counters: dict[str, int] = {}
        self._active: dict[str, dict] = {}
        self._log: list[dict] = []
        self._id_counter = itertools.count(1)

    def _debounce(self, key: str, triggered: bool) -> bool:
        """Retourne True seulement quand `triggered` est vrai depuis `debounce_ticks` pas consécutifs."""
        if triggered:
            self._counters[key] = self._counters.get(key, 0) + 1
        else:
            self._counters[key] = 0
        return self._counters[key] >= self.cfg["debounce_ticks"]

    def _emit(self, timestamp: datetime, key: str, level: AlertLevel, alert_type: AlertType,
              equipment: str, message: str, value: float, recommendation: str) -> dict:
        alert = {
            "id": f"a{next(self._id_counter)}", "timestamp": timestamp, "level": level,
            "type": alert_type, "equipment": equipment, "message": message,
            "value": value, "recommendation": recommendation, "active": True,
        }
        if key not in self._active:
            self._log.append(alert)
        self._active[key] = alert
        return alert

    def _clear(self, key: str) -> None:
        if key in self._active:
            self._active[key]["active"] = False
            del self._active[key]

    def evaluate_tick(self, timestamp: datetime, *, fouling_index: float | None,
                       fouling_days_to_cleaning: float | None, yield_pred, yield_actual,
                       yield_roll_mean, yield_roll_std, quality_pred=None, quality_limits=None,
                       specific_energy: float | None = None, specific_energy_baseline: float | None = None
                       ) -> list[dict]:
        # --- Fouling ---
        if fouling_days_to_cleaning is not None:
            critical = fouling_days_to_cleaning <= self.cfg["fouling_critical_days"]
            warning = fouling_days_to_cleaning <= self.cfg["fouling_warning_days"]
            if self._debounce("fouling_critical", critical):
                self._emit(timestamp, "fouling", "critical", "fouling", "preheat_train",
                           f"Nettoyage requis sous {fouling_days_to_cleaning:.1f} jours",
                           fouling_index or 0.0, "Planifier un nettoyage en urgence")
            elif self._debounce("fouling_warning", warning):
                self._emit(timestamp, "fouling", "warning", "fouling", "preheat_train",
                           f"Encrassement en hausse, nettoyage estimé dans {fouling_days_to_cleaning:.1f} jours",
                           fouling_index or 0.0, "Planifier un nettoyage sous 2 semaines")
            elif not warning and self._debounce("fouling_ok", True):
                self._clear("fouling")

        # --- Dérive de rendement (> 2σ) ---
        names = ["naphtha", "kerosene", "gasoil", "residue"]
        for i, name in enumerate(names):
            if yield_roll_std[i] <= 1e-9:
                continue
            z = abs(yield_actual[i] - yield_roll_mean[i]) / yield_roll_std[i]
            key = f"yield_drift_{name}"
            triggered = z > self.cfg["yield_drift_sigma"]
            if self._debounce(key, triggered):
                level = "critical" if z > 3 * self.cfg["yield_drift_sigma"] else "warning"
                self._emit(timestamp, key, level, "yield_drift", "column",
                           f"Dérive du rendement {name} ({z:.1f}σ)", float(yield_actual[i]),
                           "Vérifier les conditions opératoires de la colonne")
            elif self._debounce(key + "_ok", not triggered):
                self._clear(key)

        # --- Qualité hors spécification ---
        if quality_pred is not None and quality_limits is not None:
            for name, value in quality_pred.items():
                lo, hi = quality_limits.get(name, (None, None))
                out_of_spec = (lo is not None and value < lo) or (hi is not None and value > hi)
                key = f"quality_{name}"
                if self._debounce(key, out_of_spec):
                    self._emit(timestamp, key, "warning", "quality", "column",
                               f"Qualité hors spécification prédite : {name} = {value:.2f}",
                               float(value), "Ajuster les conditions opératoires avant le prochain prélèvement")
                elif self._debounce(key + "_ok", not out_of_spec):
                    self._clear(key)

        # --- Anomalie énergétique ---
        if specific_energy is not None and specific_energy_baseline:
            excess_pct = 100.0 * (specific_energy - specific_energy_baseline) / specific_energy_baseline
            triggered = excess_pct > self.cfg["energy_anomaly_pct"]
            if self._debounce("energy", triggered):
                self._emit(timestamp, "energy", "warning", "energy", "furnace",
                           f"Consommation énergétique anormale (+{excess_pct:.1f}%)",
                           float(specific_energy), "Vérifier le four et le train de préchauffe")
            elif self._debounce("energy_ok", not triggered):
                self._clear("energy")

        return self.active_alerts()

    def active_alerts(self) -> list[dict]:
        return list(self._active.values())

    def log_alerts(self, limit: int = 50) -> list[dict]:
        return list(reversed(self._log))[:limit]
