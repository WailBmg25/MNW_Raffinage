"""Wrapper haut niveau pour la détection de fouling (backend + notebook 06) : score
d'anomalie du modèle de production, comparaison au seuil, tendance et estimation du
nombre de jours avant nettoyage (projection linéaire simple de la tendance récente —
un utilitaire d'affichage, pas un modèle de prédiction : la détection elle-même reste
assurée par le réseau de neurones)."""
from __future__ import annotations

import numpy as np

from src.model_registry import ModelRegistry


class FoulingDetector:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.threshold = self._production_threshold()

    @property
    def available(self) -> bool:
        return self.registry.fouling.available

    def _production_threshold(self) -> float:
        meta = self.registry.fouling_production_meta or {}
        if "threshold" in meta:
            return float(meta["threshold"])
        method = meta.get("model_type", "gru_residual")
        thresholds = self.registry.fouling_thresholds or {}
        return float(thresholds.get(method, 1.0))

    def score(self, X_window: np.ndarray) -> float | None:
        return self.registry.fouling_anomaly_score(X_window)

    def is_alarm(self, current_score: float) -> bool:
        return current_score >= self.threshold

    def normalized_index(self, current_score: float) -> float:
        """Ramène le score brut à un indice ~[0,1] relatif au seuil, pour l'affichage (gauge)."""
        if self.threshold <= 0:
            return 0.0
        return float(np.clip(current_score / self.threshold, 0.0, 1.5) / 1.5)

    def trend_and_days_to_cleaning(self, recent_scores: list[float], hours_per_point: float = 1.0
                                    ) -> tuple[str, float | None]:
        """`recent_scores` : historique récent (le plus ancien en premier). Projection linéaire
        de la pente récente pour estimer le nombre de jours avant franchissement du seuil."""
        if len(recent_scores) < 2:
            return "stable", None
        y = np.array(recent_scores[-48:], dtype=float)
        x = np.arange(len(y)) * hours_per_point
        slope = float(np.polyfit(x, y, 1)[0])  # pente heure par heure (projection simple, pas un modèle)
        if abs(slope) < 1e-9:
            trend = "stable"
        else:
            trend = "up" if slope > 0 else "down"

        current = y[-1]
        if slope <= 0 or current >= self.threshold:
            return trend, 0.0 if current >= self.threshold else None
        hours_to_threshold = (self.threshold - current) / slope
        return trend, round(hours_to_threshold / 24.0, 1)
