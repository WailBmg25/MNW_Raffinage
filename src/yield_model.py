"""Wrapper haut niveau pour la prédiction des rendements (utilisé par le backend ET le
notebook 06) : construit la fenêtre d'entrée à partir de conditions "what-if" et appelle
le modèle de production chargé par `ModelRegistry`."""
from __future__ import annotations

import numpy as np

from src.model_registry import ModelRegistry

YIELD_NAMES = ["naphtha_yield", "kerosene_yield", "gasoil_yield", "residue_yield"]

RAW_CONDITION_COLS = ["feed_rate", "furnace_cot", "reflux_ratio", "stripping_steam",
                      "column_top_temp", "column_top_pressure"]
CRUDE_ONEHOT_COLS = {"leger": "crude_leger", "moyen": "crude_moyen", "lourd": "crude_lourd"}


class YieldModel:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    @property
    def available(self) -> bool:
        return self.registry.yields.available

    def predict_window(self, X_window: np.ndarray) -> np.ndarray | None:
        """X_window : (1, 24, n_features) dans l'ordre yields_feature_names, non scalé."""
        return self.registry.predict_yields(X_window)

    def build_whatif_window(self, base_window: np.ndarray, conditions: dict) -> np.ndarray:
        """Copie `base_window` (1, 24, F) et remplace les colonnes de conditions brutes au
        dernier pas de temps par les valeurs "what-if" fournies (simulateur de scénario).
        Les features dérivées (lags/rolling/deltas) restent celles de l'historique réel :
        approximation pragmatique pour un simulateur interactif temps réel."""
        feature_names = self.registry.yields.feature_names
        window = base_window.copy()
        for col in RAW_CONDITION_COLS:
            if col in conditions and conditions[col] is not None and col in feature_names:
                idx = feature_names.index(col)
                window[0, -1, idx] = conditions[col]
        crude_type = conditions.get("crude_type")
        if crude_type in CRUDE_ONEHOT_COLS:
            for ctype, col in CRUDE_ONEHOT_COLS.items():
                if col in feature_names:
                    window[0, -1, feature_names.index(col)] = 1.0 if ctype == crude_type else 0.0
        return window

    def predict_whatif(self, base_window: np.ndarray, conditions: dict) -> np.ndarray | None:
        if not self.available:
            return None
        window = self.build_whatif_window(base_window, conditions)
        return self.predict_window(window)
