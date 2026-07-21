"""Chargement unique des artefacts (.pt + scalers) au démarrage, exposition de predict_*().

Fonctionne en **mode dégradé explicite** si un artefact est absent (ne plante pas l'API,
signale juste `models_loaded[...] = False` sur /api/health et renvoie des valeurs de repli
là où c'est pertinent).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import torch

from src.models.autoencoders import Conv1DAutoencoder, DenseAutoencoder, LSTMAutoencoder, VAE
from src.models.cnn import CNN1DRegressor, CNNLSTMRegressor, TCNRegressor
from src.models.mlp import MLPRegressor
from src.models.rnn import RNNRegressor
from src.models.surrogate import SurrogateMLP
from src.models.transformer import TransformerRegressor
from src.seed_utils import get_device

logger = logging.getLogger("refinery.model_registry")

MODEL_CLASSES = {
    "mlp": MLPRegressor,
    "rnn": RNNRegressor, "lstm": RNNRegressor, "gru": RNNRegressor, "bilstm": RNNRegressor,
    "gru_residual": RNNRegressor,
    "cnn1d": CNN1DRegressor, "tcn": TCNRegressor, "cnnlstm": CNNLSTMRegressor,
    "transformer": TransformerRegressor,
    "dense_ae": DenseAutoencoder, "conv_ae": Conv1DAutoencoder, "lstm_ae": LSTMAutoencoder, "vae": VAE,
    "surrogate": SurrogateMLP,
}


def _load_checkpoint_as_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = dict(ckpt["config"])
    model_type = config.pop("model_type")
    cls = MODEL_CLASSES[model_type]
    model = cls(**config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt.get("scaler_path")


class ArtifactBundle:
    """Regroupe modèle + scaler (X, et éventuellement y) + noms de features pour une tâche donnée."""

    def __init__(self, model, scaler, feature_names: list[str] | None, scaler_y=None):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.scaler_y = scaler_y

    @property
    def available(self) -> bool:
        return self.model is not None


class ModelRegistry:
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.device = get_device()

        self.yields = self._try_load_bundle("yields_best.pt", "yields_scaler_X.joblib", "yields_feature_names.joblib")
        self.yields_meta = self._try_load_json("yields_best_model.json")

        self.fouling_production_meta = self._try_load_json("fouling_production_model.json")
        fouling_ckpt_name = (self.fouling_production_meta or {}).get("checkpoint", "fouling_gru_residual.pt")
        self.fouling = self._try_load_bundle(fouling_ckpt_name, "fouling_scaler_X.joblib", "fouling_feature_names.joblib")
        self.fouling_thresholds = self._try_load_json("fouling_thresholds.json") or {}

        self.quality_production_meta = self._try_load_json("quality_production_model.json")
        quality_ckpt_name = (self.quality_production_meta or {}).get("checkpoint", "quality_soft_sensor.pt")
        self.quality = self._try_load_bundle(quality_ckpt_name, "quality_scaler_X.joblib", "quality_feature_names.joblib",
                                              y_scaler_file="quality_scaler_y.joblib")

        self.surrogate = self._try_load_bundle("surrogate_energy.pt", "surrogate_scaler_X.joblib",
                                                "surrogate_feature_names.joblib")

        logger.info("ModelRegistry chargé (device=%s) : yields=%s fouling=%s quality=%s surrogate=%s",
                    self.device, self.yields.available, self.fouling.available,
                    self.quality.available, self.surrogate.available)

    def _try_load_bundle(self, model_file: str, scaler_file: str, feature_names_file: str,
                          y_scaler_file: str | None = None) -> ArtifactBundle:
        model_path = self.artifacts_dir / model_file
        if not model_path.exists():
            logger.warning("Artefact manquant : %s (mode dégradé pour cette tâche)", model_path)
            return ArtifactBundle(None, None, None)
        try:
            model, _ = _load_checkpoint_as_model(model_path, self.device)
            scaler_path = self.artifacts_dir / scaler_file
            scaler = joblib.load(scaler_path) if scaler_path.exists() else None
            feat_path = self.artifacts_dir / feature_names_file
            feature_names = joblib.load(feat_path) if feat_path.exists() else None
            scaler_y = None
            if y_scaler_file is not None:
                y_scaler_path = self.artifacts_dir / y_scaler_file
                scaler_y = joblib.load(y_scaler_path) if y_scaler_path.exists() else None
            return ArtifactBundle(model, scaler, feature_names, scaler_y=scaler_y)
        except Exception:
            logger.exception("Échec du chargement de %s (mode dégradé pour cette tâche)", model_path)
            return ArtifactBundle(None, None, None)

    def _try_load_json(self, filename: str) -> dict | None:
        path = self.artifacts_dir / filename
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.exception("Échec de lecture de %s", path)
            return None

    def models_loaded_summary(self) -> dict[str, bool]:
        return {
            "yields": self.yields.available,
            "fouling": self.fouling.available,
            "quality": self.quality.available,
            "energy": self.surrogate.available,
        }

    # -------------------------------------------------------------------
    def predict_yields(self, X_window: np.ndarray) -> np.ndarray | None:
        """X_window : (1, 24, n_features) déjà dans l'ordre de yields_feature_names, NON scalé."""
        if not self.yields.available:
            return None
        Xs = self.yields.scaler.transform(X_window.reshape(-1, X_window.shape[-1])).reshape(X_window.shape)
        with torch.no_grad():
            t = torch.tensor(Xs, dtype=torch.float32, device=self.device)
            out = self.yields.model(t).cpu().numpy()[0]
        out = np.clip(out, 0, None)
        return out / out.sum()

    def fouling_anomaly_score(self, X_window: np.ndarray) -> float | None:
        """X_window : (1, 48, n_features) dans l'ordre de fouling_feature_names, NON scalé.
        Retourne l'erreur de reconstruction (AE/VAE) ou le résidu (GRU) — plus haut = plus de fouling."""
        if not self.fouling.available:
            return None
        Xs = self.fouling.scaler.transform(X_window.reshape(-1, X_window.shape[-1])).reshape(X_window.shape)
        model_type = (self.fouling_production_meta or {}).get("model_type", "gru_residual")
        with torch.no_grad():
            t = torch.tensor(Xs, dtype=torch.float32, device=self.device)
            if model_type == "gru_residual":
                idx = (self.fouling_production_meta or {}).get("preheat_outlet_temp_index")
                if idx is None and self.fouling.feature_names and "preheat_outlet_temp" in self.fouling.feature_names:
                    idx = self.fouling.feature_names.index("preheat_outlet_temp")
                pred = self.fouling.model(t).cpu().numpy()[0, 0]
                actual = X_window[0, -1, idx] if idx is not None else 0.0
                return float(actual - pred)
            if model_type == "vae":
                recon, _, _ = self.fouling.model(t)
                return float(torch.mean((recon - t) ** 2).item())
            recon = self.fouling.model(t)
            return float(torch.mean((recon - t) ** 2).item())

    def predict_surrogate(self, X_row: np.ndarray) -> np.ndarray | None:
        """X_row : (1, n_features) conditions instantanées, NON scalé. Retourne (5,) = 4 rendements + énergie."""
        if not self.surrogate.available:
            return None
        Xs = self.surrogate.scaler.transform(X_row)
        with torch.no_grad():
            t = torch.tensor(Xs, dtype=torch.float32, device=self.device)
            return self.surrogate.model(t).cpu().numpy()[0]

    def predict_quality(self, X_window: np.ndarray) -> np.ndarray | None:
        """Retourne les 5 cibles qualité en unités réelles (dé-standardisées via scaler_y
        si le modèle a été entraîné sur des cibles normalisées — cf. notebook 06)."""
        if not self.quality.available:
            return None
        Xs = self.quality.scaler.transform(X_window.reshape(-1, X_window.shape[-1])).reshape(X_window.shape)
        with torch.no_grad():
            t = torch.tensor(Xs, dtype=torch.float32, device=self.device)
            out = self.quality.model(t).cpu().numpy()[0]
        if self.quality.scaler_y is not None:
            out = self.quality.scaler_y.inverse_transform(out.reshape(1, -1))[0]
        return out
