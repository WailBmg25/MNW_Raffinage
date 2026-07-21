"""Configuration du backend : lecture de config.yaml (racine du projet) + variables d'environnement."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from src.config_loader import load_config as _load_project_config

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings:
    def __init__(self) -> None:
        self.project_config = _load_project_config(PROJECT_ROOT / "config.yaml")
        self.artifacts_dir = Path(
            os.getenv("MODELS_ARTIFACTS_DIR", PROJECT_ROOT / self.project_config["paths"]["backend_artifacts_dir"])
        )
        self.data_processed_dir = Path(
            os.getenv("DATA_PROCESSED_DIR", PROJECT_ROOT / self.project_config["paths"]["processed_dir"])
        )
        self.data_raw_dir = Path(
            os.getenv("DATA_RAW_DIR", PROJECT_ROOT / self.project_config["paths"]["raw_dir"])
        )
        self.cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
        self.realtime_tick_seconds = float(os.getenv(
            "REALTIME_TICK_SECONDS", self.project_config["realtime"]["tick_seconds"]
        ))
        self.refinery_name = os.getenv("REFINERY_NAME", "Raffinerie MNW — 200 000 bbl/j")


@lru_cache
def get_settings() -> Settings:
    return Settings()
