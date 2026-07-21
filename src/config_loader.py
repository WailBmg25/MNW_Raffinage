"""Chargement centralisé de config.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Charge le fichier config.yaml à la racine du projet."""
    cfg_path = Path(path) if path is not None else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str | Path) -> Path:
    """Résout un chemin relatif défini dans config.yaml par rapport à la racine du projet."""
    p = Path(relative_path)
    return p if p.is_absolute() else PROJECT_ROOT / p
