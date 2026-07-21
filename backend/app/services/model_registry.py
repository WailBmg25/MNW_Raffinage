"""Point d'accès FastAPI au registre de modèles générique défini dans `src/model_registry.py`
(partagé avec le notebook 06). Un seul chargement au démarrage (lifespan), via un singleton
mis en cache par répertoire d'artefacts."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.model_registry import ModelRegistry


@lru_cache
def get_model_registry(artifacts_dir: str) -> ModelRegistry:
    return ModelRegistry(Path(artifacts_dir))
