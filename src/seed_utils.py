"""Utilitaires de reproductibilité et de détection de device (partagés notebooks + backend)."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Fixe la graine aléatoire pour random, numpy et torch (CPU/CUDA) + déterminisme cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Détecte automatiquement le device disponible (tout doit pouvoir tourner sur CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
