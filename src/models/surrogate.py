"""Réseau surrogate pour l'optimisation énergétique : approxime
(conditions, COT, reflux) -> (4 rendements, énergie spécifique) de façon différentiable,
pour permettre une optimisation par descente de gradient sur les entrées (COT, reflux)."""
from __future__ import annotations

import torch
import torch.nn as nn


class SurrogateMLP(nn.Module):
    """MLP profond servant de modèle différentiable du procédé (remplace un bilan physique
    complet). Une fois entraîné et gelé, ses gradients par rapport aux entrées guident
    l'optimisation des points de consigne (COT, reflux)."""

    def __init__(self, input_dim: int, hidden_sizes: list[int] | None = None,
                 output_dim: int = 5, dropout: float = 0.3):
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 128, 64]
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
