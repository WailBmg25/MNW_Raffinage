"""Baseline MLP (perceptron multicouche) — sur la dernière observation aplatie."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """MLP dense classique : Linear -> BatchNorm -> ReLU -> Dropout, empilés.

    Sert de baseline simple pour la prédiction des rendements : ne voit que
    l'observation courante (pas d'historique), contrairement aux modèles séquentiels.
    """

    def __init__(self, input_dim: int, hidden_sizes: list[int] | None = None,
                 output_dim: int = 4, dropout: float = 0.3):
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 64]
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, input_dim) — si une séquence (batch, seq_len, features) est fournie,
        # on ne garde que la dernière observation aplatie (rôle de baseline "sans mémoire" du MLP).
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.net(x)
