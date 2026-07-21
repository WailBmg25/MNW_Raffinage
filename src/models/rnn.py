"""Modèles séquentiels récurrents : RNN simple, LSTM, GRU, LSTM bidirectionnel."""
from __future__ import annotations

import torch
import torch.nn as nn

_CELLS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


class RNNRegressor(nn.Module):
    """Encodeur récurrent générique (RNN/LSTM/GRU, uni ou bidirectionnel) + tête de régression.

    Le dernier état caché (concaténé sur les deux sens si bidirectionnel) résume
    l'historique de la fenêtre temporelle et alimente une tête dense pour prédire
    les rendements multi-cibles.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 output_size: int = 4, dropout: float = 0.3, cell_type: str = "lstm",
                 bidirectional: bool = False):
        super().__init__()
        cell_type = cell_type.lower()
        if cell_type not in _CELLS:
            raise ValueError(f"cell_type doit être parmi {list(_CELLS)}, reçu {cell_type}")
        cell_cls = _CELLS[cell_type]
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = cell_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                             batch_first=True, dropout=rnn_dropout, bidirectional=bidirectional)
        self.cell_type = cell_type
        n_dir = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * n_dir, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, input_size)
        out, _ = self.rnn(x)
        last = out[:, -1, :]  # dernier pas de temps (contient déjà les 2 sens si bidirectionnel)
        return self.head(self.dropout(last))
