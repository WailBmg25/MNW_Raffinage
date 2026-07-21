"""Transformer encoder pour séries temporelles : encodage positionnel + multi-head attention."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Encodage positionnel sinusoïdal fixe (Vaswani et al.) : injecte l'ordre temporel
    puisque l'attention seule est invariante par permutation des pas de temps."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerRegressor(nn.Module):
    """Encodeur Transformer : projection linéaire -> encodage positionnel -> N couches
    d'auto-attention multi-têtes (LayerNorm intégrée) -> pooling temporel -> tête dense.
    """

    def __init__(self, input_size: int, d_model: int = 64, n_heads: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128, output_size: int = 4,
                 dropout: float = 0.2, max_len: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, input_size)
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        h = self.norm(h)
        pooled = h.mean(dim=1)  # pooling temporel moyen sur toute la fenêtre encodée
        return self.head(self.dropout(pooled))
