"""Modèles convolutionnels : CNN 1D, TCN (convolutions dilatées causales), hybride CNN-LSTM."""
from __future__ import annotations

import torch
import torch.nn as nn


class CNN1DRegressor(nn.Module):
    """CNN 1D classique : 3 blocs Conv-BatchNorm-ReLU-MaxPool puis tête dense.

    Les convolutions capturent des motifs locaux dans la fenêtre temporelle
    (ex. dynamique rapide d'un changement de brut) avant réduction par pooling.
    """

    def __init__(self, input_size: int, channels: list[int] | None = None,
                 kernel_size: int = 3, output_size: int = 4, dropout: float = 0.3):
        super().__init__()
        channels = channels or [32, 64, 64]
        blocks = []
        prev = input_size
        for ch in channels:
            blocks += [
                nn.Conv1d(prev, ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, ceil_mode=True),
                nn.Dropout(dropout),
            ]
            prev = ch
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(prev, prev // 2 if prev > 1 else prev),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prev // 2 if prev > 1 else prev, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, input_size) -> (batch, input_size, seq_len) pour Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class _Chomp1d(nn.Module):
    """Supprime le padding excédentaire à droite pour garantir la causalité stricte."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class _TemporalBlock(nn.Module):
    """Bloc résiduel TCN : deux convolutions causales dilatées + connexion résiduelle."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class TCNRegressor(nn.Module):
    """Temporal Convolutional Network : blocs résiduels à dilatation croissante (1,2,4,...).

    Champ réceptif exponentiel avec convolutions strictement causales (pas de fuite du futur),
    alternative aux RNN pour capturer des dépendances longues sans récurrence séquentielle.
    """

    def __init__(self, input_size: int, channels: list[int] | None = None,
                 kernel_size: int = 3, output_size: int = 4, dropout: float = 0.2):
        super().__init__()
        channels = channels or [32, 32, 64]
        layers = []
        prev = input_size
        for i, ch in enumerate(channels):
            layers.append(_TemporalBlock(prev, ch, kernel_size, dilation=2 ** i, dropout=dropout))
            prev = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(prev, prev // 2 if prev > 1 else prev),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prev // 2 if prev > 1 else prev, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # (batch, input_size, seq_len)
        out = self.tcn(x)
        last = out[:, :, -1]           # dernier pas de temps = résume tout le passé causal
        return self.head(last)


class CNNLSTMRegressor(nn.Module):
    """Modèle hybride : extracteur de motifs locaux (Conv1D) puis mémoire séquentielle (LSTM)."""

    def __init__(self, input_size: int, conv_channels: int = 32, kernel_size: int = 3,
                 lstm_hidden: int = 64, lstm_layers: int = 1, output_size: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(conv_channels, lstm_hidden, num_layers=lstm_layers,
                             batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)              # (batch, input_size, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)              # (batch, seq_len, conv_channels)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last))
