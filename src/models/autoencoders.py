"""Modèles non supervisés pour la détection d'anomalies (encrassement) :
Autoencodeur dense, Autoencodeur convolutionnel 1D, Autoencodeur LSTM (seq2seq), VAE.

Tous sont entraînés sur des séquences "propres" (post-nettoyage) ; l'erreur de
reconstruction sur de nouvelles séquences sert de score d'anomalie (dérive = encrassement).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DenseAutoencoder(nn.Module):
    """Autoencodeur dense sur fenêtre aplatie : encodeur/décodeur symétriques."""

    def __init__(self, input_dim: int, hidden_sizes: list[int] | None = None, dropout: float = 0.2):
        super().__init__()
        hidden_sizes = hidden_sizes or [64, 16]
        enc_layers = []
        prev = input_dim
        for h in hidden_sizes:
            enc_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        prev = hidden_sizes[-1]
        for h in reversed(hidden_sizes[:-1]):
            dec_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        z = self.encoder(x)
        out = self.decoder(z)
        return out.reshape(original_shape)


class Conv1DAutoencoder(nn.Module):
    """Autoencodeur convolutionnel 1D : encodeur par convolutions/pooling, décodeur symétrique
    par convolutions transposées. Capture les motifs locaux de la séquence temporelle."""

    def __init__(self, input_size: int, channels: list[int] | None = None,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        channels = channels or [32, 16]
        enc = []
        prev = input_size
        for ch in channels:
            enc += [nn.Conv1d(prev, ch, kernel_size, padding=kernel_size // 2, stride=2),
                    nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(dropout)]
            prev = ch
        self.encoder = nn.Sequential(*enc)

        dec = []
        rev_channels = list(reversed(channels[:-1]))
        for ch in rev_channels:
            dec += [nn.ConvTranspose1d(prev, ch, kernel_size, padding=kernel_size // 2,
                                        stride=2, output_padding=1),
                    nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(dropout)]
            prev = ch
        # dernière couche : pas de BatchNorm/ReLU (reconstruction non bornée)
        dec.append(nn.ConvTranspose1d(prev, input_size, kernel_size, padding=kernel_size // 2,
                                       stride=2, output_padding=1))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, input_size)
        seq_len = x.size(1)
        h = x.transpose(1, 2)
        z = self.encoder(h)
        out = self.decoder(z)
        out = out[:, :, :seq_len]  # ajustement de longueur (arrondis stride/pooling)
        if out.size(2) < seq_len:
            out = nn.functional.pad(out, (0, seq_len - out.size(2)))
        return out.transpose(1, 2)


class LSTMAutoencoder(nn.Module):
    """Autoencodeur LSTM séquence-à-séquence : un encodeur LSTM résume la fenêtre en un
    vecteur latent, un décodeur LSTM reconstruit la séquence à partir de ce vecteur répété."""

    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                                batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        _, (h_n, _) = self.encoder(x)
        latent = h_n[-1]  # (batch, hidden_size) : dernier état caché de l'encodeur
        latent_seq = latent.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(self.dropout(latent_seq))
        return self.output_proj(decoded)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)
        return h_n[-1]


class VAE(nn.Module):
    """Autoencodeur variationnel (dense) : encodeur -> (mu, logvar) -> reparamétrisation
    -> décodeur. La régularisation KL structure l'espace latent (utile pour la visualisation
    2D et une détection d'anomalie plus robuste que l'AE déterministe)."""

    def __init__(self, input_dim: int, hidden_size: int = 32, latent_dim: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, input_dim),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z).reshape(original_shape)
        return recon, mu, logvar

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        return self.fc_mu(self.encoder(x))


def vae_loss(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor, kl_weight: float = 1e-3) -> torch.Tensor:
    """ELBO négatif : erreur de reconstruction (MSE) + divergence KL pondérée (beta-VAE)."""
    recon_loss = nn.functional.mse_loss(recon, target, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl
