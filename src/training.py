"""Boucle d'entraînement générique réutilisée par tous les notebooks (03-06).

Implémente les exigences obligatoires de la spécification (Partie A.3) :
tqdm par epoch, early stopping (patience 10, restauration des meilleurs poids),
gradient clipping (max_norm=1.0), scheduler (ReduceLROnPlateau ou CosineAnnealing),
AdamW par défaut, historique complet pour tracer les courbes d'apprentissage.
"""
from __future__ import annotations

import copy
import io
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class TrainConfig:
    """Hyperparamètres d'entraînement, affichés explicitement dans chaque notebook."""
    epochs_max: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    grad_clip_max_norm: float = 1.0
    scheduler_type: str = "plateau"     # "plateau" ou "cosine"
    optimizer_type: str = "adamw"       # "adamw", "adam" ou "sgd"
    momentum: float = 0.9               # utilisé seulement pour SGD
    verbose: bool = True


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_epoch: int = 0
    training_time_s: float = 0.0
    n_params_total: int = 0
    n_params_trainable: int = 0
    model_size_mb: float = 0.0


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Retourne (nb total de paramètres, nb de paramètres entraînables)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model: nn.Module) -> float:
    """Taille approximative du modèle sérialisé (Mo), via un buffer en mémoire."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if cfg.optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if cfg.optimizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
    raise ValueError(f"optimizer_type inconnu : {cfg.optimizer_type}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    if cfg.scheduler_type == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(3, cfg.patience // 3))
    if cfg.scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs_max)
    raise ValueError(f"scheduler_type inconnu : {cfg.scheduler_type}")


def _run_epoch(model: nn.Module, loader: DataLoader, loss_fn: Callable, device: torch.device,
               optimizer: torch.optim.Optimizer | None, grad_clip_max_norm: float | None) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_samples = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if is_train:
                optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if is_train:
                loss.backward()
                if grad_clip_max_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
                optimizer.step()
            bs = xb.size(0)
            total_loss += loss.item() * bs
            n_samples += bs
    return total_loss / max(n_samples, 1)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 cfg: TrainConfig, loss_fn: Callable | None = None,
                 device: torch.device | None = None,
                 model_name: str = "model") -> tuple[nn.Module, TrainHistory]:
    """Entraîne `model` avec early stopping, gradient clipping et scheduler.

    Retourne le modèle restauré à ses meilleurs poids (val loss minimale) et l'historique complet.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = loss_fn or nn.MSELoss()
    model = model.to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    history = TrainHistory()
    history.n_params_total, history.n_params_trainable = count_parameters(model)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    start_time = time.time()

    pbar = tqdm(range(1, cfg.epochs_max + 1), desc=f"Entraînement {model_name}", disable=not cfg.verbose)
    for epoch in pbar:
        train_loss = _run_epoch(model, train_loader, loss_fn, device, optimizer, cfg.grad_clip_max_norm)
        val_loss = _run_epoch(model, val_loader, loss_fn, device, None, None)

        current_lr = optimizer.param_groups[0]["lr"]
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.lr.append(current_lr)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        pbar.set_postfix({"train_loss": f"{train_loss:.5f}", "val_loss": f"{val_loss:.5f}", "lr": f"{current_lr:.2e}"})

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            history.best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.patience:
            history.stopped_epoch = epoch
            if cfg.verbose:
                pbar.write(f"Early stopping à l'epoch {epoch} (meilleure epoch : {history.best_epoch}, "
                            f"val_loss={best_val_loss:.5f})")
            break
    else:
        history.stopped_epoch = cfg.epochs_max

    model.load_state_dict(best_state)
    history.training_time_s = time.time() - start_time
    history.model_size_mb = model_size_mb(model)
    return model, history


def save_checkpoint(path: str, model: nn.Module, config: dict, scaler_path: str | None = None,
                     extra: dict | None = None) -> None:
    """Sauvegarde standardisée : state_dict + config d'architecture + chemin du scaler associé."""
    payload = {"state_dict": model.state_dict(), "config": config, "scaler_path": scaler_path}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
