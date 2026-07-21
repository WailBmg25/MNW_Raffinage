"""Fonctions de tracé partagées par les notebooks (style cohérent, sauvegarde uniforme)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def plot_learning_curves(history, title: str, save_path: str | Path, log_scale: bool = False) -> None:
    """Trace loss train vs val par epoch, avec annotation de l'epoch d'arrêt (early stopping)."""
    fig, ax1 = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = np.arange(1, len(history.train_loss) + 1)
    ax1[0].plot(epochs, history.train_loss, label="Train", color="#0891b2", linewidth=1.8)
    ax1[0].plot(epochs, history.val_loss, label="Validation", color="#f59e0b", linewidth=1.8)
    ax1[0].axvline(history.best_epoch, color="#10b981", linestyle="--", linewidth=1.3,
                    label=f"Meilleure epoch ({history.best_epoch})")
    if log_scale:
        ax1[0].set_yscale("log")
    ax1[0].set_xlabel("Epoch")
    ax1[0].set_ylabel("Loss")
    ax1[0].set_title(f"{title} — Courbes d'apprentissage")
    ax1[0].legend()

    ax1[1].plot(epochs, history.lr, color="#7c3aed", linewidth=1.8)
    ax1[1].set_xlabel("Epoch")
    ax1[1].set_ylabel("Learning rate")
    ax1[1].set_title("Évolution du learning rate (scheduler)")
    ax1[1].set_yscale("log")

    fig.suptitle(f"Temps d'entraînement : {history.training_time_s:.1f}s | "
                 f"Paramètres : {history.n_params_total:,} | Taille : {history.model_size_mb:.2f} Mo".replace(",", " "))
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()


def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str],
                 title: str, save_path: str | Path) -> None:
    """Nuage de points prédiction vs réel (parity plot) par cible, avec la diagonale idéale."""
    n_targets = len(target_names)
    fig, axes = plt.subplots(1, n_targets, figsize=(4.2 * n_targets, 4))
    if n_targets == 1:
        axes = [axes]
    for i, (ax, name) in enumerate(zip(axes, target_names)):
        ax.scatter(y_true[:, i], y_pred[:, i], s=6, alpha=0.35, color="#0891b2")
        lims = [min(y_true[:, i].min(), y_pred[:, i].min()), max(y_true[:, i].max(), y_pred[:, i].max())]
        ax.plot(lims, lims, color="#ef4444", linestyle="--", linewidth=1.3)
        ax.set_xlabel("Réel")
        ax.set_ylabel("Prédit")
        ax.set_title(name)
    fig.suptitle(title)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()


def plot_timeseries_comparison(x, series: dict[str, np.ndarray], title: str, ylabel: str,
                                 save_path: str | Path, vlines: list | None = None) -> None:
    """Superpose plusieurs séries temporelles (ex. prédit vs réel) avec lignes verticales optionnelles
    (ex. nettoyages, décokages)."""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    colors = ["#0891b2", "#f59e0b", "#10b981", "#ef4444", "#7c3aed"]
    for i, (label, y) in enumerate(series.items()):
        ax.plot(x, y, label=label, linewidth=1.4, color=colors[i % len(colors)])
    if vlines:
        for v in vlines:
            ax.axvline(v, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()
