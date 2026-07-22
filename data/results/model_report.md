# Rapport de modèle — Jumeau numérique CDU & Vapocraqueur

*Généré automatiquement par `notebooks/06_realtime_system.ipynb` — 2026-07-22 10:06*


## Synthèse : 5/5 objectifs atteints

| # | Objectif | Critère | Valeur atteinte | Statut |
|---|----------|---------|------------------|--------|
| 1 | Prédiction des rendements | MAPE < 5% par coupe | 2.99% (architecture RNN simple) | ✅ |
| 2 | Détection du fouling | > 24h avant nettoyage | 3022.2h (méthode gru_residual) | ✅ |
| 3 | Optimisation énergétique | Gain > 5% | 5.53% (774 $/j, 4.13 tCO2/j) | ✅ |
| 4 | Soft sensor qualité | Corrélation > 0.9 | 0.971 | ✅ |
| 5 | Système d'alerte temps réel | Latence < 1 min | 22.6 ms (moyenne), 174.0 ms (max) | ✅ |

## Détails par notebook

- **Notebook 03** (rendements) : architecture retenue = `RNN simple`, 13956 paramètres.
- **Notebook 04** (fouling) : méthode retenue = `gru_residual`, seuil = 231.9325.
- **Notebook 05** (énergie) : taux de respect de la contrainte de rendement = 96.3%.
- **Notebook 06** (temps réel) : 2633 heures rejouées, 107 alertes générées ({'info': 0, 'warning': 101, 'critical': 6}).
