# MNW_Raffinage — Jumeau Numérique CDU & Vapocraqueur

Optimisation du raffinage et de la pétrochimie par Deep Learning — application « jumeau
numérique » pour une raffinerie de **200 000 barils/jour** (unité de distillation atmosphérique
CDU + vapocraqueur). Données **100 % synthétiques**, générées par bilans matière/énergie
simplifiés. Toutes les solutions de modélisation sont des **réseaux de neurones PyTorch** —
aucun algorithme de machine learning classique (scikit-learn n'est utilisé que pour
`StandardScaler`, les métriques et les utilitaires de split).

## Contexte métier et objectifs

| # | Objectif | Critère de succès | Solution DL | Résultat obtenu |
|---|----------|--------------------|--------------|-----------------|
| 1 | Prédire les rendements des coupes (naphta, kérosène, gazole, résidu) | MAPE < 5 % par coupe | LSTM/GRU/TCN/Transformer (8 architectures comparées) | **2.99 %** (RNN simple) ✅ |
| 2 | Détecter le fouling (encrassement) | Détection > 24 h avant nettoyage | Autoencodeurs + résidus GRU (5 approches comparées) | **~3764 h** (autoencodeur dense) ✅ |
| 3 | Optimiser la température du four (COT) | Gain énergétique > 5 % | Surrogate NN + gradient sur les entrées | **5.53 %**, 774 $/j, 4.13 tCO₂/j ✅ |
| 4 | Prédire la qualité des produits (soft sensor labo) | Corrélation > 0.9 | GRU multi-sorties | **0.971** ✅ |
| 5 | Système d'alerte temps réel | Latence < 1 min | Pipeline d'inférence + moteur d'alertes | **~23 ms** en moyenne ✅ |

Le rapport détaillé, généré automatiquement par le notebook 06, est disponible dans
[`data/results/model_report.md`](data/results/model_report.md).

## Architecture

```
Données synthétiques (src/data_generator.py)
        │
        ▼
Préprocessing (src/preprocessing.py) ──► data/processed/*.npy + scalers
        │
        ▼
Notebooks d'entraînement (03 à 06) ──► backend/models_artifacts/*.pt + scalers + JSON
        │
        ▼
Backend FastAPI (backend/app/)                     Frontend Next.js (frontend/)
  - src/model_registry.py : charge les .pt            - Dashboard « salle de contrôle »
  - src/realtime_monitor.py : rejeu temps réel         - Synoptique React Flow (/jumeau)
  - src/alert_system.py : moteur d'alertes             - WebSocket temps réel
  - /api/*, /ws/realtime  ◄──────── HTTP + WS ────────►  - TanStack Query + zustand
```

Les modules `src/yield_model.py`, `src/fouling_detector.py`, `src/energy_optimizer.py`,
`src/realtime_monitor.py` et `src/alert_system.py` sont **partagés** entre le notebook 06
(backtest) et le backend FastAPI (service temps réel) — un même code de production.

## Stack technique

- **Deep Learning** : PyTorch ≥ 2.1, torchinfo, tqdm — MLP, RNN/LSTM/GRU/BiLSTM, CNN1D/TCN,
  CNN-LSTM, Transformer, autoencodeurs (dense/conv1D/LSTM seq2seq), VAE, surrogate MLP.
- **Backend** : FastAPI, Pydantic v2, uvicorn, websockets.
- **Frontend** : Next.js (App Router, TypeScript), Tailwind CSS + shadcn/ui, Recharts,
  @xyflow/react (synoptique), framer-motion, TanStack Query, zustand.
- **Données** : pandas, numpy, scikit-learn (StandardScaler/métriques/split uniquement),
  statsmodels (STL, ACF/PACF), matplotlib/seaborn.
- **Infra** : Docker (backend + frontend), docker-compose, `uv` pour l'environnement Python local.

## Structure du projet

```
├── data/
│   ├── raw/                    # cdu_data.csv, energy_data.csv, cracker_data.csv, lab_data.csv
│   ├── processed/               # X/y .npy + scalers .joblib (régénérables, non versionnés)
│   └── results/
│       ├── figures/              # toutes les figures générées par les notebooks
│       ├── yield_predictions.npy
│       └── model_report.md       # généré automatiquement (notebook 06)
├── notebooks/                    # 01 à 06, exécutés (voir "Ordre d'exécution")
├── src/                          # génération de données, préprocessing, modèles, entraînement,
│                                  # et services d'inférence partagés avec le backend
├── backend/                      # API FastAPI (voir backend/app/)
├── frontend/                     # Dashboard Next.js
├── docker-compose.yml
├── config.yaml                   # configuration centrale (seed, hyperparamètres, seuils, prix)
├── train_all.py                  # CLI bonus : entraîne tout le pipeline en une commande
└── requirements.txt
```

## Installation locale (Python / notebooks)

Le projet utilise [`uv`](https://docs.astral.sh/uv/) pour l'environnement Python (3.11) :

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
```

## Ordre d'exécution

```bash
# 1. Génération des données synthétiques (2 ans horaires, seed=42)
.venv/bin/python -m src.data_generator

# 2. Préprocessing (nettoyage, feature engineering, séquences, split, scalers)
.venv/bin/python -m src.preprocessing

# 3. Notebooks d'entraînement, dans l'ordre (chacun sauvegarde ses artefacts dans
#    backend/models_artifacts/) :
.venv/bin/jupyter lab
#   -> 01_eda_refining.ipynb          (exploration, pas d'entraînement)
#   -> 02_preprocessing.ipynb         (walkthrough pédagogique du pipeline)
#   -> 03_yield_prediction.ipynb      (8 architectures comparées, MAPE < 5%)
#   -> 04_fouling_detection.ipynb     (5 approches comparées, détection > 24h)
#   -> 05_energy_optimization.ipynb   (surrogate + gradient, gain > 5%)
#   -> 06_realtime_system.ipynb       (soft sensor qualité + pipeline temps réel + model_report.md)

# --- OU, en une seule commande (bonus) ---
.venv/bin/python train_all.py

# 4. Backend + Frontend via Docker
docker compose up --build
#   -> Backend  : http://localhost:8000  (docs interactifs : /docs)
#   -> Frontend : http://localhost:3000
```

> Le backend fonctionne en **mode dégradé explicite** si un artefact est absent (`/api/health`
> l'indique) — chaque modèle « s'allume » automatiquement dès que son checkpoint apparaît dans
> `backend/models_artifacts/`, sans redémarrage de code nécessaire.

## Description des données

Quatre tables horaires sur 2 ans (`2024-01-01` → `2025-12-31`, ≈17 520 lignes), 100 %
synthétiques (`src/data_generator.py`), basées sur des bilans matière/énergie simplifiés :

- **`cdu_data.csv`** : débit de charge, type/API du brut, COT four, conditions colonne, taux de
  reflux, rendements des 4 coupes (somme = 1). Dérive lente sur 2 capteurs, ~1 % de valeurs
  manquantes et des outliers sont injectés volontairement (matière à l'EDA/préprocessing).
- **`energy_data.csv`** : résistance d'encrassement (**vérité terrain cachée**, jamais une
  feature), température de sortie du train de préchauffe, consommation de gaz combustible,
  énergie spécifique, CO₂, labels de nettoyage à venir (24h/48h).
- **`cracker_data.csv`** : conditions du vapocraqueur, rendements éthylène/propylène/C4/pygas,
  épaisseur de coke, température de peau des tubes.
- **`lab_data.csv`** : échantillonnage toutes les 8 h, délai de disponibilité de 4 h — 5 mesures
  qualité (point final naphta, point éclair kérosène, indice de cétane gazole, viscosité résidu,
  teneur en soufre).

Le préprocessing (`src/preprocessing.py`) respecte strictement l'absence de fuite de données :
jointure asof avec le labo selon son délai réel de disponibilité, split temporel 70/15/15 sans
mélange, `StandardScaler` ajusté uniquement sur le train.

## Le dashboard (frontend)

Thème sombre « salle de contrôle » (cyan/ambre/rouge/émeraude), 7 pages :

- **`/`** — Command Center : KPI animés, aire empilée des rendements en temps réel, flux d'alertes.
- **`/jumeau`** — synoptique interactif du procédé complet (React Flow), capteurs live,
  santé des équipements, tiroir de détail par capteur (mesuré vs prédit).
- **`/rendements`** — prédit vs réel par coupe, simulateur what-if.
- **`/encrassement`** — indice de fouling, timeline 2 ans, épisodes détectés.
- **`/energie`** — comparaison énergie réelle vs optimisée, bouton d'optimisation à la demande.
- **`/alertes`** — journal filtrable des alertes.
- **`/documentation`** — récapitulatif des objectifs et des 8 architectures comparées.

## Limites et perspectives

- Les données étant 100 % synthétiques, les corrélations apprises reflètent la physique
  simplifiée du générateur — un déploiement réel nécessiterait un réétalonnage sur données
  historiques réelles.
- La détection de fouling présente une précision faible (fort déséquilibre de classes, ~0.9 %
  d'heures positives) malgré une avance de détection très confortable ; un seuil différent ou
  des données réelles changeraient cet arbitrage précision/rappel.
- Le simulateur what-if (`/rendements`) construit sa fenêtre d'entrée en ne modifiant que le
  dernier pas de temps des conditions ; une version plus rigoureuse recalculerait aussi les
  features dérivées (lags/rolling) sur toute la fenêtre.
- Pistes d'amélioration : ré-entraînement continu (drift monitoring), API d'optimisation
  multi-objectifs (énergie + qualité + fouling simultanément), authentification du dashboard,
  tests automatisés (pytest/httpx) pour le backend.
