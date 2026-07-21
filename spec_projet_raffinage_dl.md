# DOCUMENT DE SPÉCIFICATION & PROMPT D'EXÉCUTION
## Projet : Optimisation du Raffinage et de la Pétrochimie par Deep Learning
### Application « Jumeau Numérique » (Digital Twin) — CDU & Vapocraqueur

Tu es un ingénieur Deep Learning senior + développeur full-stack. Construis **l'intégralité** de ce projet, de bout en bout : génération des données, notebooks, modèles, backend FastAPI, frontend Next.js (dashboard BI type jumeau numérique), Docker, documentation. **Tout le contenu visible (notebooks, commentaires, docstrings, README, UI, rapports) est en FRANÇAIS.**

---

# PARTIE A — RÈGLES GLOBALES

## A.1 Contrainte fondamentale : DEEP LEARNING UNIQUEMENT
**INTERDIT** : XGBoost, Random Forest, régression linéaire/logistique, SVM, Isolation Forest, k-means, ou tout algorithme de ML classique. Aucune librairie de ML classique ne doit apparaître (pas de `sklearn` pour des modèles — `sklearn` est autorisé UNIQUEMENT pour `StandardScaler`, les métriques et les utilitaires de split).

**Toutes les solutions sont des réseaux de neurones PyTorch** :
- Baseline : **MLP** (perceptron multicouche)
- Séquentiel : **RNN simple, LSTM, GRU, LSTM bidirectionnel**
- Convolutionnel : **CNN 1D, TCN (Temporal Convolutional Network, convolutions dilatées causales)**
- Hybride : **CNN-LSTM**
- Attention : **Transformer encoder** (encodage positionnel, multi-head attention)
- Non supervisé / détection d'anomalies : **Autoencodeur dense, Autoencodeur convolutionnel 1D, Autoencodeur LSTM (seq2seq), VAE** (variationnel)
- Optimisation : **réseau surrogate + descente de gradient sur les entrées** (+ variante RL légère type REINFORCE en bonus)

## A.2 Stack technique
- **PyTorch >= 2.1** + **tqdm** (barre de progression sur CHAQUE boucle d'entraînement, affichant epoch, loss train, loss val)
- **torchinfo** pour les résumés d'architecture
- Python 3.11 ; numpy, pandas, matplotlib, seaborn, plotly, pyyaml, joblib, fastapi, uvicorn[standard], pydantic v2, websockets
- Reproductibilité : `seed=42` (random, numpy, torch), `cudnn.deterministic=True`
- Détection automatique du device : `device = "cuda" if torch.cuda.is_available() else "cpu"` — tout doit tourner sur CPU
- Données **100 % synthétiques** générées par `src/data_generator.py`, aucun téléchargement externe

## A.3 Exigences d'entraînement — OBLIGATOIRES pour chaque modèle, dans chaque notebook
Pour **chaque réseau** entraîné, le notebook doit contenir, dans cet ordre :
1. **Affichage de l'architecture** : `print(model)` **ET** `torchinfo.summary(model, input_size=...)` avec nombre de paramètres totaux/entraînables — suivi d'une cellule markdown expliquant en français les choix d'architecture (pourquoi ces couches, ces tailles)
2. **Hyperparamètres explicites** dans un dict affiché : couches, hidden size, dropout, learning rate, batch size, epochs max, weight decay
3. **Régularisation** : Dropout (0.2–0.4) dans tous les modèles ; LayerNorm dans le Transformer ; BatchNorm dans les CNN ; weight decay (AdamW)
4. **Optimiseurs** : AdamW par défaut ; le notebook 03 doit inclure une **mini-étude comparative d'optimiseurs** (SGD+momentum vs Adam vs AdamW sur le même LSTM, courbes superposées)
5. **Scheduler** : ReduceLROnPlateau (ou CosineAnnealing pour le Transformer) — tracer le learning rate au fil des epochs
6. **Early stopping** (patience 10, restauration des meilleurs poids) + **gradient clipping** (max_norm=1.0)
7. **Courbes d'apprentissage** : loss train vs val par epoch (matplotlib, sauvegardées dans `data/results/figures/`), avec annotation de l'epoch d'arrêt
8. Sauvegarde du meilleur modèle : `torch.save({'state_dict':..., 'config':..., 'scaler_path':...}, ...)`
9. Temps d'entraînement et taille du modèle (Mo) rapportés dans le tableau comparatif final

---

# PARTIE B — CONTEXTE MÉTIER (à reprendre dans README + notebook 01)

Raffinerie de **200 000 barils/jour**. L'unité de distillation atmosphérique (CDU) et le vapocraqueur sont des goulets d'étranglement. Mission et objectifs chiffrés :

| # | Objectif | Critère de succès | Solution DL |
|---|----------|-------------------|-------------|
| 1 | Prédire les rendements des coupes (naphta, kérosène, gazole, résidu) | MAPE < 5 % par coupe | LSTM/GRU/TCN/Transformer (comparés) |
| 2 | Détecter le fouling (encrassement) | Détection > 24 h avant nettoyage | Autoencodeurs + résidus LSTM |
| 3 | Optimiser la température du four (COT) | Gain énergétique > 5 % | Surrogate NN + gradient sur entrées |
| 4 | Prédire la qualité des produits (soft sensor labo) | Corrélation > 0.9 | GRU multi-sorties |
| 5 | Système d'alerte temps réel | Latence < 1 min | Pipeline d'inférence + moteur d'alertes |

---

# PARTIE C — STRUCTURE DU PROJET (obligatoire)

```
projet_refining_optimization/
├── data/
│   ├── raw/                    # cdu_data.csv, cracker_data.csv, lab_data.csv, energy_data.csv
│   ├── processed/              # X/y .npy + scalers .joblib
│   └── results/
│       ├── figures/
│       ├── yield_predictions.npy
│       └── model_report.md     # généré automatiquement (notebook 06)
├── notebooks/
│   ├── 01_eda_refining.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_yield_prediction.ipynb
│   ├── 04_fouling_detection.ipynb
│   ├── 05_energy_optimization.ipynb
│   └── 06_realtime_system.ipynb
├── src/
│   ├── data_generator.py       # génération physique des données
│   ├── preprocessing.py
│   ├── models/                 # UN fichier par famille, réutilisés par les notebooks ET le backend
│   │   ├── mlp.py, rnn.py (RNN/LSTM/GRU/BiLSTM), cnn.py (CNN1D/TCN/CNN-LSTM),
│   │   ├── transformer.py, autoencoders.py (AE dense/conv/LSTM/VAE), surrogate.py
│   ├── training.py             # boucle d'entraînement générique (tqdm, early stopping, clipping)
│   ├── yield_model.py
│   ├── fouling_detector.py
│   ├── energy_optimizer.py
│   ├── realtime_monitor.py
│   └── alert_system.py
├── backend/                    # FastAPI (détail Partie F)
├── frontend/                   # Next.js (détail Partie G)
├── docker-compose.yml
├── config.yaml
├── requirements.txt
└── README.md
```

---

# PARTIE D — GÉNÉRATEUR DE DONNÉES SYNTHÉTIQUES (physique simplifiée)

Générer **2 ans de données horaires** (index datetime commun `2024-01-01` → `2025-12-31`, ≈17 520 lignes/table). Basé sur bilans matière/énergie, paramétrable via `config.yaml`. Exécution : `python -m src.data_generator` (avec tqdm).

### D.1 `cdu_data.csv`
- `feed_rate` (m³/h ~1325, rampes de charge réalistes), `crude_type` (`leger/moyen/lourd`, changement tous les 5–15 j, transition 6–12 h), `crude_api` (34/31/27 + bruit)
- `furnace_cot` (355–375 °C, contrôleur imparfait), `column_top_temp`, `column_top_pressure`, `reflux_ratio`, `stripping_steam`
- `naphtha_yield`, `kerosene_yield`, `gasoil_yield`, `residue_yield` : fractions massiques **somme = 1**, dérivées d'une courbe TBP par brut + effets non linéaires du COT (↑COT ⇒ ↑distillats/↓résidu) et du reflux (netteté des coupes) + bruit capteur ~0.3 %
- Injecter volontairement : **dérive lente sur 2–3 capteurs, valeurs manquantes (~1 %), outliers** (pour l'EDA/préprocessing)

### D.2 `energy_data.csv`
- `fouling_resistance` (m²K/W) : croissance asymptotique type Ebert–Panchal simplifié, **reset à ~0 lors de 4–5 nettoyages sur 2 ans**. C'est la **vérité terrain cachée** : sert à générer les labels, **jamais utilisée comme feature d'entrée**
- `preheat_outlet_temp` (↓ avec le fouling), `fuel_gas_flow`, `furnace_duty` (MW, ↑ avec fouling et charge), `specific_energy` (kWh/baril), `co2_emissions` (t/h)
- Labels : `cleaning_needed_within_24h` / `_48h` (franchissement futur d'un seuil de fouling)

### D.3 `cracker_data.csv`
- `naphtha_feed`, `coil_outlet_temp` (820–860 °C), `steam_to_oil_ratio`, `residence_time`
- `ethylene_yield` (↑ avec sévérité puis plateau), `propylene_yield` (en cloche), `c4_yield`, `pygas_yield`
- `coke_thickness` (↑ avec sévérité, reset aux décokages ~45–60 j), `tube_metal_temp` corrélée

### D.4 `lab_data.csv`
- Échantillonnage **toutes les 8 h**, **délai de disponibilité 4 h** (`sample_time`, `result_time`)
- `naphtha_final_boiling_point`, `kerosene_flash_point`, `gasoil_cetane_index`, `residue_viscosity`, `sulfur_content` — fonctions physiques plausibles des conditions au prélèvement + bruit labo

---

# PARTIE E — NOTEBOOKS (français, figures dans `data/results/figures/`)

### 01 — EDA complète
Stats descriptives, valeurs manquantes (heatmap), outliers (IQR/z-score), séries temporelles clés, effets des changements de brut, corrélations, distributions des rendements par brut, cycle de fouling visualisé (nettoyages marqués), décomposition STL, ACF/PACF. **Conclusion markdown à la fin de chaque section.**

### 02 — Préprocessing
Imputation par interpolation temporelle, clipping des outliers, correction de dérive capteur, jointure asof avec le labo **en respectant le délai de 4 h (zéro fuite de données)**, feature engineering (lags, moyennes glissantes 6h/24h, deltas, encodage crude_type), création des séquences (fenêtre 24 h → rendements t+1 ; fenêtre 48 h → fouling), **split temporel strict 70/15/15 sans shuffle**, StandardScaler fit sur train uniquement, sauvegarde .npy + .joblib.

### 03 — Prédiction des rendements : GRAND COMPARATIF DL
Entraîner et comparer **8 architectures** (sortie multi-cibles = 4 rendements) :
1. MLP (baseline, sur la dernière observation aplatie)
2. RNN simple
3. LSTM (2 couches, hidden 64)
4. GRU
5. LSTM bidirectionnel
6. CNN 1D (3 blocs conv-BN-ReLU-pool)
7. TCN (convolutions dilatées causales, blocs résiduels)
8. Transformer encoder (d_model 64, 4 têtes, 2 couches, encodage positionnel sinusoïdal)

Chaque modèle suit **intégralement les exigences A.3** (print architecture + torchinfo, courbes, dropout, scheduler...). Inclure la mini-étude d'optimiseurs (A.3.4) et une courbe comparant dropout 0.0/0.2/0.4 sur le LSTM (mise en évidence du surapprentissage).
**Tableau final** : MAPE/RMSE/R² par coupe + global, nb paramètres, temps d'entraînement, taille (Mo) — avec barplot comparatif et discussion en français. Vérifier **MAPE < 5 %**. Parity plots, prédictions vs réel sur 2 semaines. Sauver le meilleur `.pt` + `yield_predictions.npy`.

### 04 — Détection du fouling : DL non supervisé + résidus
1. **Approche par résidus** : GRU entraîné à prédire `preheat_outlet_temp` sur les périodes propres (post-nettoyage) ; en exploitation, résidu lissé EWMA = indice de fouling
2. **Autoencodeur dense**, **autoencodeur conv 1D**, **autoencodeur LSTM seq2seq**, **VAE** : entraînés sur séquences propres, erreur de reconstruction = score d'anomalie
Comparer les 5 approches : précision/rappel/F1 sur `cleaning_needed_within_24h`, **avance moyenne de détection (> 24 h requis)**, courbes ROC, visualisation des reconstructions, espace latent du VAE (projection 2D), timeline vérité cachée vs indices estimés vs alertes.

### 05 — Optimisation énergétique
- **Surrogate MLP profond** : (conditions, COT, reflux) → (4 rendements, énergie spécifique) — entraîné selon A.3
- **Optimisation par gradient sur les entrées** : COT et reflux = `nn.Parameter`, poids du surrogate gelés, minimisation énergie sous contraintes par pénalités (distillats ≥ actuel − 0.5 pt, bornes opératoires) ; comparer à un random search ; **bonus** : petite politique REINFORCE
- Backtest sur test : énergie baseline vs optimisée, **gain % (> 5 %)**, économies € et CO₂ (prix dans config.yaml), distribution des COT recommandés

### 06 — Système temps réel + rapport final
Rejeu du test set heure par heure (tqdm), pipeline complet (préproc → 3 modèles → alertes) avec **latence mesurée en ms** ; soft sensor qualité GRU en continu (corr > 0.9) ; moteur d'alertes (fouling, dérive rendement > 2σ, qualité hors spec, énergie anormale ; niveaux info/warning/critique ; anti-rebond) ; **génération automatique de `model_report.md`** : tableau des 5 objectifs, valeurs atteintes, ✅/❌.

---

# PARTIE F — BACKEND FastAPI (spécification détaillée)

### F.1 Architecture
```
backend/app/
├── main.py                 # FastAPI, lifespan (chargement modèles), CORS, middleware latence
├── core/config.py          # lecture config.yaml + env
├── schemas.py              # Pydantic v2 : OperatingConditions, YieldPrediction, FoulingStatus,
│                           #   OptimizationResult, Alert, SensorReading, TwinState, KpiSummary
├── services/
│   ├── model_registry.py   # charge les .pt + scalers une seule fois, expose predict_*()
│   ├── twin_engine.py      # moteur de simulation : rejoue le test set, calcule l'état complet
│   │                       #   du jumeau (capteurs + prédictions + santé équipements)
│   └── alert_engine.py     # règles, niveaux, anti-rebond, journal en mémoire + JSON
└── routers/
    ├── kpi.py, yields.py, fouling.py, energy.py, alerts.py, twin.py, realtime.py
```

### F.2 Endpoints
| Méthode | Route | Description |
|---|---|---|
| GET | `/api/health` | statut + modèles chargés + device |
| GET | `/api/kpi/summary` | KPI du jour : débit, rendements, énergie spécifique, indice fouling, alertes actives, statut des 5 objectifs |
| GET | `/api/yields/history?hours=168` | historique réel + prédit par coupe |
| POST | `/api/yields/predict` | conditions → 4 rendements prédits (simulation what-if) |
| GET | `/api/fouling/status` | indice actuel, tendance, **jours estimés avant nettoyage**, historique + nettoyages |
| POST | `/api/energy/optimize` | conditions → COT/reflux recommandés, gain %, €/jour, tCO₂/jour évitées |
| GET | `/api/twin/state` | **état complet du jumeau** : liste des capteurs (id, nom, valeur, unité, statut ok/warning/alarme, sparkline 24 h) + santé de chaque équipement (train de préchauffe, four, colonne, strippers, vapocraqueur) |
| GET | `/api/twin/sensor/{id}` | détail d'un capteur : trend 7 j, valeur prédite vs mesurée, limites |
| GET | `/api/alerts?limit=50` / `/api/alerts/active` | journal / alertes actives |
| WS | `/ws/realtime` | pousse toutes les 2 s (configurable) un `TwinState` complet : le frontend est un vrai jumeau numérique vivant |

### F.3 Comportement
- Chargement des artefacts au démarrage (lifespan) ; **mode dégradé explicite** si artefacts absents
- Middleware qui logge et renvoie la latence dans un header `X-Process-Time-Ms`
- Le `twin_engine` rejoue le jeu de test en accéléré (1 h de données / tick websocket) en boucle infinie ; chaque tick contient mesures + inférences des 3 modèles + alertes générées
- Schémas Pydantic pour toutes les E/S ; erreurs HTTP propres ; tests rapides avec `httpx` (bonus)

---

# PARTIE G — FRONTEND : DASHBOARD BI « JUMEAU NUMÉRIQUE » (spécification détaillée)

### G.1 Stack imposée
- **Next.js 14 (App Router) + TypeScript**
- **Tailwind CSS** + **shadcn/ui** (cards, tabs, badges, drawer/sheet, table, tooltip, dialog)
- **Recharts** : courbes, aires, barres, gauges radiales, sparklines
- **React Flow (@xyflow/react)** : le **synoptique interactif du procédé** (cœur du jumeau numérique)
- **lucide-react** (icônes), **framer-motion** (animations d'entrée, pulsation des alarmes, flux animés)
- **TanStack Query** (fetch REST + cache) et hook websocket custom pour `/ws/realtime`
- **zustand** pour l'état global (état du jumeau, alertes)
- `NEXT_PUBLIC_API_URL` en variable d'environnement

### G.2 Direction artistique (à respecter strictement)
- **Thème sombre "salle de contrôle"** : fond `#0B1120`/slate-950, cartes en surfaces `#111827` avec bordure subtile `slate-800`, coins arrondis 2xl, ombres douces
- Accents : **cyan/teal** pour les mesures, **ambre** pour warnings, **rouge** pour alarmes (avec pulsation framer-motion), **vert émeraude** pour les statuts sains et les gains IA
- Typographie : Inter (texte) + **chiffres tabulaires façon instrumentation** pour les valeurs de capteurs ; grandes valeurs KPI en 3xl-4xl
- Micro-interactions : transitions douces, hover states, skeletons de chargement, compteurs animés
- En-tête permanent : nom de la raffinerie, horodatage simulé du jumeau, pastille « ● Temps réel connecté », compteur d'alertes
- Rendu global : moderne, dense mais lisible, digne d'une démo industrielle — **pas** un template Bootstrap générique

### G.3 Pages (sidebar de navigation avec icônes)

**1. `/` — Vue d'ensemble (Command Center)**
- Rangée de 6 **cartes KPI animées** : débit de charge, rendement distillats (%), énergie spécifique (kWh/bbl) avec delta vs baseline, indice de fouling (gauge), alertes actives, « Objectifs IA : 5/5 ✅ »
- Grand graphique temps réel (aire empilée des 4 rendements) alimenté par le websocket, fenêtre glissante 48 h
- Colonne droite : flux d'alertes en direct (badges par niveau) + mini-cartes « gain énergie réalisé aujourd'hui » et « CO₂ évité »

**2. `/jumeau` — Jumeau numérique (page vitrine, la plus soignée)**
- **Synoptique React Flow du procédé complet**, custom nodes stylisés : Brut → Dessaleur → **Train de préchauffe** (3 échangeurs E-101/102/103) → **Four F-101** → **Colonne CDU C-101** avec soutirages latéraux (naphta / kérosène / gazole / résidu) → **Vapocraqueur** (naphta → éthylène/propylène)
- Chaque équipement : icône/forme dédiée, **badge de santé coloré** (vert/ambre/rouge) calculé par les modèles (ex. train de préchauffe passe en ambre quand l'indice fouling monte)
- **Capteurs live** : petites étiquettes sur les arêtes/nœuds affichant TI/PI/FI avec valeur et unité, mises à jour à chaque tick websocket ; arêtes animées (flux) dont l'épaisseur reflète le débit
- **Clic sur un capteur ou équipement → Drawer latéral** : trend 7 jours (Recharts), valeur mesurée vs **valeur prédite par le réseau** (l'écart matérialise l'anomalie), limites opératoires, alertes associées
- Bandeau bas : « Ce synoptique est piloté par 3 réseaux de neurones en inférence continue » avec latence affichée en ms

**3. `/rendements`**
- Courbes prédit vs réel par coupe (4 petits multiples), gauges de MAPE par coupe vs objectif 5 %
- **Simulateur what-if** : sliders (COT, reflux, débit) + select type de brut → appel `/api/yields/predict` → jauge/barres des rendements prédits avec deltas animés vs conditions actuelles
- Encart « Modèle en production » : nom de l'architecture gagnante, nb de paramètres, MAPE test

**4. `/encrassement`**
- Gauge de l'indice de fouling + carte « **Nettoyage estimé dans X jours** »
- Timeline 2 ans : indice estimé par l'autoencodeur, épisodes détectés, nettoyages marqués (lignes verticales)
- Graphique « erreur de reconstruction » avec seuil, tableau des épisodes (début, détection, avance en h — mettre en évidence > 24 h)

**5. `/energie`**
- Comparaison énergie spécifique réelle vs optimisée (aire entre les courbes = économies, colorée en émeraude)
- Bouton « **Optimiser maintenant** » → POST `/api/energy/optimize` → panneau de recommandation : COT recommandé vs actuel, gain %, €/jour, tCO₂/jour, avec contraintes affichées (« rendements préservés ✓ »)
- Compteurs cumulés animés : € économisés et CO₂ évité depuis le début de la simulation

**6. `/alertes`**
- Table shadcn filtrable (niveau, type, équipement, statut), tri, badge coloré, détail en dialog (contexte, valeurs, recommandation)

**7. `/documentation`**
- Rappel des objectifs et résultats (tableau ✅), schéma d'architecture (mermaid ou SVG), liste des 8 architectures comparées avec leurs métriques

### G.4 Qualité frontend
- Composants réutilisables : `KpiCard`, `SensorBadge`, `HealthBadge`, `TrendChart`, `GaugeChart`, `AlertFeed`, `EquipmentNode`
- États de chargement (skeletons) et d'erreur sur chaque page ; reconnexion automatique du websocket avec indicateur
- Responsive (le synoptique reste utilisable en zoom/pan sur mobile)

---

# PARTIE H — DOCKER & DOCUMENTATION

- `backend/Dockerfile` : python:3.11-slim, torch CPU (`--index-url https://download.pytorch.org/whl/cpu`), uvicorn, HEALTHCHECK `/api/health`
- `frontend/Dockerfile` : multi-stage node:20-alpine, `output: "standalone"`
- `docker-compose.yml` : `backend` (8000) + `frontend` (3000), depends_on condition healthy, volumes `models_artifacts` et `data`
- `config.yaml` : seed, chemins, paramètres du générateur, hyperparamètres par modèle, seuils d'alerte, prix gaz/CO₂, vitesse de simulation
- `README.md` (français) : contexte, architecture (schéma), installation locale + Docker, ordre d'exécution (`python -m src.data_generator` → notebooks 01→06 ou `python train_all.py` → `docker compose up --build`), description des données, tableau de résultats, description du dashboard, limites & perspectives
- Bonus : `train_all.py` (CLI qui entraîne tous les modèles avec tqdm et copie les artefacts vers `backend/models_artifacts/`)

---

# PARTIE I — CRITÈRES D'ACCEPTATION (à vérifier toi-même avant de conclure)

1. Aucun modèle de ML classique nulle part — uniquement des réseaux PyTorch
2. Chaque réseau entraîné affiche son architecture (`print` + torchinfo), ses courbes d'apprentissage train/val, utilise dropout + AdamW + scheduler + early stopping + gradient clipping
3. Notebook 03 : tableau comparatif des **8 architectures** avec MAPE < 5 % par coupe pour la meilleure
4. Notebook 04 : 5 approches DL comparées, avance de détection > 24 h démontrée
5. Notebook 05 : gain énergétique > 5 % en backtest, contraintes respectées
6. Soft sensor qualité : corrélation > 0.9 vs labo
7. Latence temps réel mesurée et affichée (< 1 min, en pratique < 1 s)
8. `docker compose up --build` → dashboard fonctionnel sur http://localhost:3000, synoptique du jumeau animé par le websocket, drawer capteur opérationnel
9. `model_report.md` généré automatiquement avec statut ✅/❌ des 5 objectifs
10. Tout le texte visible est en français ; design conforme à G.2

**Ordre d'exécution** : générateur de données → notebooks 01→06 (avec les modules `src/`) → backend → frontend → docker → README. Annonce un plan bref, puis exécute-le entièrement.
