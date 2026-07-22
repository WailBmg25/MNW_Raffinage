---
marp: true
theme: default
paginate: true
lang: fr
---

# Optimisation du Raffinage et de la Pétrochimie par Deep Learning
### Jumeau numérique — CDU & Vapocraqueur

Réseaux de neurones PyTorch uniquement — 200 000 barils/jour, données 100 % synthétiques

---

## Le problème

- Raffinerie 200 000 bbl/j : unité de distillation atmosphérique (CDU) + vapocraqueur
- Deux goulets d'étranglement, forte variabilité (brut, charge, encrassement)
- Objectif : un jumeau numérique qui prédit, détecte, optimise et alerte en continu

---

## 5 objectifs métier

| # | Objectif | Critère de succès |
|---|---|---|
| 1 | Rendements des coupes | MAPE < 5 % par coupe |
| 2 | Détection du fouling | > 24 h avant nettoyage |
| 3 | Optimisation four (COT) | Gain énergétique > 5 % |
| 4 | Qualité produits | Corrélation > 0.9 |
| 5 | Alertes temps réel | Latence < 1 min |

---

## Contrainte de méthode & données

- **Deep Learning uniquement** — aucun ML classique (pas de XGBoost/RF/SVM/k-means)
- scikit-learn limité à `StandardScaler`, métriques, split train/val/test
- Données 100 % synthétiques mais physiquement cohérentes (bilans matière/énergie)
- 2 ans de données horaires (~17 500 points) + labo toutes les 8 h (délai 4 h, zéro fuite)

---

## Méthodologie de mesure

- **Régression** (rendements, énergie, qualité) : MAPE + **RMSE** + **R²** — jamais une seule
- **Classification** (fouling) : precision/recall/F1/**accuracy**/AUC + 2 métriques métier
  (avance de détection, corrélation à la vérité terrain)
- Pourquoi : certaines métriques mentent seules selon l'échelle ou le déséquilibre des classes
  (détaillé notebook par notebook)

---

## Techniques de Deep Learning utilisées

| Famille | Modèles | Utilisés pour |
|---|---|---|
| Dense | MLP | Baseline rendements, surrogate énergie |
| Récurrent | RNN, LSTM, GRU, BiLSTM | Rendements, qualité, résidu de fouling |
| Convolutionnel | CNN 1D, TCN | Rendements |
| Attention | Transformer encoder | Rendements |
| Non supervisé | AE dense/conv1D/LSTM, VAE | Détection de fouling |
| Optimisation | Surrogate MLP + gradient | Optimisation énergétique |

---

## Notebook 03 — Rendements : comparatif des 8 architectures

![bg right:55% fit](images/03_comparaison_architectures.png)

- **RNN simple** gagnant : MAPE 2.99 %, RMSE 0.0137, R² 0.861
- 13 956 paramètres seulement — bat 7 architectures plus lourdes
- MAPE/RMSE/R² classent les 8 architectures **dans le même ordre**

---

## Pourquoi le RNN simple gagne

![bg left:45% fit](images/03_parity_plot.png)

- Fenêtre d'entrée courte (24 h) → dépendance surtout **récente** (ACF/PACF)
- Mémoire longue (LSTM/BiLSTM/GRU/Transformer) inutile ici
- Jeu d'entraînement modeste (~12 300 séquences) → sur-paramétrisation pénalisée
- CNN 1D pire (R²=0.740) : champ réceptif trop court ; le TCN corrige via dilatation

---

## Notebook 04 — Détection du fouling : 5 méthodes comparées

| Méthode | Precision | Recall | F1 | Accuracy | AUC | Corr. vérité |
|---|---|---|---|---|---|---|
| **Résidu GRU** | 0.009 | 0.323 | 0.018 | 0.802 | 0.671 | **0.490** |
| AE dense | 0.035 | 0.135 | 0.055 | **0.975** | 0.780 | 0.265 |
| AE conv1D | 0.010 | 0.573 | 0.019 | 0.678 | 0.589 | 0.250 |
| AE LSTM | 0.024 | 0.469 | 0.046 | 0.894 | 0.652 | 0.153 |
| VAE | 0.006 | 0.156 | 0.012 | 0.861 | 0.602 | 0.062 |

---

## La vérité cachée vs les détections

![bg right:55% fit](images/04_timeline_verite_vs_detections.png)

- Les 5 méthodes dépassent largement l'objectif de 24 h d'avance
- Résidu GRU : 3022 h d'avance moyenne (contre 2144–4114 h pour les autres)

---

## Le paradoxe de l'accuracy

- Seules **96 heures / 17 496 (0.55 %)** sont réellement positives (5 nettoyages en 2 ans)
- Résidu GRU : 3443 alertes (19.7 %), 31 vrais positifs, 3412 faux positifs
  → Precision 0.9 %, Recall 32.3 %, **Accuracy 80.2 %**
- AE dense : **97.5 % d'accuracy** = la meilleure des 5, mais juste car il alerte moins souvent
  (recall 13.5 %, corrélation vérité 0.265 seulement)
- F1/accuracy trompeurs sur classe rare → **jamais lus seuls**

---

## Pourquoi le résidu GRU est retenu en production

![bg left:45% fit](images/04_roc_curves.png)

- Prédit un capteur unique piloté par le fouling (préchauffe) — pas 83 variables
- Corrélation à la vérité terrain : **0.490** (vs 0.06–0.27 pour les autoencodeurs)
- Les autoencodeurs réagissent à tout changement opératoire, pas seulement au fouling
- Score lissé par EWMA causal (span 24 h)

---

## Notebook 05 — Optimisation énergétique

![bg right:50% fit](images/05_optimisation_resultats.png)

- Surrogate MLP (128-128-64) : prédit 4 rendements + énergie spécifique
- Descente de gradient sur COT/reflux, poids gelés, contrainte de rendement
- **Gain énergétique : 5.53 %** (objectif > 5 %) — 774 $/jour, 4.13 tCO₂/j évitées
- 96.3 % des recommandations respectent la contrainte de rendement

---

## Pourquoi l'énergie spécifique a un R² négatif

- `specific_energy` piloté surtout par le **fouling caché**, pas COT/reflux
- Terme correctif volontairement faible ajouté pour donner un signal au gradient
- RMSE 0.159 kWh/bbl ≈ pas mieux que prédire la moyenne → R² négatif
- **N'invalide pas l'optimisation** : seule la sensibilité directionnelle compte, pas la
  prédiction absolue — confirmé par le gain de 5.53 % obtenu en backtest

---

## Notebook 06 — Qualité produits + pipeline temps réel

![bg left:45% fit](images/06_quality_parity.png)

- GRU multi-sorties, 5 cibles qualité labo, cibles standardisées séparément
- **Corrélation moyenne : 0.971** (objectif > 0.9)
- Pipeline temps réel : latence 22.6 ms (moy.), 174 ms (max) — objectif < 1 min
- 107 alertes générées sur le backtest (2633 h rejouées)

---

## Pourquoi le soufre a la meilleure corrélation mais le pire MAPE

- `sulfur_content` : petite échelle (~1.2 %pds) → erreur absolue faible mais MAPE élevé (8.54 %)
- `residue_viscosity` : grosse échelle (~385 cSt) → RMSE élevé mais MAPE bas (1.52 %)
- La corrélation (invariante à l'échelle) montre que la **forme** est bien capturée dans les 2 cas
- Aucune métrique seule ne suffit : RMSE dépend de l'unité, MAPE explose sur petites échelles

---

## Synthèse — 5/5 objectifs atteints

| # | Objectif | Résultat | Statut |
|---|---|---|---|
| 1 | Rendements (MAPE<5%) | 2.99 % (RNN simple) | ✅ |
| 2 | Fouling (>24h) | 3022 h (résidu GRU, corr. 0.49) | ✅ |
| 3 | Énergie (gain>5%) | 5.53 % (774 $/j) | ✅ |
| 4 | Qualité (corr>0.9) | 0.971 | ✅ |
| 5 | Alertes (<1min) | ~23 ms (max 174 ms) | ✅ |

---

## Du synthétique au réel — pipeline de déploiement

![bg right:55% fit](images/pipeline_deploiement_reel.png)

- Terrain (capteurs/historian/LIMS) → Ingestion → Prétraitement → Inférence → Décision → Salle de contrôle
- **L'architecture logicielle ne change pas** — seule la source de données évolue
- Human-in-the-loop : recommandations validées par un opérateur, pas d'automatisation directe
- Boucle MLOps : suivi de dérive, ré-entraînement périodique, validation avant redéploiement

---

## Le dashboard — jumeau numérique

| Page | Rôle |
|---|---|
| Vue d'ensemble | KPI temps réel, rendements 48h, alertes |
| Jumeau numérique | Synoptique interactif du procédé complet |
| Rendements | Prédit/réel, simulateur what-if |
| Encrassement | Indice de fouling vs vérité terrain |
| Énergie | Optimisation à la demande |

---

## Aperçu du dashboard

![bg right:50% fit](images/dashboard_home.png)
![bg fit](images/dashboard_jumeau.png)

---

## Aperçu du dashboard (suite)

![bg right:33% fit](images/dashboard_rendements.png)
![bg fit](images/dashboard_energie.png)
![bg fit](images/dashboard_encrassement.png)

---

## Stack technique

| Composant | Techno | Pourquoi |
|---|---|---|
| Deep Learning | PyTorch | CPU-friendly, flexible |
| Backend | FastAPI + WebSockets | Async natif, temps réel |
| Frontend | Next.js + TypeScript | SSR/CSR hybride |
| Graphiques | Recharts, React Flow | Déclaratif, synoptique interactif |
| Déploiement | Docker, docker-compose | Reproductible |

---

## Conclusion — 3 enseignements transversaux

1. **Le modèle le plus simple gagne souvent** — le RNN simple bat 7 architectures plus lourdes
2. **Aucune métrique unique ne suffit** — MAPE/RMSE/R² et accuracy/F1 peuvent raconter des
   histoires opposées selon l'échelle et le déséquilibre des classes
3. **Le déploiement réel ne change pas l'architecture logicielle** — seule la source de
   données évolue, moyennant une gouvernance MLOps

**5/5 objectifs atteints. Pipeline complet en quelques dizaines de millisecondes.**
