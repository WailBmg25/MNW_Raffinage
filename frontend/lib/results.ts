// Résultats réels des notebooks 03-06 (source de vérité : data/results/model_report.md
// et les tableaux comparatifs des notebooks eux-mêmes). Mis à jour après exécution complète.

export interface ArchitectureResult {
  name: string;
  type: string;
  mapeGlobal: number;
  params: number;
  trainingTimeS: number;
  sizeMb: number;
}

export const architectureResults: ArchitectureResult[] = [
  { name: "RNN simple", type: "Récurrent", mapeGlobal: 2.99, params: 13_956, trainingTimeS: 52.9, sizeMb: 0.056 },
  { name: "TCN", type: "Convolutionnel dilaté causal", mapeGlobal: 3.26, params: 42_884, trainingTimeS: 103.1, sizeMb: 0.170 },
  { name: "MLP", type: "Baseline dense", mapeGlobal: 3.26, params: 19_652, trainingTimeS: 24.1, sizeMb: 0.081 },
  { name: "LSTM", type: "Récurrent (2 couches)", mapeGlobal: 3.29, params: 75_844, trainingTimeS: 114.4, sizeMb: 0.293 },
  { name: "LSTM bidirectionnel", type: "Récurrent bidirectionnel", mapeGlobal: 3.34, params: 184_132, trainingTimeS: 185.8, sizeMb: 0.708 },
  { name: "GRU", type: "Récurrent (2 couches)", mapeGlobal: 3.40, params: 57_988, trainingTimeS: 161.5, sizeMb: 0.225 },
  { name: "Transformer", type: "Attention multi-têtes", mapeGlobal: 3.57, params: 74_660, trainingTimeS: 314.7, sizeMb: 0.421 },
  { name: "CNN 1D", type: "Convolutionnel", mapeGlobal: 4.19, params: 29_092, trainingTimeS: 79.4, sizeMb: 0.120 },
];

export const productionModel = {
  name: "RNN simple",
  params: 13_956,
  mapeTest: 2.99,
};

export const perCutMape = {
  naphtha: 4.95,
  kerosene: 1.82,
  gasoil: 1.12,
  residue: 4.06,
};

export const objectivesSummary = [
  { id: 1, label: "Rendements des coupes (MAPE < 5%)", achieved: true, value: "2.99 % (RNN simple)" },
  { id: 2, label: "Détection du fouling (> 24h avant nettoyage)", achieved: true, value: "3022 h (résidus GRU, corr. vérité terrain 0.49)" },
  { id: 3, label: "Optimisation énergétique (gain > 5%)", achieved: true, value: "5.53 % (774 $/j, 4.13 tCO2/j)" },
  { id: 4, label: "Qualité produits (corrélation > 0.9)", achieved: true, value: "0.971 (GRU multi-sorties)" },
  { id: 5, label: "Alertes temps réel (latence < 1 min)", achieved: true, value: "~55 ms (moyenne), 226 ms (max)" },
];

// `corrVeriteTerrain` : corrélation de Pearson entre le score brut de la méthode et la
// résistance d'encrassement cachée (vérité terrain), sur l'historique complet (2 ans).
// Ajoutée après coup car precision/rappel/F1 sont tous quasi nuls ici (déséquilibre extrême
// de l'étiquette rare "cleaning_needed_within_24h") et ne discriminent pas entre méthodes ;
// la corrélation est le critère qui a déterminé le choix du modèle de production.
export const foulingResults = [
  { method: "gru_residual", precision: 0.009, recall: 0.323, f1: 0.018, auc: 0.671, leadTimeH: 3022.25, params: 57_793, corrVeriteTerrain: 0.490 },
  { method: "dense_ae", precision: 0.035, recall: 0.135, f1: 0.055, auc: 0.780, leadTimeH: 2143.50, params: 516_128, corrVeriteTerrain: 0.265 },
  { method: "conv_ae", precision: 0.010, recall: 0.573, f1: 0.019, auc: 0.589, leadTimeH: 3751.50, params: 19_331, corrVeriteTerrain: 0.250 },
  { method: "lstm_ae", precision: 0.024, recall: 0.469, f1: 0.046, auc: 0.652, leadTimeH: 4114.00, params: 26_163, corrVeriteTerrain: 0.153 },
  { method: "vae", precision: 0.006, recall: 0.156, f1: 0.012, auc: 0.602, leadTimeH: 3974.25, params: 259_808, corrVeriteTerrain: 0.062 },
];

export const productionFoulingMethod = { name: "gru_residual", corrVeriteTerrain: 0.490, leadTimeH: 3022.25 };

export const qualityCorrelations = {
  naphtha_final_boiling_point: 0.966,
  kerosene_flash_point: 0.965,
  gasoil_cetane_index: 0.972,
  residue_viscosity: 0.967,
  sulfur_content: 0.985,
};
