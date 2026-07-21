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
  { id: 2, label: "Détection du fouling (> 24h avant nettoyage)", achieved: true, value: "3764 h (autoencodeur dense)" },
  { id: 3, label: "Optimisation énergétique (gain > 5%)", achieved: true, value: "5.53 % (716 $/j, 4.13 tCO2/j)" },
  { id: 4, label: "Qualité produits (corrélation > 0.9)", achieved: true, value: "0.971 (GRU multi-sorties)" },
  { id: 5, label: "Alertes temps réel (latence < 1 min)", achieved: true, value: "~23 ms" },
];

export const foulingResults = [
  { method: "dense_ae", precision: 0.032, recall: 0.542, f1: 0.060, auc: 0.765, leadTimeH: 3764.25, params: 516_128 },
  { method: "lstm_ae", precision: 0.012, recall: 0.583, f1: 0.024, auc: 0.633, leadTimeH: 4128.75, params: 26_163 },
  { method: "vae", precision: 0.012, recall: 0.219, f1: 0.024, auc: 0.712, leadTimeH: 3413.75, params: 259_808 },
  { method: "conv_ae", precision: 0.011, recall: 0.615, f1: 0.022, auc: 0.624, leadTimeH: 3850.50, params: 19_331 },
  { method: "gru_residual", precision: 0.009, recall: 0.323, f1: 0.018, auc: 0.676, leadTimeH: 2971.75, params: 57_793 },
];

export const qualityCorrelations = {
  naphtha_final_boiling_point: 0.966,
  kerosene_flash_point: 0.965,
  gasoil_cetane_index: 0.972,
  residue_viscosity: 0.967,
  sulfur_content: 0.985,
};
