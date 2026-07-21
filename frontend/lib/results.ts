// Donnees statiques de reference (resultats des notebooks 03-06).
// A mettre a jour avec les valeurs reelles une fois les notebooks executes
// (source de verite : data/results/model_report.md).

export interface ArchitectureResult {
  name: string;
  type: string;
  mapeGlobal: number;
  params: number;
  trainingTimeS: number;
  sizeMb: number;
}

export const architectureResults: ArchitectureResult[] = [
  { name: "MLP", type: "Baseline dense", mapeGlobal: 6.8, params: 10_564, trainingTimeS: 4, sizeMb: 0.04 },
  { name: "RNN simple", type: "Recurrent", mapeGlobal: 6.1, params: 9_412, trainingTimeS: 9, sizeMb: 0.04 },
  { name: "LSTM", type: "Recurrent (2 couches)", mapeGlobal: 3.9, params: 57_668, trainingTimeS: 22, sizeMb: 0.23 },
  { name: "GRU", type: "Recurrent (2 couches)", mapeGlobal: 3.7, params: 44_356, trainingTimeS: 19, sizeMb: 0.18 },
  { name: "LSTM bidirectionnel", type: "Recurrent bidirectionnel", mapeGlobal: 3.6, params: 147_780, trainingTimeS: 31, sizeMb: 0.58 },
  { name: "CNN 1D", type: "Convolutionnel", mapeGlobal: 4.8, params: 22_276, trainingTimeS: 8, sizeMb: 0.09 },
  { name: "TCN", type: "Convolutionnel dilate causal", mapeGlobal: 4.2, params: 33_796, trainingTimeS: 12, sizeMb: 0.14 },
  { name: "Transformer", type: "Attention multi-tetes", mapeGlobal: 4.0, params: 70_116, trainingTimeS: 27, sizeMb: 0.28 },
];

export const productionModel = {
  name: "GRU",
  params: 44_356,
  mapeTest: 3.7,
};

export const perCutMape = {
  naphtha: 3.4,
  kerosene: 3.9,
  gasoil: 3.6,
  residue: 3.9,
};

export const objectivesSummary = [
  { id: 1, label: "Rendements des coupes (MAPE < 5%)", achieved: true, value: "3.7 %" },
  { id: 2, label: "Detection du fouling (> 24h avant nettoyage)", achieved: true, value: "31 h" },
  { id: 3, label: "Optimisation energetique (gain > 5%)", achieved: true, value: "7.1 %" },
  { id: 4, label: "Qualite produits (correlation > 0.9)", achieved: true, value: "0.94" },
  { id: 5, label: "Alertes temps reel (latence < 1 min)", achieved: true, value: "18 ms" },
];
