// Types alignes sur le contrat API du backend FastAPI (voir Partie F de la specification).

export type CrudeType = "leger" | "moyen" | "lourd";
export type SensorStatus = "ok" | "warning" | "alarm";
export type EquipmentHealth = "ok" | "warning" | "alarm";
export type AlertLevel = "info" | "warning" | "critical";
export type AlertType = "fouling" | "yield_drift" | "quality" | "energy";
export type Trend = "up" | "down" | "stable";

export interface HealthResponse {
  status: "ok" | "degraded";
  device: "cpu" | "cuda";
  models_loaded: { yields: boolean; fouling: boolean; energy: boolean };
  timestamp: string;
}

export interface Objective {
  id: number;
  label: string;
  achieved: boolean;
  value: string;
}

export interface KpiSummary {
  date: string;
  feed_rate: number;
  distillate_yield_pct: number;
  specific_energy: number;
  specific_energy_baseline: number;
  specific_energy_delta_pct: number;
  fouling_index: number;
  fouling_days_before_cleaning: number | null;
  active_alerts: number;
  eur_saved_today: number;
  co2_avoided_today_t: number;
  objectives: Objective[];
}

export interface YieldSeries {
  naphtha: number[];
  kerosene: number[];
  gasoil: number[];
  residue: number[];
}

export interface YieldsHistory {
  timestamps: string[];
  actual: YieldSeries;
  predicted: YieldSeries;
}

export interface YieldsPredictRequest {
  feed_rate: number;
  crude_type: CrudeType;
  furnace_cot: number;
  reflux_ratio: number;
  stripping_steam?: number;
  column_top_temp?: number;
  column_top_pressure?: number;
}

export interface YieldsPredictResponse {
  naphtha_yield: number;
  kerosene_yield: number;
  gasoil_yield: number;
  residue_yield: number;
  model_name: string;
  mape_test: number;
}

export interface FoulingEpisode {
  start: string;
  detected_at: string;
  cleaning_at: string;
  lead_time_h: number;
}

export interface FoulingStatus {
  current_index: number;
  trend: Trend;
  estimated_days_to_cleaning: number | null;
  threshold: number;
  history: { timestamps: string[]; index: number[]; hidden_truth: number[] };
  cleanings: string[];
  episodes: FoulingEpisode[];
}

export type EnergyOptimizeRequest = YieldsPredictRequest;

export interface EnergyOptimizeResponse {
  cot_current: number;
  cot_recommended: number;
  reflux_current: number;
  reflux_recommended: number;
  gain_pct: number;
  eur_per_day: number;
  tco2_per_day: number;
  constraints_ok: boolean;
}

export interface TwinSensor {
  id: string;
  name: string;
  value: number;
  unit: string;
  status: SensorStatus;
  sparkline: number[];
}

export interface TwinEquipment {
  id: string;
  name: string;
  health: EquipmentHealth;
}

export interface TwinState {
  timestamp: string;
  latency_ms: number;
  sensors: TwinSensor[];
  equipment: TwinEquipment[];
  yields_stream: { naphtha: number; kerosene: number; gasoil: number; residue: number };
  active_alerts_count: number;
}

export interface SensorDetail {
  id: string;
  name: string;
  unit: string;
  trend: { timestamps: string[]; measured: number[]; predicted: number[] };
  limits: { low: number; high: number };
  alerts: Alert[];
}

export interface Alert {
  id: string;
  timestamp: string;
  level: AlertLevel;
  type: AlertType;
  equipment: string;
  message: string;
  value: number;
  recommendation: string;
  active: boolean;
}
