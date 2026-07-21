"""Schémas Pydantic v2 pour toutes les entrées/sorties de l'API (Partie F.2 de la spécification)."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

CrudeType = Literal["leger", "moyen", "lourd"]
AlertLevel = Literal["info", "warning", "critical"]
AlertType = Literal["fouling", "yield_drift", "quality", "energy"]
HealthStatus = Literal["ok", "warning", "alarm"]
Trend = Literal["up", "down", "stable"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    device: str
    models_loaded: dict[str, bool]
    timestamp: datetime


class Objective(BaseModel):
    id: int
    label: str
    achieved: bool
    value: str


class KpiSummary(BaseModel):
    date: datetime
    feed_rate: float
    distillate_yield_pct: float
    specific_energy: float
    specific_energy_baseline: float
    specific_energy_delta_pct: float
    fouling_index: float
    fouling_days_before_cleaning: float | None
    active_alerts: int
    usd_saved_today: float
    co2_avoided_today_t: float
    objectives: list[Objective]


class YieldSeries(BaseModel):
    naphtha: list[float]
    kerosene: list[float]
    gasoil: list[float]
    residue: list[float]


class YieldHistory(BaseModel):
    timestamps: list[datetime]
    actual: YieldSeries
    predicted: YieldSeries


class OperatingConditions(BaseModel):
    feed_rate: float = Field(..., gt=0)
    crude_type: CrudeType
    furnace_cot: float
    reflux_ratio: float
    stripping_steam: float | None = None
    column_top_temp: float | None = None
    column_top_pressure: float | None = None


class YieldPrediction(BaseModel):
    naphtha_yield: float
    kerosene_yield: float
    gasoil_yield: float
    residue_yield: float
    model_name: str
    mape_test: float


class FoulingEpisode(BaseModel):
    start: datetime
    detected_at: datetime
    cleaning_at: datetime
    lead_time_h: float


class FoulingHistory(BaseModel):
    timestamps: list[datetime]
    index: list[float]
    hidden_truth: list[float] | None = None


class FoulingStatus(BaseModel):
    current_index: float
    trend: Trend
    estimated_days_to_cleaning: float | None
    threshold: float
    history: FoulingHistory
    cleanings: list[datetime]
    episodes: list[FoulingEpisode]


class OptimizationResult(BaseModel):
    cot_current: float
    cot_recommended: float
    reflux_current: float
    reflux_recommended: float
    gain_pct: float
    usd_per_day: float
    tco2_per_day: float
    constraints_ok: bool


class SensorReading(BaseModel):
    id: str
    name: str
    value: float
    unit: str
    status: HealthStatus
    sparkline: list[float]


class EquipmentHealth(BaseModel):
    id: str
    name: str
    health: HealthStatus


class YieldsStream(BaseModel):
    naphtha: float
    kerosene: float
    gasoil: float
    residue: float


class TwinState(BaseModel):
    timestamp: datetime
    latency_ms: float
    sensors: list[SensorReading]
    equipment: list[EquipmentHealth]
    yields_stream: YieldsStream
    active_alerts_count: int


class SensorTrend(BaseModel):
    timestamps: list[datetime]
    measured: list[float]
    predicted: list[float]


class SensorLimits(BaseModel):
    low: float
    high: float


class Alert(BaseModel):
    id: str
    timestamp: datetime
    level: AlertLevel
    type: AlertType
    equipment: str
    message: str
    value: float
    recommendation: str
    active: bool


class SensorDetail(BaseModel):
    id: str
    name: str
    unit: str
    trend: SensorTrend
    limits: SensorLimits
    alerts: list[Alert]
