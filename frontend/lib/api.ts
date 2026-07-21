import type {
  Alert,
  EnergyOptimizeRequest,
  EnergyOptimizeResponse,
  FoulingStatus,
  HealthResponse,
  KpiSummary,
  SensorDetail,
  TwinState,
  YieldsHistory,
  YieldsPredictRequest,
  YieldsPredictResponse,
} from "@/lib/types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text || `Erreur ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
  kpiSummary: () => request<KpiSummary>("/api/kpi/summary"),
  yieldsHistory: (hours = 168) =>
    request<YieldsHistory>(`/api/yields/history?hours=${hours}`),
  yieldsPredict: (body: YieldsPredictRequest) =>
    request<YieldsPredictResponse>("/api/yields/predict", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  foulingStatus: () => request<FoulingStatus>("/api/fouling/status"),
  energyOptimize: (body: EnergyOptimizeRequest) =>
    request<EnergyOptimizeResponse>("/api/energy/optimize", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  twinState: () => request<TwinState>("/api/twin/state"),
  twinSensor: (id: string) => request<SensorDetail>(`/api/twin/sensor/${id}`),
  alerts: (limit = 50) => request<Alert[]>(`/api/alerts?limit=${limit}`),
  alertsActive: () => request<Alert[]>("/api/alerts/active"),
};

export function wsRealtimeUrl(): string {
  const httpUrl = new URL(API_BASE_URL);
  const proto = httpUrl.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${httpUrl.host}/ws/realtime`;
}
