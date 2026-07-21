"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { EnergyOptimizeRequest, YieldsPredictRequest } from "@/lib/types";

export function useKpiSummary() {
  return useQuery({ queryKey: ["kpi-summary"], queryFn: api.kpiSummary, refetchInterval: 15_000 });
}

export function useYieldsHistory(hours = 168) {
  return useQuery({
    queryKey: ["yields-history", hours],
    queryFn: () => api.yieldsHistory(hours),
    refetchInterval: 30_000,
  });
}

export function useYieldsPredict() {
  return useMutation({
    mutationFn: (body: YieldsPredictRequest) => api.yieldsPredict(body),
  });
}

export function useFoulingStatus() {
  return useQuery({ queryKey: ["fouling-status"], queryFn: api.foulingStatus, refetchInterval: 30_000 });
}

export function useEnergyOptimize() {
  return useMutation({
    mutationFn: (body: EnergyOptimizeRequest) => api.energyOptimize(body),
  });
}

export function useTwinSensor(id: string | null) {
  return useQuery({
    queryKey: ["twin-sensor", id],
    queryFn: () => api.twinSensor(id as string),
    enabled: !!id,
  });
}

export function useAlerts(limit = 50) {
  return useQuery({ queryKey: ["alerts", limit], queryFn: () => api.alerts(limit), refetchInterval: 20_000 });
}

export function useAlertsActive() {
  return useQuery({ queryKey: ["alerts-active"], queryFn: api.alertsActive, refetchInterval: 15_000 });
}

export function useHealth() {
  return useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 30_000, retry: 1 });
}
