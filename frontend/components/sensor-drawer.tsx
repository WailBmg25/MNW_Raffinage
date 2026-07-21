"use client";

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { TrendChart } from "@/components/trend-chart";
import { AlertFeed } from "@/components/alert-feed";
import { useTwinSensor } from "@/lib/hooks";

interface SensorDrawerProps {
  sensorId: string | null;
  onOpenChange: (open: boolean) => void;
}

export function SensorDrawer({ sensorId, onOpenChange }: SensorDrawerProps) {
  const { data, isLoading, isError } = useTwinSensor(sensorId);

  return (
    <Sheet open={!!sensorId} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full border-slate-800 bg-[#0b1120] sm:max-w-lg">
        <SheetHeader>
          <SheetTitle className="text-slate-100">{data?.name ?? sensorId}</SheetTitle>
          <SheetDescription className="text-slate-500">
            Tendance 7 jours — valeur mesurée vs valeur prédite par le réseau
          </SheetDescription>
        </SheetHeader>
        <div className="flex flex-col gap-4 px-4 pb-6">
          {isLoading && <Skeleton className="h-64 rounded-xl" />}
          {isError && (
            <p className="text-sm text-amber-400">
              Impossible de charger le détail de ce capteur (backend indisponible).
            </p>
          )}
          {data && (
            <>
              <TrendChart
                data={data.trend.timestamps.map((ts, i) => ({
                  ts: new Date(ts).toLocaleDateString("fr-FR", { day: "2-digit", month: "2-digit" }),
                  Mesuré: data.trend.measured[i],
                  Prédit: data.trend.predicted[i],
                }))}
                xKey="ts"
                height={240}
                series={[
                  { key: "Mesuré", label: "Mesuré", color: "#06b6d4" },
                  { key: "Prédit", label: "Prédit (réseau)", color: "#f59e0b", strokeDasharray: "4 4" },
                ]}
                referenceLines={[
                  { y: data.limits.high, label: "Limite haute", color: "#ef4444" },
                  { y: data.limits.low, label: "Limite basse", color: "#ef4444" },
                ]}
              />
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3 text-xs text-slate-400">
                Unité : <span className="text-slate-200">{data.unit}</span> · Bornes opératoires :{" "}
                <span className="text-slate-200">
                  [{data.limits.low}, {data.limits.high}]
                </span>
              </div>
              <div>
                <h4 className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
                  Alertes associées
                </h4>
                <AlertFeed alerts={data.alerts} emptyLabel="Aucune alerte associée à ce capteur" />
              </div>
            </>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
