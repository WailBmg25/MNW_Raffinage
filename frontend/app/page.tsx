"use client";

import { Activity, CheckCircle2, Flame, Fuel, Gauge as GaugeIcon, Leaf, Zap } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { KpiCard } from "@/components/kpi-card";
import { GaugeChart } from "@/components/gauge-chart";
import { TrendChart } from "@/components/trend-chart";
import { AlertFeed } from "@/components/alert-feed";
import { AnimatedNumber } from "@/components/animated-number";
import { useKpiSummary, useAlertsActive } from "@/lib/hooks";
import { useTwinStore } from "@/lib/store";

export default function CommandCenterPage() {
  const { data: kpi, isLoading, isError } = useKpiSummary();
  const { data: activeAlerts } = useAlertsActive();
  const yieldsStreamHistory = useTwinStore((s) => s.yieldsStreamHistory);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">Vue d&apos;ensemble — Command Center</h1>
        <p className="text-sm text-slate-500">
          État consolidé du jumeau numérique CDU &amp; Vapocraqueur en temps réel.
        </p>
      </div>

      {isError && (
        <Card className="border-amber-500/30 bg-amber-500/5">
          <CardContent className="py-4 text-sm text-amber-300">
            Impossible de contacter l&apos;API backend ({process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}).
            Vérifiez que le service FastAPI est démarré.
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
        {isLoading || !kpi ? (
          Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-28 rounded-2xl" />)
        ) : (
          <>
            <KpiCard label="Débit de charge" value={kpi.feed_rate} decimals={0} suffix=" m³/h" icon={Fuel} accent="cyan" />
            <KpiCard
              label="Rendement distillats"
              value={kpi.distillate_yield_pct}
              decimals={1}
              suffix=" %"
              icon={Activity}
              accent="cyan"
            />
            <KpiCard
              label="Énergie spécifique"
              value={kpi.specific_energy}
              decimals={2}
              suffix=" kWh/bbl"
              icon={Zap}
              accent="emerald"
              delta={kpi.specific_energy_delta_pct}
              deltaGoodDirection="down"
            />
            <Card className="border-slate-800 bg-[#111827]">
              <CardContent className="flex items-center justify-center py-1">
                <GaugeChart
                  value={kpi.fouling_index}
                  max={1}
                  label="Indice de fouling"
                  goodBelow={0.4}
                  warnBelow={0.7}
                  size={110}
                />
              </CardContent>
            </Card>
            <KpiCard label="Alertes actives" value={kpi.active_alerts} decimals={0} icon={GaugeIcon} accent="amber" />
            <Card className="border-slate-800 bg-[#111827]">
              <CardContent className="flex h-full flex-col justify-center gap-1 py-1">
                <span className="text-xs font-medium uppercase tracking-wide text-slate-400">
                  Objectifs IA
                </span>
                <span className="flex items-center gap-1.5 text-2xl font-semibold text-emerald-400">
                  <CheckCircle2 className="h-5 w-5" />
                  {kpi.objectives.filter((o) => o.achieved).length}/{kpi.objectives.length} ✅
                </span>
              </CardContent>
            </Card>
          </>
        )}
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <Card className="border-slate-800 bg-[#111827] xl:col-span-2">
          <CardHeader>
            <CardTitle className="text-sm font-medium text-slate-300">
              Rendements des coupes — fenêtre glissante 48h (flux temps réel)
            </CardTitle>
          </CardHeader>
          <CardContent>
            {yieldsStreamHistory.length === 0 ? (
              <Skeleton className="h-64 rounded-xl" />
            ) : (
              <TrendChart
                data={yieldsStreamHistory.map((p) => ({
                  ts: p.timestamp,
                  Naphta: p.naphtha,
                  Kerosene: p.kerosene,
                  Gazole: p.gasoil,
                  Residu: p.residue,
                }))}
                xKey="ts"
                xTickFormatter={(ts) =>
                  new Date(ts).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" })
                }
                variant="area"
                height={280}
                series={[
                  { key: "Naphta", label: "Naphta", color: "#06b6d4" },
                  { key: "Kerosene", label: "Kérosène", color: "#f59e0b" },
                  { key: "Gazole", label: "Gazole", color: "#10b981" },
                  { key: "Residu", label: "Résidu", color: "#7c3aed" },
                ]}
              />
            )}
          </CardContent>
        </Card>

        <div className="flex flex-col gap-4">
          <Card className="border-slate-800 bg-[#111827]">
            <CardHeader>
              <CardTitle className="text-sm font-medium text-slate-300">Alertes en direct</CardTitle>
            </CardHeader>
            <CardContent>
              <AlertFeed alerts={activeAlerts ?? []} />
            </CardContent>
          </Card>

          <div className="grid grid-cols-2 gap-4">
            <Card className="border-emerald-500/20 bg-emerald-500/5">
              <CardContent className="flex flex-col gap-1 py-3">
                <span className="flex items-center gap-1.5 text-xs text-emerald-300">
                  <Leaf className="h-3.5 w-3.5" /> Gain énergie
                </span>
                <span className="text-xl font-semibold text-emerald-400">
                  <AnimatedNumber value={kpi?.usd_saved_today ?? 0} decimals={0} suffix=" $" />
                </span>
              </CardContent>
            </Card>
            <Card className="border-emerald-500/20 bg-emerald-500/5">
              <CardContent className="flex flex-col gap-1 py-3">
                <span className="flex items-center gap-1.5 text-xs text-emerald-300">
                  <Flame className="h-3.5 w-3.5" /> CO₂ évité
                </span>
                <span className="text-xl font-semibold text-emerald-400">
                  <AnimatedNumber value={kpi?.co2_avoided_today_t ?? 0} decimals={1} suffix=" t" />
                </span>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
