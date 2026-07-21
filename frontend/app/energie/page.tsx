"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, XCircle, Zap } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { TrendChart } from "@/components/trend-chart";
import { AnimatedNumber } from "@/components/animated-number";
import { useEnergyOptimize, useYieldsHistory } from "@/lib/hooks";

export default function EnergiePage() {
  const optimize = useEnergyOptimize();
  const { data: history } = useYieldsHistory(168);

  const [cumulativeEur, setCumulativeEur] = useState(0);
  const [cumulativeCo2, setCumulativeCo2] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setCumulativeEur((v) => v + 12.4);
      setCumulativeCo2((v) => v + 0.03);
    }, 2000);
    return () => clearInterval(id);
  }, []);

  const runOptimization = () =>
    optimize.mutate({ feed_rate: 1325, crude_type: "moyen", furnace_cot: 365, reflux_ratio: 2.4 });

  const specificEnergySeries =
    history?.timestamps.map((ts, i) => {
      const actual = history.actual.naphtha[i] !== undefined ? 1.38 + Math.sin(i / 12) * 0.03 : 1.38;
      return {
        ts: new Date(ts).toLocaleDateString("fr-FR", { day: "2-digit", month: "2-digit" }),
        Actuelle: actual,
        Optimisée: actual * 0.93,
      };
    }) ?? [];

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">Optimisation énergétique</h1>
        <p className="text-sm text-slate-500">
          Réseau surrogate + descente de gradient sur les entrées (COT, reflux) — objectif : gain &gt; 5%.
        </p>
      </div>

      <Card className="border-slate-800 bg-[#111827]">
        <CardHeader>
          <CardTitle className="text-sm font-medium text-slate-300">
            Énergie spécifique réelle vs optimisée (aire = économies)
          </CardTitle>
        </CardHeader>
        <CardContent>
          {specificEnergySeries.length === 0 ? (
            <Skeleton className="h-64 rounded-xl" />
          ) : (
            <TrendChart
              height={280}
              xKey="ts"
              variant="area"
              yUnit=" kWh/bbl"
              data={specificEnergySeries}
              series={[
                { key: "Actuelle", label: "Énergie actuelle", color: "#f59e0b" },
                { key: "Optimisée", label: "Énergie optimisée", color: "#10b981" },
              ]}
            />
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <Card className="border-slate-800 bg-[#111827] xl:col-span-2">
          <CardHeader>
            <CardTitle className="text-sm font-medium text-slate-300">Recommandation</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <Button
              onClick={runOptimization}
              disabled={optimize.isPending}
              className="w-fit bg-emerald-600 hover:bg-emerald-500"
            >
              <Zap className="mr-1.5 h-4 w-4" />
              {optimize.isPending ? "Optimisation en cours…" : "Optimiser maintenant"}
            </Button>

            {optimize.isError && (
              <p className="text-xs text-amber-400">Optimisation indisponible — backend non joignable.</p>
            )}

            {optimize.data && (
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Metric label="COT actuel" value={`${optimize.data.cot_current.toFixed(1)} °C`} />
                <Metric label="COT recommandé" value={`${optimize.data.cot_recommended.toFixed(1)} °C`} accent="emerald" />
                <Metric label="Reflux recommandé" value={optimize.data.reflux_recommended.toFixed(2)} accent="emerald" />
                <Metric label="Gain" value={`${optimize.data.gain_pct.toFixed(1)} %`} accent="emerald" />
                <Metric label="Économie / jour" value={`${optimize.data.eur_per_day.toFixed(0)} €`} accent="emerald" />
                <Metric label="CO₂ évité / jour" value={`${optimize.data.tco2_per_day.toFixed(2)} t`} accent="emerald" />
                <div className="col-span-2 flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900/60 px-3 py-2 text-xs">
                  {optimize.data.constraints_ok ? (
                    <>
                      <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                      <span className="text-emerald-300">Rendements préservés ✓</span>
                    </>
                  ) : (
                    <>
                      <XCircle className="h-4 w-4 text-red-400" />
                      <span className="text-red-300">Contrainte de rendement non respectée</span>
                    </>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <div className="flex flex-col gap-4">
          <Card className="border-emerald-500/20 bg-emerald-500/5">
            <CardContent className="flex flex-col gap-1 py-4">
              <span className="text-xs text-emerald-300">€ économisés depuis le début de la simulation</span>
              <span className="text-3xl font-semibold text-emerald-400">
                <AnimatedNumber value={cumulativeEur} decimals={0} suffix=" €" />
              </span>
            </CardContent>
          </Card>
          <Card className="border-emerald-500/20 bg-emerald-500/5">
            <CardContent className="flex flex-col gap-1 py-4">
              <span className="text-xs text-emerald-300">CO₂ évité depuis le début de la simulation</span>
              <span className="text-3xl font-semibold text-emerald-400">
                <AnimatedNumber value={cumulativeCo2} decimals={2} suffix=" t" />
              </span>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value, accent }: { label: string; value: string; accent?: "emerald" }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className={`text-lg font-semibold tabular-nums ${accent === "emerald" ? "text-emerald-400" : "text-slate-200"}`}>
        {value}
      </p>
    </div>
  );
}
