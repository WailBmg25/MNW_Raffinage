"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { TrendChart } from "@/components/trend-chart";
import { GaugeChart } from "@/components/gauge-chart";
import { useYieldsHistory, useYieldsPredict } from "@/lib/hooks";
import { perCutMape, productionModel } from "@/lib/results";
import type { CrudeType } from "@/lib/types";

const CUTS: { key: "naphtha" | "kerosene" | "gasoil" | "residue"; label: string; color: string }[] = [
  { key: "naphtha", label: "Naphta", color: "#06b6d4" },
  { key: "kerosene", label: "Kérosène", color: "#f59e0b" },
  { key: "gasoil", label: "Gazole", color: "#10b981" },
  { key: "residue", label: "Résidu", color: "#7c3aed" },
];

export default function RendementsPage() {
  const { data: history, isLoading, isError } = useYieldsHistory(168);
  const predict = useYieldsPredict();

  const [feedRate, setFeedRate] = useState(1325);
  const [cot, setCot] = useState(365);
  const [reflux, setReflux] = useState(2.4);
  const [crudeType, setCrudeType] = useState<CrudeType>("moyen");

  const runSimulation = () =>
    predict.mutate({ feed_rate: feedRate, crude_type: crudeType, furnace_cot: cot, reflux_ratio: reflux });

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">Rendements des coupes</h1>
        <p className="text-sm text-slate-500">
          Comparaison prédit vs réel et simulateur what-if piloté par le modèle en production.
        </p>
      </div>

      {isError && (
        <Card className="border-amber-500/30 bg-amber-500/5">
          <CardContent className="py-4 text-sm text-amber-300">
            Historique indisponible — backend non joignable.
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
        {CUTS.map((cut) => (
          <Card key={cut.key} className="border-slate-800 bg-[#111827]">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-slate-300">{cut.label}</CardTitle>
            </CardHeader>
            <CardContent>
              {isLoading || !history ? (
                <Skeleton className="h-40 rounded-lg" />
              ) : (
                <TrendChart
                  height={160}
                  xKey="ts"
                  data={history.timestamps.map((ts, i) => ({
                    ts: new Date(ts).toLocaleDateString("fr-FR", { day: "2-digit", month: "2-digit" }),
                    Réel: history.actual[cut.key][i],
                    Prédit: history.predicted[cut.key][i],
                  }))}
                  series={[
                    { key: "Réel", label: "Réel", color: cut.color },
                    { key: "Prédit", label: "Prédit", color: "#94a3b8", strokeDasharray: "4 4" },
                  ]}
                />
              )}
              <div className="mt-2 flex items-center justify-center">
                <GaugeChart
                  value={perCutMape[cut.key]}
                  max={10}
                  label="MAPE (objectif < 5%)"
                  unit=" %"
                  goodBelow={5}
                  warnBelow={7}
                  size={100}
                />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <Card className="border-slate-800 bg-[#111827] xl:col-span-2">
          <CardHeader>
            <CardTitle className="text-sm font-medium text-slate-300">Simulateur what-if</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-5">
            <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
              <SliderField label="Débit de charge" value={feedRate} min={1000} max={1600} step={5} unit=" m³/h" onChange={setFeedRate} />
              <SliderField label="COT four" value={cot} min={355} max={375} step={0.5} unit=" °C" onChange={setCot} />
              <SliderField label="Taux de reflux" value={reflux} min={1.8} max={3.2} step={0.05} unit="" onChange={setReflux} />
              <div className="flex flex-col gap-2">
                <label className="text-xs font-medium text-slate-400">Type de brut</label>
                <Select value={crudeType} onValueChange={(v) => setCrudeType(v as CrudeType)}>
                  <SelectTrigger className="border-slate-700 bg-slate-900 text-slate-200">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="leger">Léger</SelectItem>
                    <SelectItem value="moyen">Moyen</SelectItem>
                    <SelectItem value="lourd">Lourd</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <Button onClick={runSimulation} disabled={predict.isPending} className="w-fit bg-cyan-600 hover:bg-cyan-500">
              {predict.isPending ? "Calcul en cours…" : "Simuler"}
            </Button>

            {predict.isError && (
              <p className="text-xs text-amber-400">Simulation indisponible — backend non joignable.</p>
            )}

            {predict.data && (
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                {(
                  [
                    ["Naphta", predict.data.naphtha_yield, "#06b6d4"],
                    ["Kérosène", predict.data.kerosene_yield, "#f59e0b"],
                    ["Gazole", predict.data.gasoil_yield, "#10b981"],
                    ["Résidu", predict.data.residue_yield, "#7c3aed"],
                  ] as [string, number, string][]
                ).map(([label, value, color]) => (
                  <div key={label} className="rounded-xl border border-slate-800 bg-slate-900/60 p-3 text-center">
                    <p className="text-xs text-slate-500">{label}</p>
                    <p className="text-lg font-semibold tabular-nums" style={{ color }}>
                      {(value * 100).toFixed(1)}%
                    </p>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="border-slate-800 bg-[#111827]">
          <CardHeader>
            <CardTitle className="text-sm font-medium text-slate-300">Modèle en production</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3 text-sm">
            <Row label="Architecture" value={productionModel.name} />
            <Row label="Paramètres" value={productionModel.params.toLocaleString("fr-FR")} />
            <Row label="MAPE test (global)" value={`${productionModel.mapeTest}%`} valueClass="text-emerald-400" />
            <p className="pt-2 text-xs text-slate-500">
              Sélectionné à l&apos;issue du grand comparatif de 8 architectures (notebook 03) sur le
              critère MAPE global le plus bas tout en respectant l&apos;objectif &lt; 5% par coupe.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium text-slate-400">{label}</span>
        <span className="font-mono tabular-nums text-cyan-300">
          {value}
          {unit}
        </span>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={(v) => onChange(Array.isArray(v) ? v[0] : v)}
      />
    </div>
  );
}

function Row({ label, value, valueClass }: { label: string; value: string; valueClass?: string }) {
  return (
    <div className="flex items-center justify-between border-b border-slate-800 pb-2">
      <span className="text-slate-500">{label}</span>
      <span className={`font-medium ${valueClass ?? "text-slate-200"}`}>{value}</span>
    </div>
  );
}
