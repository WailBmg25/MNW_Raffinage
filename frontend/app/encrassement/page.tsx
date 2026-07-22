"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { GaugeChart } from "@/components/gauge-chart";
import { TrendChart } from "@/components/trend-chart";
import { useFoulingStatus } from "@/lib/hooks";

export default function EncrassementPage() {
  const { data, isLoading, isError } = useFoulingStatus();

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">Détection de l&apos;encrassement (fouling)</h1>
        <p className="text-sm text-slate-500">
          Indice estimé par les réseaux de neurones vs vérité terrain cachée, objectif : détection &gt; 24h avant nettoyage.
        </p>
      </div>

      {isError && (
        <Card className="border-amber-500/30 bg-amber-500/5">
          <CardContent className="py-4 text-sm text-amber-300">Backend non joignable.</CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Card className="border-slate-800 bg-[#111827]">
          <CardContent className="flex items-center justify-center py-4">
            {isLoading || !data ? (
              <Skeleton className="h-28 w-28 rounded-full" />
            ) : (
              <GaugeChart
                value={data.current_index}
                max={data.threshold * 1.4}
                label={`Indice de fouling (tendance : ${data.trend})`}
                goodBelow={data.threshold * 0.6}
                warnBelow={data.threshold * 0.9}
              />
            )}
          </CardContent>
        </Card>
        <Card className="border-slate-800 bg-[#111827]">
          <CardContent className="flex flex-col items-center justify-center gap-1 py-4">
            <span className="text-xs uppercase tracking-wide text-slate-400">Nettoyage estimé</span>
            <span className="text-4xl font-semibold text-amber-400 tabular-nums">
              {data?.estimated_days_to_cleaning != null ? Math.round(data.estimated_days_to_cleaning) : "—"}
            </span>
            <span className="text-xs text-slate-500">jours</span>
          </CardContent>
        </Card>
      </div>

      <Card className="border-slate-800 bg-[#111827]">
        <CardHeader>
          <CardTitle className="text-sm font-medium text-slate-300">
            Historique {data ? Math.round(data.history.timestamps.length / 24) : "…"} jours — indice estimé vs vérité terrain
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading || !data ? (
            <Skeleton className="h-72 rounded-xl" />
          ) : (
            <TrendChart
              height={300}
              xKey="ts"
              data={data.history.timestamps.map((ts, i) => ({
                ts,
                "Indice estimé": data.history.index[i],
                "Vérité terrain (cachée)": data.history.hidden_truth[i],
              }))}
              xTickFormatter={(ts) =>
                new Date(ts).toLocaleDateString("fr-FR", { day: "2-digit", month: "short", year: "2-digit" })
              }
              series={[
                { key: "Indice estimé", label: "Indice estimé (réseau)", color: "#ef4444" },
                { key: "Vérité terrain (cachée)", label: "Vérité terrain (cachée, normalisée)", color: "#475569", strokeDasharray: "3 3" },
              ]}
              referenceLines={[{ y: data.threshold, label: "Seuil de nettoyage", color: "#f59e0b" }]}
            />
          )}
        </CardContent>
      </Card>

      <Card className="border-slate-800 bg-[#111827]">
        <CardHeader>
          <CardTitle className="text-sm font-medium text-slate-300">Épisodes détectés</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow className="border-slate-800">
                <TableHead>Début</TableHead>
                <TableHead>Détecté</TableHead>
                <TableHead>Nettoyage</TableHead>
                <TableHead className="text-right">Avance</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data?.episodes.map((ep) => (
                <TableRow key={ep.cleaning_at} className="border-slate-800">
                  <TableCell className="text-slate-400">{new Date(ep.start).toLocaleDateString("fr-FR")}</TableCell>
                  <TableCell className="text-slate-400">{new Date(ep.detected_at).toLocaleDateString("fr-FR")}</TableCell>
                  <TableCell className="text-slate-400">{new Date(ep.cleaning_at).toLocaleDateString("fr-FR")}</TableCell>
                  <TableCell className="text-right">
                    <Badge
                      className={
                        ep.lead_time_h > 24
                          ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                          : "border-amber-500/30 bg-amber-500/10 text-amber-300"
                      }
                    >
                      {ep.lead_time_h.toFixed(0)} h
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
              {!isLoading && (!data || data.episodes.length === 0) && (
                <TableRow>
                  <TableCell colSpan={4} className="text-center text-slate-500">
                    Aucun épisode disponible.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
