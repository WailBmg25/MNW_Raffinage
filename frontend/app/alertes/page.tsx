"use client";

import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { useAlerts } from "@/lib/hooks";
import type { Alert, AlertLevel } from "@/lib/types";

const LEVEL_BADGE: Record<AlertLevel, string> = {
  info: "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
  warning: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  critical: "border-red-500/30 bg-red-500/10 text-red-300",
};

export default function AlertesPage() {
  const { data, isLoading, isError } = useAlerts(100);
  const [levelFilter, setLevelFilter] = useState<string>("all");
  const [selected, setSelected] = useState<Alert | null>(null);

  const filtered = useMemo(() => {
    if (!data) return [];
    return levelFilter === "all" ? data : data.filter((a) => a.level === levelFilter);
  }, [data, levelFilter]);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-slate-100">Journal des alertes</h1>
          <p className="text-sm text-slate-500">Fouling, dérive de rendement, qualité hors spec, énergie anormale.</p>
        </div>
        <Select value={levelFilter} onValueChange={(v) => setLevelFilter(v ?? "all")}>
          <SelectTrigger className="w-44 border-slate-700 bg-slate-900 text-slate-200">
            <SelectValue placeholder="Filtrer par niveau" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">Tous niveaux</SelectItem>
            <SelectItem value="info">Info</SelectItem>
            <SelectItem value="warning">Avertissement</SelectItem>
            <SelectItem value="critical">Critique</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {isError && (
        <Card className="border-amber-500/30 bg-amber-500/5">
          <CardContent className="py-4 text-sm text-amber-300">Backend non joignable.</CardContent>
        </Card>
      )}

      <Card className="border-slate-800 bg-[#111827]">
        <CardHeader>
          <CardTitle className="text-sm font-medium text-slate-300">
            {filtered.length} alerte(s)
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <Skeleton className="h-72 rounded-xl" />
          ) : (
            <Table>
              <TableHeader>
                <TableRow className="border-slate-800">
                  <TableHead>Horodatage</TableHead>
                  <TableHead>Niveau</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Équipement</TableHead>
                  <TableHead>Statut</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map((alert) => (
                  <TableRow
                    key={alert.id}
                    onClick={() => setSelected(alert)}
                    className="cursor-pointer border-slate-800 hover:bg-slate-800/40"
                  >
                    <TableCell className="text-slate-400">
                      {new Date(alert.timestamp).toLocaleString("fr-FR")}
                    </TableCell>
                    <TableCell>
                      <Badge className={LEVEL_BADGE[alert.level]}>{alert.level}</Badge>
                    </TableCell>
                    <TableCell className="text-slate-300">{alert.type}</TableCell>
                    <TableCell className="text-slate-300">{alert.equipment}</TableCell>
                    <TableCell>
                      <Badge
                        className={
                          alert.active
                            ? "border-red-500/30 bg-red-500/10 text-red-300"
                            : "border-slate-700 bg-slate-800 text-slate-400"
                        }
                      >
                        {alert.active ? "Active" : "Résolue"}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
                {filtered.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={5} className="text-center text-slate-500">
                      Aucune alerte pour ce filtre.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Dialog open={!!selected} onOpenChange={(open) => !open && setSelected(null)}>
        <DialogContent className="border-slate-800 bg-[#111827] text-slate-200">
          <DialogHeader>
            <DialogTitle>Détail de l&apos;alerte</DialogTitle>
            <DialogDescription className="text-slate-500">{selected?.equipment}</DialogDescription>
          </DialogHeader>
          {selected && (
            <div className="flex flex-col gap-3 text-sm">
              <Badge className={LEVEL_BADGE[selected.level]}>{selected.level}</Badge>
              <p>{selected.message}</p>
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3">
                <p className="text-xs text-slate-500">Valeur observée</p>
                <p className="text-lg font-semibold tabular-nums text-cyan-300">{selected.value}</p>
              </div>
              <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-3">
                <p className="text-xs text-emerald-400">Recommandation</p>
                <p className="text-emerald-200">{selected.recommendation}</p>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
