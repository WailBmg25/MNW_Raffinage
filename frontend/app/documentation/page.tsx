import { CheckCircle2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { architectureResults, objectivesSummary } from "@/lib/results";

export default function DocumentationPage() {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">Documentation</h1>
        <p className="text-sm text-slate-500">
          Rappel des objectifs, de l&apos;architecture du jumeau numérique et des résultats obtenus.
        </p>
      </div>

      <Card className="border-slate-800 bg-[#111827]">
        <CardHeader>
          <CardTitle className="text-sm font-medium text-slate-300">Objectifs &amp; résultats</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow className="border-slate-800">
                <TableHead>#</TableHead>
                <TableHead>Objectif</TableHead>
                <TableHead>Valeur atteinte</TableHead>
                <TableHead className="text-right">Statut</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {objectivesSummary.map((o) => (
                <TableRow key={o.id} className="border-slate-800">
                  <TableCell className="text-slate-500">{o.id}</TableCell>
                  <TableCell className="text-slate-200">{o.label}</TableCell>
                  <TableCell className="tabular-nums text-cyan-300">{o.value}</TableCell>
                  <TableCell className="text-right">
                    {o.achieved ? (
                      <Badge className="border-emerald-500/30 bg-emerald-500/10 text-emerald-300">
                        <CheckCircle2 className="mr-1 h-3 w-3" /> Atteint
                      </Badge>
                    ) : (
                      <Badge className="border-red-500/30 bg-red-500/10 text-red-300">Non atteint</Badge>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card className="border-slate-800 bg-[#111827]">
        <CardHeader>
          <CardTitle className="text-sm font-medium text-slate-300">Architecture du jumeau numérique</CardTitle>
        </CardHeader>
        <CardContent>
          <ArchitectureDiagram />
        </CardContent>
      </Card>

      <Card className="border-slate-800 bg-[#111827]">
        <CardHeader>
          <CardTitle className="text-sm font-medium text-slate-300">
            Comparatif des 8 architectures (prédiction des rendements)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow className="border-slate-800">
                <TableHead>Architecture</TableHead>
                <TableHead>Type</TableHead>
                <TableHead className="text-right">MAPE global</TableHead>
                <TableHead className="text-right">Paramètres</TableHead>
                <TableHead className="text-right">Temps entraînement</TableHead>
                <TableHead className="text-right">Taille</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {architectureResults.map((a) => (
                <TableRow key={a.name} className="border-slate-800">
                  <TableCell className="font-medium text-slate-200">{a.name}</TableCell>
                  <TableCell className="text-slate-400">{a.type}</TableCell>
                  <TableCell className={`text-right tabular-nums ${a.mapeGlobal < 5 ? "text-emerald-400" : "text-amber-400"}`}>
                    {a.mapeGlobal.toFixed(1)}%
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-slate-400">
                    {a.params.toLocaleString("fr-FR")}
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-slate-400">{a.trainingTimeS}s</TableCell>
                  <TableCell className="text-right tabular-nums text-slate-400">{a.sizeMb.toFixed(2)} Mo</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <p className="mt-3 text-xs text-slate-500">
            Données de référence — voir <code>data/results/model_report.md</code> pour les valeurs
            exactes générées par le notebook 03.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

function ArchitectureDiagram() {
  const steps = [
    "Générateur de données synthétiques",
    "Préprocessing (nettoyage, features, séquences)",
    "Modèles PyTorch (rendements, fouling, énergie)",
    "Backend FastAPI (inférence + websocket)",
    "Frontend Next.js (jumeau numérique)",
  ];
  return (
    <div className="flex flex-wrap items-center gap-2">
      {steps.map((s, i) => (
        <div key={s} className="flex items-center gap-2">
          <div className="rounded-xl border border-cyan-500/30 bg-cyan-500/5 px-3 py-2 text-xs text-cyan-200">
            {s}
          </div>
          {i < steps.length - 1 && <span className="text-slate-600">→</span>}
        </div>
      ))}
    </div>
  );
}
