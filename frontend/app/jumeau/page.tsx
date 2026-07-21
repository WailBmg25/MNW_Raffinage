"use client";

import { useMemo, useState } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Cpu } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { EquipmentNode, type EquipmentNodeData } from "@/components/equipment-node";
import { SensorDrawer } from "@/components/sensor-drawer";
import { useTwinStore } from "@/lib/store";
import type { EquipmentHealth, TwinSensor } from "@/lib/types";

const nodeTypes = { equipment: EquipmentNode };

function sensorsFor(all: TwinSensor[], ids: string[]): TwinSensor[] {
  return ids
    .map((id) => all.find((s) => s.id === id))
    .filter((s): s is TwinSensor => !!s);
}

function healthFor(equipment: { id: string; health: EquipmentHealth }[], id: string): EquipmentHealth {
  return equipment.find((e) => e.id === id)?.health ?? "ok";
}

export default function JumeauPage() {
  const twinState = useTwinStore((s) => s.twinState);
  const [selectedSensor, setSelectedSensor] = useState<string | null>(null);

  const { nodes, edges } = useMemo(() => {
    const sensors = twinState?.sensors ?? [];
    const equipment = twinState?.equipment ?? [];
    const onSensorClick = (id: string) => setSelectedSensor(id);

    const mk = (
      id: string,
      x: number,
      y: number,
      data: Partial<EquipmentNodeData> & { label: string }
    ): Node => ({
      id,
      type: "equipment",
      position: { x, y },
      data: { health: "ok", ...data } as unknown as Record<string, unknown>,
    });

    const nodes: Node[] = [
      mk("brut", 0, 180, { label: "Brut", subtitle: "Alimentation" }),
      mk("dessaleur", 190, 180, { label: "Dessaleur", subtitle: "Dessalage" }),
      mk("preheat", 380, 180, {
        label: "Train de préchauffe",
        subtitle: "E-101 / E-102 / E-103",
        health: healthFor(equipment, "preheat_train"),
        sensors: sensorsFor(sensors, ["TI-301"]),
        onSensorClick,
      }),
      mk("furnace", 590, 180, {
        label: "Four F-101",
        subtitle: "COT",
        health: healthFor(equipment, "furnace"),
        sensors: sensorsFor(sensors, ["TI-201", "FI-301"]),
        onSensorClick,
      }),
      mk("column", 800, 180, {
        label: "Colonne CDU C-101",
        subtitle: "Distillation atmosphérique",
        health: healthFor(equipment, "column"),
        sensors: sensorsFor(sensors, ["TI-202", "PI-201", "FI-202", "FI-203"]),
        onSensorClick,
      }),
      mk("naphtha", 1030, 20, { label: "Naphta", variant: "stream" }),
      mk("kerosene", 1030, 110, { label: "Kérosène", variant: "stream" }),
      mk("gasoil", 1030, 250, { label: "Gazole", variant: "stream" }),
      mk("residue", 1030, 340, { label: "Résidu", variant: "stream" }),
      mk("cracker", 1230, 20, {
        label: "Vapocraqueur",
        subtitle: "Naphta → oléfines",
        health: healthFor(equipment, "cracker"),
        sensors: sensorsFor(sensors, ["TI-401", "FI-401"]),
        onSensorClick,
      }),
      mk("ethylene", 1450, -40, { label: "Éthylène", variant: "stream" }),
      mk("propylene", 1450, 60, { label: "Propylène", variant: "stream" }),
    ];

    const flow = twinState?.sensors.find((s) => s.id === "FI-101")?.value ?? 1;
    const strokeWidth = Math.min(6, Math.max(1.5, flow / 400));

    const baseEdge = (id: string, source: string, target: string, animated = true): Edge => ({
      id,
      source,
      target,
      animated,
      style: { stroke: "#0891b2", strokeWidth },
    });

    const edges: Edge[] = [
      baseEdge("e1", "brut", "dessaleur"),
      baseEdge("e2", "dessaleur", "preheat"),
      baseEdge("e3", "preheat", "furnace"),
      baseEdge("e4", "furnace", "column"),
      baseEdge("e5", "column", "naphtha"),
      baseEdge("e6", "column", "kerosene"),
      baseEdge("e7", "column", "gasoil"),
      baseEdge("e8", "column", "residue"),
      baseEdge("e9", "naphtha", "cracker"),
      baseEdge("e10", "cracker", "ethylene"),
      baseEdge("e11", "cracker", "propylene"),
    ];

    return { nodes, edges };
  }, [twinState]);

  return (
    <div className="flex h-[calc(100vh-6.5rem)] flex-col gap-4">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">Jumeau numérique — Synoptique du procédé</h1>
        <p className="text-sm text-slate-500">
          Cliquez sur un capteur ou un équipement pour voir sa tendance 7 jours et la comparaison
          mesuré / prédit.
        </p>
      </div>

      <Card className="flex-1 overflow-hidden border-slate-800 bg-[#0b1120] p-0">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          minZoom={0.3}
          maxZoom={1.5}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} color="#1e293b" gap={20} />
          <Controls className="!bg-slate-900 !text-slate-200 [&>button]:!border-slate-700 [&>button]:!bg-slate-900" />
          <MiniMap
            className="!bg-slate-900"
            maskColor="rgba(11,17,32,0.7)"
            nodeColor="#1e293b"
          />
        </ReactFlow>
      </Card>

      <Card className="border-slate-800 bg-[#111827]">
        <CardContent className="flex items-center gap-2 py-3 text-xs text-slate-400">
          <Cpu className="h-4 w-4 text-cyan-400" />
          Ce synoptique est piloté par 3 réseaux de neurones en inférence continue — latence mesurée :{" "}
          <span className="font-mono tabular-nums text-cyan-300">
            {twinState ? `${twinState.latency_ms.toFixed(1)} ms` : "—"}
          </span>
        </CardContent>
      </Card>

      <SensorDrawer sensorId={selectedSensor} onOpenChange={(open) => !open && setSelectedSensor(null)} />
    </div>
  );
}
