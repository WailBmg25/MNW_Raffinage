"use client";

import { Handle, Position, type NodeProps } from "@xyflow/react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { HealthBadge } from "@/components/health-badge";
import type { EquipmentHealth, TwinSensor } from "@/lib/types";

export interface EquipmentNodeData {
  label: string;
  subtitle?: string;
  health: EquipmentHealth;
  sensors?: TwinSensor[];
  icon?: React.ReactNode;
  onSensorClick?: (id: string) => void;
  onEquipmentClick?: () => void;
  variant?: "equipment" | "stream";
  [key: string]: unknown;
}

const HEALTH_RING: Record<EquipmentHealth, string> = {
  ok: "ring-emerald-500/40",
  warning: "ring-amber-500/50",
  alarm: "ring-red-500/60",
};

export function EquipmentNode({ data }: NodeProps) {
  const d = data as unknown as EquipmentNodeData;
  const isStream = d.variant === "stream";

  return (
    <div
      className={cn(
        "rounded-2xl border bg-[#111827] px-3 py-2.5 shadow-lg shadow-black/30 ring-1 transition-shadow",
        isStream ? "w-[120px] border-slate-800/70" : "w-[190px] border-slate-700",
        HEALTH_RING[d.health]
      )}
      onClick={d.onEquipmentClick}
      role={d.onEquipmentClick ? "button" : undefined}
    >
      <Handle type="target" position={Position.Left} className="!bg-cyan-500" />
      <div className="flex items-center justify-between gap-2">
        <span className={cn("font-medium text-slate-100", isStream ? "text-[11px]" : "text-xs")}>
          {d.label}
        </span>
        {!isStream && (
          <motion.span
            animate={d.health === "alarm" ? { scale: [1, 1.15, 1] } : {}}
            transition={{ duration: 1.2, repeat: Infinity }}
          >
            <HealthBadge health={d.health} />
          </motion.span>
        )}
      </div>
      {d.subtitle && <p className="mt-0.5 text-[10px] text-slate-500">{d.subtitle}</p>}
      {d.sensors && d.sensors.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {d.sensors.map((s) => (
            <button
              key={s.id}
              onClick={(e) => {
                e.stopPropagation();
                d.onSensorClick?.(s.id);
              }}
              className={cn(
                "rounded-md border px-1.5 py-0.5 font-mono text-[10px] tabular-nums transition-transform hover:scale-105",
                s.status === "ok" && "border-cyan-500/30 bg-cyan-500/10 text-cyan-200",
                s.status === "warning" && "border-amber-500/40 bg-amber-500/10 text-amber-200",
                s.status === "alarm" && "border-red-500/40 bg-red-500/10 text-red-200"
              )}
              title={s.name}
            >
              {s.id}: {s.value.toFixed(1)}
              {s.unit}
            </button>
          ))}
        </div>
      )}
      <Handle type="source" position={Position.Right} className="!bg-cyan-500" />
    </div>
  );
}
