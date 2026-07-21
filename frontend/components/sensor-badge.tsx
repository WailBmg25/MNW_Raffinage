"use client";

import { cn } from "@/lib/utils";
import type { SensorStatus } from "@/lib/types";

const STATUS_STYLES: Record<SensorStatus, string> = {
  ok: "border-cyan-500/30 bg-cyan-500/10 text-cyan-200",
  warning: "border-amber-500/40 bg-amber-500/10 text-amber-200",
  alarm: "border-red-500/40 bg-red-500/10 text-red-200",
};

interface SensorBadgeProps {
  id: string;
  value: number;
  unit: string;
  status: SensorStatus;
  onClick?: () => void;
  decimals?: number;
}

export function SensorBadge({ id, value, unit, status, onClick, decimals = 1 }: SensorBadgeProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex items-center gap-1.5 rounded-lg border px-2 py-1 text-[11px] leading-none shadow-sm transition-transform hover:scale-105",
        STATUS_STYLES[status],
        onClick && "cursor-pointer"
      )}
    >
      <span className="font-semibold tracking-wide">{id}</span>
      <span className="tabular-nums font-mono">
        {value.toLocaleString("fr-FR", { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}
      </span>
      <span className="text-[10px] opacity-70">{unit}</span>
    </button>
  );
}
