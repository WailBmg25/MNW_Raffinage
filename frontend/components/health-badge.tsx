"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import type { EquipmentHealth } from "@/lib/types";

const STYLES: Record<EquipmentHealth, { label: string; dot: string; text: string; bg: string }> = {
  ok: { label: "Sain", dot: "bg-emerald-400", text: "text-emerald-300", bg: "bg-emerald-500/10 border-emerald-500/30" },
  warning: { label: "Attention", dot: "bg-amber-400", text: "text-amber-300", bg: "bg-amber-500/10 border-amber-500/30" },
  alarm: { label: "Alarme", dot: "bg-red-400", text: "text-red-300", bg: "bg-red-500/10 border-red-500/30" },
};

export function HealthBadge({ health, className }: { health: EquipmentHealth; className?: string }) {
  const s = STYLES[health];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium",
        s.bg,
        s.text,
        className
      )}
    >
      <motion.span
        className={cn("h-1.5 w-1.5 rounded-full", s.dot)}
        animate={health === "alarm" ? { opacity: [1, 0.3, 1] } : { opacity: 1 }}
        transition={health === "alarm" ? { duration: 1.1, repeat: Infinity } : undefined}
      />
      {s.label}
    </span>
  );
}
