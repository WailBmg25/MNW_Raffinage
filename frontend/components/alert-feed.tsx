"use client";

import { motion } from "framer-motion";
import { AlertTriangle, Info, ShieldAlert } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Alert, AlertLevel } from "@/lib/types";

const LEVEL_STYLES: Record<AlertLevel, { icon: typeof Info; text: string; bg: string; label: string }> = {
  info: { icon: Info, text: "text-cyan-300", bg: "bg-cyan-500/10 border-cyan-500/30", label: "Info" },
  warning: { icon: AlertTriangle, text: "text-amber-300", bg: "bg-amber-500/10 border-amber-500/30", label: "Avertissement" },
  critical: { icon: ShieldAlert, text: "text-red-300", bg: "bg-red-500/10 border-red-500/30", label: "Critique" },
};

export function AlertFeed({ alerts, emptyLabel = "Aucune alerte active" }: { alerts: Alert[]; emptyLabel?: string }) {
  if (alerts.length === 0) {
    return <p className="py-6 text-center text-sm text-slate-500">{emptyLabel}</p>;
  }

  return (
    <ul className="flex flex-col gap-2">
      {alerts.map((alert, i) => {
        const s = LEVEL_STYLES[alert.level];
        const Icon = s.icon;
        return (
          <motion.li
            key={alert.id}
            initial={{ opacity: 0, x: 12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.25, delay: i * 0.03 }}
            className={cn("flex items-start gap-2 rounded-xl border px-3 py-2 text-xs", s.bg)}
          >
            <Icon className={cn("mt-0.5 h-3.5 w-3.5 shrink-0", s.text)} />
            <div className="flex flex-col gap-0.5">
              <div className="flex items-center gap-2">
                <span className={cn("font-medium", s.text)}>{s.label}</span>
                <span className="text-slate-500">{alert.equipment}</span>
                <span className="ml-auto text-slate-500">
                  {new Date(alert.timestamp).toLocaleTimeString("fr-FR")}
                </span>
              </div>
              <p className="text-slate-300">{alert.message}</p>
            </div>
          </motion.li>
        );
      })}
    </ul>
  );
}
