"use client";

import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";
import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { AnimatedNumber } from "@/components/animated-number";
import { cn } from "@/lib/utils";

interface KpiCardProps {
  label: string;
  value: number;
  decimals?: number;
  suffix?: string;
  icon?: LucideIcon;
  delta?: number;
  deltaGoodDirection?: "up" | "down";
  accent?: "cyan" | "amber" | "emerald" | "red" | "violet";
  footer?: React.ReactNode;
}

const ACCENTS: Record<NonNullable<KpiCardProps["accent"]>, string> = {
  cyan: "text-cyan-400",
  amber: "text-amber-400",
  emerald: "text-emerald-400",
  red: "text-red-400",
  violet: "text-violet-400",
};

export function KpiCard({
  label,
  value,
  decimals = 1,
  suffix = "",
  icon: Icon,
  delta,
  deltaGoodDirection = "down",
  accent = "cyan",
  footer,
}: KpiCardProps) {
  const deltaIsGood =
    delta !== undefined &&
    (deltaGoodDirection === "down" ? delta <= 0 : delta >= 0);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
    >
      <Card className="border-slate-800 bg-[#111827] shadow-lg shadow-black/20">
        <CardContent className="flex flex-col gap-2 py-1">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-400">
              {label}
            </span>
            {Icon && <Icon className={cn("h-4 w-4", ACCENTS[accent])} />}
          </div>
          <div className={cn("text-3xl font-semibold tabular-nums", ACCENTS[accent])}>
            <AnimatedNumber value={value} decimals={decimals} suffix={suffix} />
          </div>
          <div className="flex items-center justify-between text-xs text-slate-400">
            {delta !== undefined ? (
              <span
                className={cn(
                  "inline-flex items-center gap-0.5 font-medium",
                  deltaIsGood ? "text-emerald-400" : "text-red-400"
                )}
              >
                {delta >= 0 ? (
                  <ArrowUpRight className="h-3.5 w-3.5" />
                ) : (
                  <ArrowDownRight className="h-3.5 w-3.5" />
                )}
                {Math.abs(delta).toFixed(1)}% vs baseline
              </span>
            ) : (
              <span />
            )}
            {footer}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
