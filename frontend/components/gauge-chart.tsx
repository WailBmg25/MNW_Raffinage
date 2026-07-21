"use client";

import { RadialBar, RadialBarChart, PolarAngleAxis } from "recharts";

interface GaugeChartProps {
  value: number;
  max: number;
  label: string;
  unit?: string;
  goodBelow?: number;
  warnBelow?: number;
  size?: number;
}

/** Jauge radiale style instrumentation : vert/ambre/rouge selon des seuils. */
export function GaugeChart({
  value,
  max,
  label,
  unit = "",
  goodBelow,
  warnBelow,
  size = 140,
}: GaugeChartProps) {
  const pct = Math.max(0, Math.min(1, value / max));
  let color = "#06b6d4";
  if (goodBelow !== undefined && warnBelow !== undefined) {
    color = value <= goodBelow ? "#10b981" : value <= warnBelow ? "#f59e0b" : "#ef4444";
  }

  const data = [{ name: label, value: pct * 100, fill: color }];

  return (
    <div className="flex flex-col items-center gap-1" style={{ width: size }}>
      <div className="relative" style={{ width: size, height: size }}>
        <RadialBarChart
          width={size}
          height={size}
          cx={size / 2}
          cy={size / 2}
          innerRadius={size * 0.34}
          outerRadius={size * 0.48}
          barSize={size * 0.12}
          data={data}
          startAngle={90}
          endAngle={-270}
        >
          <PolarAngleAxis type="number" domain={[0, 100]} tick={false} axisLine={false} />
          <RadialBar dataKey="value" cornerRadius={20} background={{ fill: "#1e293b" }} />
        </RadialBarChart>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xl font-semibold tabular-nums" style={{ color }}>
            {value.toLocaleString("fr-FR", { maximumFractionDigits: 2 })}
            <span className="ml-0.5 text-xs text-slate-400">{unit}</span>
          </span>
        </div>
      </div>
      <span className="text-center text-xs text-slate-400">{label}</span>
    </div>
  );
}
