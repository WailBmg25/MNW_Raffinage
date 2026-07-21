"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface TrendSeriesConfig {
  key: string;
  label: string;
  color: string;
  strokeDasharray?: string;
}

interface TrendChartProps {
  data: Record<string, number | string>[];
  xKey: string;
  series: TrendSeriesConfig[];
  height?: number;
  variant?: "line" | "area";
  yUnit?: string;
  referenceLines?: { y: number; label: string; color?: string }[];
}

export function TrendChart({
  data,
  xKey,
  series,
  height = 260,
  variant = "line",
  yUnit = "",
  referenceLines,
}: TrendChartProps) {
  const Chart = variant === "area" ? AreaChart : LineChart;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <Chart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis
          dataKey={xKey}
          tick={{ fill: "#94a3b8", fontSize: 11 }}
          tickLine={{ stroke: "#1e293b" }}
          axisLine={{ stroke: "#1e293b" }}
        />
        <YAxis
          tick={{ fill: "#94a3b8", fontSize: 11 }}
          tickLine={{ stroke: "#1e293b" }}
          axisLine={{ stroke: "#1e293b" }}
          unit={yUnit}
          width={48}
        />
        <Tooltip
          contentStyle={{
            background: "#111827",
            border: "1px solid #1e293b",
            borderRadius: 12,
            fontSize: 12,
          }}
          labelStyle={{ color: "#e2e8f0" }}
          isAnimationActive={false}
        />
        <Legend wrapperStyle={{ fontSize: 12, color: "#94a3b8" }} />
        {referenceLines?.map((r) => (
          <ReferenceLine
            key={r.label}
            y={r.y}
            stroke={r.color ?? "#f59e0b"}
            strokeDasharray="4 4"
            label={{ value: r.label, fill: r.color ?? "#f59e0b", fontSize: 10, position: "insideTopRight" }}
          />
        ))}
        {series.map((s) =>
          variant === "area" ? (
            <Area
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.label}
              stroke={s.color}
              fill={s.color}
              fillOpacity={0.25}
              strokeWidth={1.6}
              dot={false}
              strokeDasharray={s.strokeDasharray}
              isAnimationActive={false}
            />
          ) : (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.label}
              stroke={s.color}
              strokeWidth={1.8}
              dot={false}
              strokeDasharray={s.strokeDasharray}
              isAnimationActive={false}
            />
          )
        )}
      </Chart>
    </ResponsiveContainer>
  );
}
