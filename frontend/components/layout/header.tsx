"use client";

import { useEffect, useState } from "react";
import { AlertOctagon } from "lucide-react";
import { cn } from "@/lib/utils";
import { useTwinStore } from "@/lib/store";

export function Header() {
  const wsStatus = useTwinStore((s) => s.wsStatus);
  const twinState = useTwinStore((s) => s.twinState);
  const [now, setNow] = useState<Date | null>(() => new Date());

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const simulatedTs = twinState?.timestamp
    ? new Date(twinState.timestamp)
    : null;

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-slate-800 bg-[#0b1120]/80 px-4 backdrop-blur">
      <div className="flex items-center gap-3">
        <span className="text-sm font-semibold text-slate-200">Raffinerie MNW</span>
        <span className="hidden text-xs text-slate-500 sm:inline">
          Horodatage jumeau :{" "}
          <span className="font-mono tabular-nums text-slate-400">
            {simulatedTs ? simulatedTs.toLocaleString("fr-FR") : "—"}
          </span>
        </span>
      </div>
      <div className="flex items-center gap-3">
        <span className="hidden text-xs text-slate-600 lg:inline font-mono tabular-nums">
          {now ? now.toLocaleTimeString("fr-FR") : ""}
        </span>
        <ConnectionPill status={wsStatus} />
        <div className="flex items-center gap-1.5 rounded-full border border-slate-800 bg-slate-900 px-2.5 py-1 text-xs text-slate-300">
          <AlertOctagon className="h-3.5 w-3.5 text-amber-400" />
          <span className="tabular-nums">{twinState?.active_alerts_count ?? 0}</span>
        </div>
      </div>
    </header>
  );
}

function ConnectionPill({ status }: { status: "connecting" | "connected" | "disconnected" }) {
  const config = {
    connected: { label: "Temps réel connecté", dot: "bg-emerald-400", text: "text-emerald-300" },
    connecting: { label: "Connexion…", dot: "bg-amber-400", text: "text-amber-300" },
    disconnected: { label: "Déconnecté — reconnexion…", dot: "bg-red-400", text: "text-red-300" },
  }[status];

  return (
    <div
      className={cn(
        "flex items-center gap-1.5 rounded-full border border-slate-800 bg-slate-900 px-2.5 py-1 text-xs",
        config.text
      )}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", config.dot)} />
      {config.label}
    </div>
  );
}
