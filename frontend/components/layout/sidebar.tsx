"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  AlertOctagon,
  BookOpenText,
  Factory,
  Flame,
  Gauge,
  LayoutDashboard,
  Zap,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/", label: "Vue d'ensemble", icon: LayoutDashboard },
  { href: "/jumeau", label: "Jumeau numérique", icon: Factory },
  { href: "/rendements", label: "Rendements", icon: Activity },
  { href: "/encrassement", label: "Encrassement", icon: Flame },
  { href: "/energie", label: "Énergie", icon: Zap },
  { href: "/alertes", label: "Alertes", icon: AlertOctagon },
  { href: "/documentation", label: "Documentation", icon: BookOpenText },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden w-60 shrink-0 flex-col border-r border-slate-800 bg-[#0b1120] px-3 py-4 md:flex">
      <div className="mb-6 flex items-center gap-2 px-2">
        <Gauge className="h-6 w-6 text-cyan-400" />
        <div className="flex flex-col leading-tight">
          <span className="text-sm font-semibold text-slate-100">MNW Raffinage</span>
          <span className="text-[11px] text-slate-500">Jumeau numérique DL</span>
        </div>
      </div>
      <nav className="flex flex-col gap-1">
        {NAV_ITEMS.map((item) => {
          const active = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-2.5 rounded-xl px-3 py-2 text-sm font-medium transition-colors",
                active
                  ? "bg-cyan-500/10 text-cyan-300"
                  : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-200"
              )}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>
      <div className="mt-auto rounded-xl border border-slate-800 bg-slate-900/60 px-3 py-2.5 text-[11px] text-slate-500">
        Raffinerie 200 000 bbl/j
        <br />
        CDU &amp; Vapocraqueur
      </div>
    </aside>
  );
}
