"use client";

import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { useRealtimeSocket } from "@/lib/use-realtime-socket";

export function AppShell({ children }: { children: React.ReactNode }) {
  useRealtimeSocket();

  return (
    <div className="flex h-full min-h-screen w-full">
      <Sidebar />
      <div className="flex min-h-screen flex-1 flex-col">
        <Header />
        <main className="flex-1 overflow-x-hidden bg-[#0b1120] p-4 md:p-6">{children}</main>
      </div>
    </div>
  );
}
