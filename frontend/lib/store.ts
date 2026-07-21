import { create } from "zustand";
import type { Alert, TwinState } from "@/lib/types";

export type WsStatus = "connecting" | "connected" | "disconnected";

export interface YieldsStreamPoint {
  timestamp: string;
  naphtha: number;
  kerosene: number;
  gasoil: number;
  residue: number;
}

const YIELDS_HISTORY_MAX = 48;

interface TwinStore {
  twinState: TwinState | null;
  wsStatus: WsStatus;
  activeAlerts: Alert[];
  yieldsStreamHistory: YieldsStreamPoint[];
  setTwinState: (state: TwinState) => void;
  setWsStatus: (status: WsStatus) => void;
  setActiveAlerts: (alerts: Alert[]) => void;
}

export const useTwinStore = create<TwinStore>((set, get) => ({
  twinState: null,
  wsStatus: "connecting",
  activeAlerts: [],
  yieldsStreamHistory: [],
  setTwinState: (state) => {
    const history = get().yieldsStreamHistory;
    const point: YieldsStreamPoint = { timestamp: state.timestamp, ...state.yields_stream };
    const nextHistory = [...history, point].slice(-YIELDS_HISTORY_MAX);
    set({ twinState: state, yieldsStreamHistory: nextHistory });
  },
  setWsStatus: (status) => set({ wsStatus: status }),
  setActiveAlerts: (alerts) => set({ activeAlerts: alerts }),
}));
