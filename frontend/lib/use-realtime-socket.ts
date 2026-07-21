"use client";

import { useEffect, useRef } from "react";
import { wsRealtimeUrl } from "@/lib/api";
import { useTwinStore } from "@/lib/store";
import type { TwinState } from "@/lib/types";

const RECONNECT_BASE_DELAY_MS = 1000;
const RECONNECT_MAX_DELAY_MS = 15000;

/**
 * Ouvre et maintient la connexion websocket /ws/realtime, avec reconnexion
 * automatique (backoff exponentiel) et mise a jour du store zustand global.
 */
export function useRealtimeSocket() {
  const setTwinState = useTwinStore((s) => s.setTwinState);
  const setWsStatus = useTwinStore((s) => s.setWsStatus);
  const wsStatus = useTwinStore((s) => s.wsStatus);

  const attemptRef = useRef(0);
  const socketRef = useRef<WebSocket | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const unmountedRef = useRef(false);

  useEffect(() => {
    unmountedRef.current = false;

    function connect() {
      if (unmountedRef.current) return;
      setWsStatus(attemptRef.current === 0 ? "connecting" : "connecting");

      let socket: WebSocket;
      try {
        socket = new WebSocket(wsRealtimeUrl());
      } catch {
        scheduleReconnect();
        return;
      }
      socketRef.current = socket;

      socket.onopen = () => {
        attemptRef.current = 0;
        setWsStatus("connected");
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as TwinState;
          setTwinState(data);
        } catch {
          // message non conforme ignore
        }
      };

      socket.onerror = () => {
        socket.close();
      };

      socket.onclose = () => {
        if (unmountedRef.current) return;
        setWsStatus("disconnected");
        scheduleReconnect();
      };
    }

    function scheduleReconnect() {
      attemptRef.current += 1;
      const delay = Math.min(
        RECONNECT_BASE_DELAY_MS * 2 ** attemptRef.current,
        RECONNECT_MAX_DELAY_MS
      );
      timeoutRef.current = setTimeout(connect, delay);
    }

    connect();

    return () => {
      unmountedRef.current = true;
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      socketRef.current?.close();
    };
  }, [setTwinState, setWsStatus]);

  return { wsStatus };
}
