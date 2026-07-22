#!/usr/bin/env bash
# Arrête le jumeau numérique.
#   ./stop.sh          -> arrête les conteneurs Docker
#   ./stop.sh dev       -> arrête les process dev locaux lancés par ./run.sh dev
set -euo pipefail
cd "$(dirname "$0")"

mode="${1:-docker}"

if [ "$mode" = "docker" ]; then
  docker compose down
elif [ "$mode" = "dev" ]; then
  for name in backend; do
    pidfile="/tmp/mnw-logs/${name}.pid"
    if [ -f "$pidfile" ]; then
      pid="$(cat "$pidfile")"
      kill "$pid" 2>/dev/null || true
      rm -f "$pidfile"
    fi
  done
  pkill -f "next dev" 2>/dev/null || true
  pkill -f "uvicorn backend.app.main:app" 2>/dev/null || true
  echo "==> Process dev arrêtés."
else
  echo "Usage: ./stop.sh [docker|dev]"
  exit 1
fi
