#!/usr/bin/env bash
# Lance le jumeau numérique (backend + frontend).
#   ./run.sh          -> mode Docker (build + up, comme en prod)
#   ./run.sh dev       -> mode dev local (uvicorn --reload + next dev), sans Docker
set -euo pipefail
cd "$(dirname "$0")"

mode="${1:-docker}"

if [ "$mode" = "docker" ]; then
  echo "==> Build + démarrage des conteneurs (backend:8000, frontend:3000/3001 selon override)..."
  docker compose up -d --build
  echo
  echo "==> En cours de démarrage. Suivre les logs : docker compose logs -f"
  echo "==> Backend  : http://localhost:8000/docs"
  echo "==> Frontend : http://localhost:3000 (ou 3001 si docker-compose.override.yml présent)"

elif [ "$mode" = "dev" ]; then
  echo "==> Mode dev local (sans Docker)"
  mkdir -p /tmp/mnw-logs

  echo "--> Backend (uvicorn --reload) sur :8000"
  ( source .venv/bin/activate && \
    MODELS_ARTIFACTS_DIR="$(pwd)/backend/models_artifacts" \
    DATA_RAW_DIR="$(pwd)/data/raw" \
    DATA_PROCESSED_DIR="$(pwd)/data/processed" \
    CORS_ORIGINS="http://localhost:3000" \
    uvicorn backend.app.main:app --reload --port 8000 \
    > /tmp/mnw-logs/backend.log 2>&1 & )
  echo $! > /tmp/mnw-logs/backend.pid

  echo "--> Frontend (next dev) sur :3000"
  ( cd frontend && NEXT_PUBLIC_API_URL="http://localhost:8000" npm run dev \
    > /tmp/mnw-logs/frontend.log 2>&1 & )

  sleep 1
  echo
  echo "==> Logs : tail -f /tmp/mnw-logs/backend.log /tmp/mnw-logs/frontend.log"
  echo "==> Backend  : http://localhost:8000/docs"
  echo "==> Frontend : http://localhost:3000"
  echo "==> Arrêt : ./stop.sh"
else
  echo "Usage: ./run.sh [docker|dev]"
  exit 1
fi
