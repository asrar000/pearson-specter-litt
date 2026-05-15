#!/usr/bin/env bash
set -euo pipefail

CMD=${1:-"server"}

case "$CMD" in
  server)
    echo "▶  Starting Pearson Specter Litt AI server..."
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
    ;;
  test)
    echo "🧪  Running test suite..."
    python -m pytest tests/ -v --tb=short
    ;;
  evaluate)
    echo "📊  Running evaluation pipeline..."
    python evaluate.py
    ;;
  docker-up)
    echo "🐳  Starting Docker stack..."
    docker compose up --build
    ;;
  docker-down)
    docker compose down
    ;;
  *)
    echo "Usage: ./run.sh [server|test|evaluate|docker-up|docker-down]"
    exit 1
    ;;
esac
