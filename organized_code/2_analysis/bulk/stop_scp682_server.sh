#!/usr/bin/env bash
set -euo pipefail

ROOT="${RNA2PHOSPHO_ROOT:-/data/lsy/Infinite_Stream}"
PORT="${RNA2PHOSPHO_PORT:-8866}"
LOG_DIR="$ROOT/02_results/public_bulk_phosphoproteome_atlas"
PID_FILE="$LOG_DIR/scp682_web_${PORT}.pid"

pids=""
if [[ -f "$PID_FILE" ]]; then
  pids="$(cat "$PID_FILE" || true)"
fi
if command -v lsof >/dev/null 2>&1; then
  pids="$pids $(lsof -ti tcp:"$PORT" || true)"
fi
pids="$(echo "$pids" | tr ' ' '\n' | awk 'NF' | sort -u | tr '\n' ' ')"

if [[ -z "$pids" ]]; then
  echo "No SCP682 process found on port $PORT"
  exit 0
fi

echo "Stopping SCP682 process(es): $pids"
kill $pids || true
rm -f "$PID_FILE"
