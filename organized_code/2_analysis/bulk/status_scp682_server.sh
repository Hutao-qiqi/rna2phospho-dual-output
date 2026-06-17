#!/usr/bin/env bash
set -euo pipefail

ROOT="${RNA2PHOSPHO_ROOT:-/data/lsy/Infinite_Stream}"
PORT="${RNA2PHOSPHO_PORT:-8866}"
LOG_DIR="$ROOT/02_results/public_bulk_phosphoproteome_atlas"
PID_FILE="$LOG_DIR/scp682_web_${PORT}.pid"
LOG_FILE="$LOG_DIR/scp682_web_${PORT}.nohup.log"

echo "SCP682 status"
echo "port: $PORT"
if [[ -f "$PID_FILE" ]]; then
  echo "pid_file: $(cat "$PID_FILE")"
else
  echo "pid_file: missing"
fi
if command -v ss >/dev/null 2>&1; then
  ss -ltnp | grep ":$PORT" || true
fi
echo "health:"
curl -fsS "http://127.0.0.1:$PORT/api/health" || true
echo
echo "log: $LOG_FILE"
tail -20 "$LOG_FILE" 2>/dev/null || true
