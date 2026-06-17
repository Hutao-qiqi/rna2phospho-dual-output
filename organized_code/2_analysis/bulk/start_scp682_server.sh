#!/usr/bin/env bash
set -euo pipefail

ROOT="${RNA2PHOSPHO_ROOT:-/data/lsy/Infinite_Stream}"
APP_DIR="$ROOT/03_code/rna2phospho_web"
PYTHON="${RNA2PHOSPHO_PYTHON:-/home/USER/.local/share/mamba/envs/omicverse/bin/python}"
HOST="${RNA2PHOSPHO_HOST:-0.0.0.0}"
PORT="${RNA2PHOSPHO_PORT:-8866}"
LOG_DIR="$ROOT/02_results/public_bulk_phosphoproteome_atlas"
LOG_FILE="$LOG_DIR/scp682_web_${PORT}.nohup.log"
PID_FILE="$LOG_DIR/scp682_web_${PORT}.pid"

mkdir -p "$LOG_DIR"

if command -v lsof >/dev/null 2>&1; then
  existing="$(lsof -ti tcp:"$PORT" || true)"
  if [[ -n "$existing" ]]; then
    echo "SCP682 already has a process on port $PORT: $existing"
    echo "$existing" > "$PID_FILE"
    exit 0
  fi
fi

cd "$APP_DIR"
RNA2PHOSPHO_ROOT="$ROOT" RNA2PHOSPHO_PYTHON="$PYTHON" RNA2PHOSPHO_HOST="$HOST" RNA2PHOSPHO_PORT="$PORT" \
  nohup "$PYTHON" server_stdlib.py > "$LOG_FILE" 2>&1 &
pid="$!"
echo "$pid" > "$PID_FILE"
sleep 2
echo "SCP682 started: pid=$pid url=http://$HOST:$PORT log=$LOG_FILE"
curl -fsS "http://127.0.0.1:$PORT/api/health" || {
  echo
  echo "health check failed; see $LOG_FILE" >&2
  exit 1
}
echo
