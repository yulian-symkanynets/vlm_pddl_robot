#!/usr/bin/env bash
# Run the planning server in the background.
#   ./serve.sh start [model] [quant]   e.g. ./serve.sh start qwen3-vl-8b 4bit   (default: $MODEL or qwen3-vl-8b)
#   ./serve.sh stop | restart | status | logs
# Env: PORT (8000), HOST (0.0.0.0), MAX_VISUAL_TOKENS (1024)
cd "$(dirname "$0")"
mkdir -p logs
PID=logs/server.pid
LOG=logs/server.log
PORT=${PORT:-8000}
[ -f .venv/bin/activate ] && source .venv/bin/activate

running() { [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; }

case "${1:-help}" in
  start)
    if running; then echo "already running (pid $(cat $PID))"; exit 0; fi
    MODEL=${2:-${MODEL:-qwen3-vl-8b}} QUANT=${3:-${QUANT:-4bit}} PORT=$PORT \
      nohup python server.py >> "$LOG" 2>&1 &
    echo $! > "$PID"
    echo "started pid $! (model ${2:-${MODEL:-qwen3-vl-8b}}), log: $LOG"
    echo -n "waiting for the model to load "
    for _ in $(seq 1 300); do
      if curl -s "http://localhost:$PORT/health" >/dev/null; then echo; echo "ready: http://localhost:$PORT/docs"; exit 0; fi
      if ! running; then echo; echo "server exited, last log lines:"; tail -n 30 "$LOG"; exit 1; fi
      echo -n "."; sleep 2
    done
    echo; echo "still loading, check: ./serve.sh logs" ;;
  stop)
    if running; then kill "$(cat $PID)" && rm -f "$PID" && echo stopped; else echo "not running"; rm -f "$PID"; fi ;;
  restart)
    "$0" stop; sleep 2; "$0" start "${@:2}" ;;
  status)
    if running; then echo "running (pid $(cat $PID))"; curl -s "http://localhost:$PORT/health"; echo
    else echo "not running"; fi ;;
  logs)
    tail -n 50 -f "$LOG" ;;
  *)
    sed -n '2,5p' "$0" ;;
esac
