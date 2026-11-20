#!/usr/bin/env bash
# Run the live benchmark for several models, one after another, and compare them (GPU machine).
#   scripts/live_compare.sh                                   # default list, 4bit
#   QUANT=bf16 scripts/live_compare.sh qwen3-vl-2b qwen3-vl-4b
#   BENCH_ARGS="--quick" scripts/live_compare.sh              # a few minutes per model
#   BENCH_ARGS="--photos live_photos" scripts/live_compare.sh # also your own photos
# Starts and stops the server itself (experience database off, so times are model times).
# Result: results/live/<time>_<model>-<quant>/report.html per model and results/live/compare_*/report.html
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
PORT=${PORT:-8010}
QUANT=${QUANT:-4bit}
MODELS=${*:-"qwen3-vl-2b qwen3-vl-4b qwen3-vl-8b"}
./serve.sh stop >/dev/null
RUNS=()
for m in $MODELS; do
  echo "=== $m ($QUANT) ==="
  if NO_MEMORY=1 PORT=$PORT ./serve.sh start "$m" "$QUANT"; then
    python live_bench.py --url "http://127.0.0.1:$PORT" --label "$m-$QUANT" $BENCH_ARGS \
      || echo "!!! benchmark failed for $m (see above); continuing"
    last=$(ls -d results/live/*_"$m-$QUANT" 2>/dev/null | tail -n 1)
    [ -n "$last" ] && RUNS+=("$last")
  else
    echo "!!! $m did not start; continuing with the next model"
  fi
  ./serve.sh stop
done
if [ "${#RUNS[@]}" -ge 2 ]; then python live_report.py "${RUNS[@]}"; fi
