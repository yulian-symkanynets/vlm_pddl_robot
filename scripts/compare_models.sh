#!/usr/bin/env bash
# Compare models on the photographed scenes: speed against correct plans (GPU machine).
#   scripts/compare_models.sh                                  # default list below, bf16
#   QUANT=4bit scripts/compare_models.sh qwen3-vl-2b qwen3-vl-8b
#   LEVELS="low medium" scripts/compare_models.sh qwen3.5-2b lfm2.5-vl-1.6b
# Stop the server first (./serve.sh stop): the sweep loads each model itself.
# Results: results/resolution_sweep_<model>.json; chart: python run_report.py or /dashboard.
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
MODELS=${*:-"qwen3-vl-2b qwen3.5-2b lfm2.5-vl-1.6b qwen3-vl-4b qwen3.5-4b qwen3-vl-8b"}
for m in $MODELS; do
  echo "=== $m (${QUANT:-bf16}) ==="
  python scripts/resolution_sweep.py --model "$m" --quant "${QUANT:-bf16}" --levels ${LEVELS:-low medium high} \
    || echo "!!! $m failed (see above); continuing with the next model"
done
python run_report.py
