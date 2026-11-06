#!/usr/bin/env bash
# Create .venv and install everything.
#   ./setup.sh            GPU machine (Linux + NVIDIA): torch with CUDA + VLM libraries + planners
#   ./setup.sh --cpu      Mac / no GPU: only planners and validation (benchmark with --models mock)
# Change CUDA_INDEX if your driver needs another CUDA build (check `nvidia-smi`, top right).
set -euo pipefail
cd "$(dirname "$0")"
CUDA_INDEX=${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}
PY=${PYTHON:-python3}

$PY -m venv .venv
.venv/bin/pip install -U pip

if [[ "${1:-}" == "--cpu" ]]; then
  .venv/bin/pip install pyyaml pillow pytest "unified-planning>=1.3" up-fast-downward up-pyperplan
else
  .venv/bin/pip install torch torchvision --index-url "$CUDA_INDEX"
  .venv/bin/pip install -r requirements.txt
fi

.venv/bin/python check_env.py
echo
echo "Next: source .venv/bin/activate && python -m pytest tests -q"
