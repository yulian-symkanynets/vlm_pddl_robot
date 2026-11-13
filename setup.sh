#!/usr/bin/env bash
# Create .venv and install everything.
#   ./setup.sh            GPU machine (Linux + NVIDIA): torch with CUDA + VLM libraries + planners
#   ./setup.sh --cpu      Mac / no GPU: only planners and validation (benchmark with --models mock)
# The CUDA build of torch is chosen from the GPU; override with CUDA_INDEX=... ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"
# PyTorch build: RTX 50xx (Blackwell, compute capability 10+/12+) needs CUDA 12.8 builds,
# older GPUs (RTX 30xx/40xx, A4500, ...) work with cu124 and older drivers.
if [[ -z "${CUDA_INDEX:-}" ]]; then
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | cut -d. -f1 || true)
  if [[ -n "$CAP" && "$CAP" -ge 10 ]]; then CUDA_INDEX=https://download.pytorch.org/whl/cu128
  else CUDA_INDEX=https://download.pytorch.org/whl/cu124; fi
fi
echo "PyTorch from: $CUDA_INDEX"
PY=${PYTHON:-python3}

if ! $PY -c 'import ensurepip' 2>/dev/null; then
  echo "python venv support missing. Run:  sudo apt update && sudo apt install -y python3-venv python3-pip"; exit 1
fi
[[ -x .venv/bin/pip ]] || { rm -rf .venv; $PY -m venv .venv; }
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
