#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"

# Default: requirements-model.txt (slim model image). Override with CDSW_REQUIREMENTS_PROFILE=full
# (e.g. create_training_job.py) for the full dev/training stack.
if [[ "${CDSW_REQUIREMENTS_PROFILE:-}" == "full" ]]; then
  REQ="$ROOT/requirements.txt"
elif [[ -f "$ROOT/requirements-model.txt" ]]; then
  REQ="$ROOT/requirements-model.txt"
elif [[ -f "$ROOT/requirements.txt" ]]; then
  REQ="$ROOT/requirements.txt"
else
  echo "cdsw-build.sh: missing $ROOT/requirements.txt" >&2
  exit 1
fi

echo "cdsw-build.sh: using $($PY --version 2>&1)"
echo "cdsw-build.sh: file=$(basename "$REQ")"

$PY -m pip install --upgrade pip --quiet

# CPU-only PyTorch before sentence-transformers (if listed) so pip does not pull CUDA wheels.
if grep -qE '^[[:space:]]*[^#[:space:]].*sentence-transformers' "$REQ"; then
  $PY -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
fi

$PY -m pip install --no-cache-dir -r "$REQ"

echo "cdsw-build.sh: finished installing from $(basename "$REQ")"
