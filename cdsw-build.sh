#!/usr/bin/env bash
# Cloudera AI / CDSW: runs early when the project builds or syncs so Jobs and deployed
# models (model_api.predict) have packages like joblib, pandas, scikit-learn available.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Full stack (notebooks, Streamlit, optional extras): requirements.txt
# Slim model image (recommended for Model deployment to avoid huge CUDA/Jupyter layers):
#   Set environment variable on the *model build*: CDSW_REQUIREMENTS_PROFILE=model
PROFILE="${CDSW_REQUIREMENTS_PROFILE:-full}"
if [[ "$PROFILE" == "model" && -f "$ROOT/requirements-model.txt" ]]; then
  REQ="$ROOT/requirements-model.txt"
elif [[ -f "$ROOT/requirements.txt" ]]; then
  REQ="$ROOT/requirements.txt"
else
  echo "cdsw-build.sh: missing $ROOT/requirements.txt" >&2
  exit 1
fi

PY="${PYTHON:-python3}"
echo "cdsw-build.sh: using $($PY --version 2>&1)"
echo "cdsw-build.sh: profile=${PROFILE} file=$(basename "$REQ")"

$PY -m pip install --upgrade pip --quiet

if [[ "$PROFILE" == "model" ]]; then
  # Default PyPI ``torch`` on Linux pulls CUDA + NVIDIA wheels (~multi-GB) and often makes the
  # model image too large to push to s2i-registry (blob upload invalid / unknown error).
  $PY -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
fi

$PY -m pip install --no-cache-dir -r "$REQ"

echo "cdsw-build.sh: finished installing from $(basename "$REQ")"
