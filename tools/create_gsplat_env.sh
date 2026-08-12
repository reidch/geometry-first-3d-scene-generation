#!/usr/bin/env bash
set -euo pipefail
ENV_NAME="${GSPLAT_ENV_NAME:-pgw_gsplat}"
PYTHON_VERSION="${GSPLAT_PYTHON_VERSION:-3.11}"
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required" >&2
  exit 2
fi
conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
# PyTorch must match the user's CUDA/driver. Override this command when needed.
conda run -n "$ENV_NAME" python -m pip install --upgrade pip ninja
if [[ -n "${TORCH_INSTALL_COMMAND:-}" ]]; then
  conda run -n "$ENV_NAME" bash -lc "$TORCH_INSTALL_COMMAND"
else
  echo "[gsplat] Install CUDA-matched PyTorch in $ENV_NAME, then run:" >&2
  echo "  conda run -n $ENV_NAME python -m pip install -r requirements-gsplat.txt" >&2
  exit 3
fi
conda run -n "$ENV_NAME" python -m pip install -r requirements.txt -r requirements-gsplat.txt
