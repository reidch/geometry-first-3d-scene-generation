#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_NAME="${NERFSTUDIO_ENV:-worldmesh-nerfstudio}"
REPO="${WORLDMESH_REPO:-external/worldmesh}"
if [[ ! -d "$REPO/.git" ]]; then
  git clone https://github.com/mschneider456/worldmesh.git "$REPO"
fi
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10 pip
fi
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install -e "$REPO/nerfstudio"
echo "Installed WorldMesh-compatible Nerfstudio in conda env: $ENV_NAME"
