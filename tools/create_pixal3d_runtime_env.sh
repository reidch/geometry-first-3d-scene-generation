#!/usr/bin/env bash
set -euo pipefail
ENV_NAME=${1:-pixal3d_runtime}
REPO=${PIXAL3D_REPO:-external/Pixal3D}
git clone https://github.com/TencentARC/Pixal3D.git "$REPO" 2>/dev/null || true
echo "Follow the official TRELLIS.2 installation, then run: pip install -r $REPO/requirements.txt"
echo "Set PIXAL3D_PYTHON to this environment's python executable."
