#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_ENV="${1:-pgw_flux}"
PIXAL3D_ENV="${2:-pixal3d_runtime}"
bash "$ROOT_DIR/tools/create_flux_env.sh" "$MAIN_ENV"
bash "$ROOT_DIR/tools/create_pixal3d_runtime_env.sh" "$PIXAL3D_ENV"
echo "Main shell: conda activate $MAIN_ENV"
echo "Pixal3D subprocess: export PIXAL3D_PYTHON=\"$(conda info --base)/envs/$PIXAL3D_ENV/bin/python\""
