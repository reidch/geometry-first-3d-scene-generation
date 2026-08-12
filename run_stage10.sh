#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source tools/scene_context.sh
pgw_resolve_scene_context "${1:-}"
CONFIG_PTR="$OUT/09_gaussian_splat/nerfstudio_config_path.txt"
if [[ ! -s "$CONFIG_PTR" ]]; then
  echo "[10] Missing Stage09 Nerfstudio config pointer: $CONFIG_PTR" >&2
  exit 1
fi
echo "[10] This independent runner will not run Stage09 or earlier stages."
echo "[10] Launching Nerfstudio viewer from the Stage09 trained config."
"${PYTHON:-python}" scripts/10_view_gaussian_scene.py --out "$OUT" --conda-env "${NERFSTUDIO_ENV:-worldmesh-nerfstudio}"
