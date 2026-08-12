#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source tools/scene_context.sh
pgw_resolve_scene_context "${1:-}"
echo "[08] Stage08 is split into 08A forward generation and 08B repair."
bash run_stage08a.sh "$OUT"
bash run_stage08b.sh "$OUT"
