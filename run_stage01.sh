#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source tools/scene_context.sh
source tools/stage_cache.sh
pgw_resolve_scene_context "${1:-}"
INPUT="$OUT/00_validated/scene.normalized.json"
pgw_require_inputs "01" "Stage00" "$INPUT"
STATE="$OUT/01_world_ir"
KEY="$(pgw_stage_key --path "$INPUT" --path src/scene_ir --path src/assets/eligibility.py --path scripts/01_build_world_ir.py)"
if pgw_should_skip "$STATE" "$KEY" "$STATE/world.pkl" "$STATE/generation_plan.json"; then pgw_skip_message "01"; exit 0; fi
pgw_rebuild_message "01" "$STATE" "$KEY"
rm -rf "$STATE"
python scripts/01_build_world_ir.py --scene "$INPUT" --out "$OUT"
pgw_cache_mark "$STATE" "$KEY"
