#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source tools/scene_context.sh
source tools/stage_cache.sh
pgw_resolve_scene_context "${1:-}"
pgw_require_inputs "05" "Stage02/04" "$OUT/02_blender_scaffold/scaffold.blend" "$OUT/04_object_assets/artifact_index.json"
STATE="$OUT/05_scene_assets"
KEY="$(pgw_stage_key --path "$OUT/02_blender_scaffold/scaffold.blend" --path "$OUT/04_object_assets/artifact_index.json" --path configs/asset_pipeline.json --path scripts/05_import_register_object_assets.py --path src/blender/prephysics_runtime/import_align_prepare_assets.py)"
if pgw_should_skip "$STATE" "$KEY" "$STATE/scene_assets.blend" "$STATE/texture_state_snapshot"; then pgw_skip_message "05"; exit 0; fi
pgw_rebuild_message "05" "$STATE" "$KEY"
rm -rf "$STATE" "$OUT/05_texture_state"
python scripts/05_import_register_object_assets.py --out "$OUT" --asset_config configs/asset_pipeline.json
pgw_cache_mark "$STATE" "$KEY"
