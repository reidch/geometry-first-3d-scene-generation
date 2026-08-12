#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source tools/scene_context.sh
source tools/stage_cache.sh
pgw_resolve_scene_context "${1:-}"
pgw_require_inputs "06" "Stage05" "$OUT/05_scene_assets/scene_assets.blend" "$OUT/05_scene_assets/texture_state_snapshot"
STATE="$OUT/06_surface_textures"
PARAMETERS_CONFIG="configs/parameters.json"
# Model-loading compatibility code is deliberately excluded from this expensive cache key.
KEY="$(pgw_stage_key --path "$SCENE_JSON" --path configs/room_surface_pipeline.json --path "$PARAMETERS_CONFIG" --path "$OUT/05_scene_assets/scene_assets.blend" --path scripts/06a_generate_surface_images.py --path scripts/06b_commit_surface_textures.py --path src/room_surfaces --path src/appearance/backends/flux1_depth_control_inpaint_backend.py)"
if pgw_should_skip "$STATE" "$KEY" "$STATE/scene_surface_textured.blend" "$STATE/texture_state_snapshot"; then pgw_skip_message "06"; exit 0; fi
pgw_rebuild_message "06" "$STATE" "$KEY"
./run_stage06a.sh "$OUT"
./run_stage06b.sh "$OUT"
python tools/sync_tree.py "$OUT/06_surface_textures/texture_state_snapshot" "$OUT/05_texture_state"
pgw_cache_mark "$STATE" "$KEY"
