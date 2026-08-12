#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source tools/scene_context.sh
source tools/stage_cache.sh
pgw_resolve_scene_context "${1:-}"
pgw_require_inputs "06a" "Stage05" "$OUT/05_scene_assets/scene_assets.blend" "$OUT/05_scene_assets/texture_state_snapshot"
STATE="$OUT/06_surface_textures/06a_state"
PARAMETERS_CONFIG="configs/parameters.json"
KEY="$(pgw_stage_key --path "$SCENE_JSON" --path configs/room_surface_pipeline.json --path "$PARAMETERS_CONFIG" --path "$OUT/05_scene_assets/scene_assets.blend" --path scripts/06a_generate_surface_images.py --path src/room_surfaces --path src/appearance/backends/flux1_depth_control_inpaint_backend.py)"
if pgw_should_skip "$STATE" "$KEY" "$STATE/artifact_index.json"; then pgw_skip_message "06a"; exit 0; fi
if [[ -f "$OUT/06_surface_textures/artifact_index_06a.json" && -f "$OUT/06_surface_textures/stage06a_report.json" && ! -f "$STATE/.done" ]]; then
  echo "[06a] Recovering completed V63 outputs into the private cache state."
  mkdir -p "$STATE"
  cp "$OUT/06_surface_textures/artifact_index_06a.json" "$STATE/artifact_index.json"
  printf 'done\n' > "$STATE/.done"
  pgw_cache_mark "$STATE" "$KEY"
  exit 0
fi
pgw_rebuild_message "06a" "$STATE" "$KEY"
rm -rf "$STATE"
python scripts/06a_generate_surface_images.py --out "$OUT" --surface_config configs/room_surface_pipeline.json --scene_json "$SCENE_JSON"
pgw_cache_mark "$STATE" "$KEY"
