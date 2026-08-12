#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source tools/scene_context.sh
source tools/stage_cache.sh
pgw_resolve_scene_context "${1:-}"
pgw_require_inputs "06b" "Stage05/06a" "$OUT/05_scene_assets/texture_state_snapshot" "$OUT/06_surface_textures/stage06a_report.json"
STATE="$OUT/06_surface_textures/06b_state"
KEY="$(pgw_stage_key --path "$SCENE_JSON" --path configs/room_surface_pipeline.json --path "$OUT/06_surface_textures/stage06a_report.json" --path scripts/06b_commit_surface_textures.py --path src/room_surfaces/surface_commit.py --path src/room_surfaces/surface_publish.py)"
if pgw_should_skip "$STATE" "$KEY" "$OUT/06_surface_textures/scene_surface_textured.blend"; then pgw_skip_message "06b"; exit 0; fi
if [[ -f "$OUT/06_surface_textures/artifact_index.json" && -f "$OUT/06_surface_textures/scene_surface_textured.blend" && ! -f "$STATE/.done" ]]; then
  echo "[06b] Recovering completed V63 outputs into the private cache state."
  mkdir -p "$STATE"
  cp "$OUT/06_surface_textures/artifact_index.json" "$STATE/artifact_index.json"
  printf 'done\n' > "$STATE/.done"
  pgw_cache_mark "$STATE" "$KEY"
  exit 0
fi
pgw_rebuild_message "06b" "$STATE" "$KEY"
rm -rf "$STATE"
# generated_locked.png is consumed here; Stage06b never invokes 06a generation.
python scripts/06b_commit_surface_textures.py --out "$OUT" --surface_config configs/room_surface_pipeline.json --scene_json "$SCENE_JSON"
cp "$OUT/06_surface_textures/artifact_index.json" "$STATE/artifact_index.json"
pgw_cache_mark "$STATE" "$KEY"
