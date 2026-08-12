#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source tools/scene_context.sh
source tools/stage_cache.sh
pgw_resolve_scene_context "${1:-}"
pgw_require_inputs "03" "Stage02" "$OUT/02_blender_scaffold/scaffold.blend" "$OUT/01_world_ir/generation_plan.json"
STATE="$OUT/03_object_representative_images"
KEY="$(pgw_stage_key --path "$OUT/02_blender_scaffold/scaffold.blend" --path "$OUT/01_world_ir/generation_plan.json" --path configs/asset_pipeline.json --path scripts/03_generate_object_representative_images.py --path src/appearance)"
if pgw_should_skip "$STATE" "$KEY" "$STATE/artifact_index.json"; then pgw_skip_message "03"; exit 0; fi
pgw_rebuild_message "03" "$STATE" "$KEY"
rm -rf "$STATE"
python scripts/03_generate_object_representative_images.py --out "$OUT" --asset_config configs/asset_pipeline.json
pgw_cache_mark "$STATE" "$KEY"
