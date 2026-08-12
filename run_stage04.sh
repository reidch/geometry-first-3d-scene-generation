#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source tools/scene_context.sh
source tools/stage_cache.sh
pgw_resolve_scene_context "${1:-}"
pgw_require_inputs "04" "Stage03" "$OUT/03_object_representative_images/artifact_index.json"
STATE="$OUT/04_object_assets"
KEY="$(pgw_stage_key --path "$OUT/03_object_representative_images/artifact_index.json" --path configs/asset_pipeline.json --path scripts/04_generate_object_3d_assets.py --path src/assets)"
if pgw_should_skip "$STATE" "$KEY" "$STATE/artifact_index.json"; then pgw_skip_message "04"; exit 0; fi
pgw_rebuild_message "04" "$STATE" "$KEY"
rm -rf "$STATE"
python scripts/04_generate_object_3d_assets.py --out "$OUT" --asset_config configs/asset_pipeline.json
pgw_cache_mark "$STATE" "$KEY"
