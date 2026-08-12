#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source tools/scene_context.sh
source tools/stage_cache.sh
pgw_resolve_scene_context "${1:-}"
pgw_require_inputs "09" "Stage08" "$OUT/08_viewwise_refinement/stage09_training_manifest.json"
STATE="$OUT/09_gaussian_splat"
KEY="$(pgw_stage_key --path configs/gaussian_pipeline.json --path "$OUT/08_viewwise_refinement/stage09_training_manifest.json" --path src/gaussian/nerfstudio_stage09.py --path scripts/09_train_gaussian_scene.py)"
if pgw_should_skip "$STATE" "$KEY" "$STATE/nerfstudio_config_path.txt"; then pgw_skip_message "09"; exit 0; fi
echo "[09] This independent runner will not run Stage00-08."
pgw_rebuild_message "09" "$STATE" "$KEY"
rm -rf "$STATE"
"${PYTHON:-python}" scripts/09_train_gaussian_scene.py --out "$OUT" --config configs/gaussian_pipeline.json
pgw_cache_mark "$STATE" "$KEY"
