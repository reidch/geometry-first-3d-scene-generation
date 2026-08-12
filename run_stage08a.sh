#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source tools/scene_context.sh
source tools/stage_cache.sh
pgw_resolve_scene_context "${1:-}"
pgw_require_inputs "08A" "Stage07" "$OUT/07_refinement_cameras/reconstruction_dataset_manifest.json" "$OUT/07_refinement_cameras/shared_buffers"
STATE="$OUT/08_viewwise_refinement"
SUBSTATE="$STATE/08a_forward"
PARAMETERS_CONFIG="configs/parameters.json"
KEY="$(pgw_stage_key --path "$SCENE_JSON" --path configs/refinement_pipeline.json --path configs/prompts.json --path "$PARAMETERS_CONFIG" --path "$OUT/07_refinement_cameras/reconstruction_dataset_manifest.json" --path run_stage08a.sh --path scripts/08_generate_final_room_views.py --path src/appearance/whole_view_generation.py --path src/appearance/stage08_console.py --path src/appearance/stage08_runtime.py --path src/appearance/multiview_consistency.py --path src/appearance/monocular_depth_validation.py --path src/appearance/backend_factory.py --path src/appearance/model_cache.py --path src/appearance/depth_control_image.py --path src/appearance/backends/flux1_depth_control_inpaint_backend.py --path src/appearance/backends/flux2_klein_multiref_backend.py)"
if pgw_should_skip "$SUBSTATE" "$KEY" "$STATE/forward_pass_summary.json" "$STATE/final_views" "$STATE/weighted_frontier_generation_order.json"; then
  pgw_skip_message "08A"
  exit 0
fi
RESUME_KEY_FILE="$SUBSTATE/.incomplete_stage_key"
if [[ -d "$STATE" && -f "$RESUME_KEY_FILE" && "$(cat "$RESUME_KEY_FILE")" == "$KEY" ]] \
   && ! _pgw_truthy "${FORCE:-0}" && ! _pgw_truthy "${FORCE_ALL:-0}"; then
  echo "[08A] Resuming incomplete forward generation with the same scoped fingerprint."
  rm -f "$STATE/.failed" "$SUBSTATE/.failed"
else
  pgw_rebuild_message "08A" "$SUBSTATE" "$KEY"
  rm -rf "$STATE"
  mkdir -p "$SUBSTATE"
  printf '%s\n' "$KEY" > "$RESUME_KEY_FILE"
fi
python scripts/08_generate_final_room_views.py \
  --phase forward \
  --out "$OUT" \
  --refinement_config configs/refinement_pipeline.json \
  --scene_json "$SCENE_JSON" \
  --prompts_json configs/prompts.json \
  --parameters_json "$PARAMETERS_CONFIG"
pgw_cache_mark "$SUBSTATE" "$KEY"
rm -f "$RESUME_KEY_FILE"
