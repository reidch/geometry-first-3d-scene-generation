#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source tools/scene_context.sh
source tools/stage_cache.sh
pgw_resolve_scene_context "${1:-}"
pgw_require_inputs "07" "Stage06" "$OUT/06_surface_textures/scene_surface_textured.blend" "$OUT/06_surface_textures/surface_material_binding.json"
STATE="$OUT/07_refinement_cameras"
KEY="$(pgw_stage_key --path "$SCENE_JSON" --path configs/cameras.json --path configs/refinement_pipeline.json --path "$OUT/06_surface_textures/scene_surface_textured.blend" --path src/cameras/room_pair_sampling.py --path src/cameras/reconstruction_view_metrics.py --path src/cameras/reconstruction_sampler.py --path src/cameras/worldmesh_coverage_sampling.py --path src/blender/prephysics_runtime/render_refinement_candidates_batch.py --path src/blender/prephysics_runtime/render_refinement_shared_buffers_batch.py)"
if pgw_should_skip "$STATE" "$KEY" "$STATE/reconstruction_dataset_manifest.json" "$STATE/shared_buffers/surface_texture_binding.json"; then pgw_skip_message "07"; exit 0; fi
pgw_rebuild_message "07" "$STATE" "$KEY"
rm -rf "$STATE"
python scripts/07_sample_refinement_cameras.py --out "$OUT" --refinement_config configs/refinement_pipeline.json --camera_config configs/cameras.json --scene_json "$SCENE_JSON"
pgw_cache_mark "$STATE" "$KEY"
