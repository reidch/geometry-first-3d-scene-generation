#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source tools/scene_context.sh
source tools/stage_cache.sh
pgw_resolve_scene_context "${1:-}"
pgw_require_inputs "09E-B" "Stage09E-A" "$OUT/09e_evaluation/A_candidates/candidate_manifest.json" "$OUT/09e_evaluation/selection.csv"
pgw_require_inputs "09E-B" "Stage09" "$OUT/09_gaussian_splat/nerfstudio_config_path.txt"
STATE="$OUT/09e_evaluation/B_metrics"
KEY="$(pgw_stage_key --path configs/evaluation_pipeline.json --path configs/gaussian_pipeline.json --path "$OUT/09e_evaluation/A_candidates/candidate_manifest.json" --path "$OUT/09e_evaluation/selection.csv" --path "$OUT/09_gaussian_splat/nerfstudio_config_path.txt" --path src/evaluation/evaluation_sampler.py --path src/evaluation/evaluation_metrics.py --path scripts/09e_render_nerfstudio_evaluation.py --path scripts/09e_guard_nerfstudio_runtime.py --path scripts/09e_ensure_quality_runtime.py --path scripts/09e_compute_image_quality.py --path scripts/09e_b_run_evaluation.py --path requirements-stage09e-quality.txt)"
if pgw_should_skip "$STATE" "$KEY" "$OUT/09e_evaluation/B_metrics/summary.json" "$OUT/09e_evaluation/B_frozen_evaluation_manifest.sha256"; then pgw_skip_message "09E-B"; exit 0; fi
echo "[09E-B] This independent runner will NOT rerun Stage09E-A or Stage00-09. It freezes the current selection.csv and evaluates exactly that set."
rm -rf "$OUT/09e_evaluation/B_renders" "$OUT/09e_evaluation/B_metrics" "$OUT/09e_evaluation/B_frozen_evaluation_manifest.json" "$OUT/09e_evaluation/B_frozen_evaluation_manifest.sha256" "$OUT/09e_evaluation/B_stage_report.json"
"${PYTHON:-python}" scripts/09e_b_run_evaluation.py --out "$OUT"
pgw_cache_mark "$STATE" "$KEY"
