#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source tools/scene_context.sh
source tools/stage_cache.sh
pgw_resolve_scene_context "${1:-}"
pgw_require_inputs "09E-A" "Stage07" "$OUT/07_refinement_cameras/cameras.accepted.json" "$OUT/07_refinement_cameras/active_owner_ids.json"
pgw_require_inputs "09E-A" "Stage08B" "$OUT/08_viewwise_refinement/stage09_training_manifest.json"
pgw_require_inputs "09E-A" "Stage09" "$OUT/09_gaussian_splat/nerfstudio_config_path.txt"
STATE="$OUT/09e_evaluation/A_candidates"
KEY="$(pgw_stage_key --path configs/evaluation_pipeline.json --path configs/cameras.json --path configs/refinement_pipeline.json --path "$SCENE_JSON" --path "$OUT/07_refinement_cameras/cameras.accepted.json" --path "$OUT/08_viewwise_refinement/stage09_training_manifest.json" --path src/evaluation/evaluation_sampler.py --path scripts/09e_a_collect_evaluation_views.py)"

# Compatibility path for evaluation candidates created before the shared cache
# artifact index was introduced. Adopt complete immutable geometry outputs in
# place without modifying the user-editable selection or rerendering candidates.
if [[ ! -f "$STATE/artifact_index.json" \
      && -s "$STATE/candidate_manifest.json" \
      && -d "$STATE/shared_buffers" \
      && -s "$OUT/09e_evaluation/selection.csv" \
      && -s "$OUT/09e_evaluation/candidate_contact_sheet_rgb.png" \
      && -s "$OUT/09e_evaluation/candidate_contact_sheet_depth.png" \
      && -s "$OUT/09e_evaluation/A_stage_report.json" ]]; then
  echo "[09E-A] Adopting complete V132 evaluation candidates; no geometry rerender is required."
  python - "$OUT" <<'PYEOF'
from pathlib import Path
import sys
from src.pipeline.artifact_index import ArtifactIndex
from src.pipeline.resume import mark_done
out = Path(sys.argv[1])
stage = out / "09e_evaluation"
state = stage / "A_candidates"
artifact = ArtifactIndex(scene_id=out.name, step="09e_a_collect_evaluation_views")
artifact.add("candidate_manifest", state / "candidate_manifest.json")
artifact.add("candidate_shared_buffers", state / "shared_buffers")
artifact.add("rgb_contact_sheet", stage / "candidate_contact_sheet_rgb.png")
artifact.add("depth_contact_sheet", stage / "candidate_contact_sheet_depth.png")
artifact.add("stage_report", stage / "A_stage_report.json")
artifact.save(state / "artifact_index.json")
mark_done(state)
PYEOF
  pgw_cache_mark "$STATE" "$KEY"
  echo "[09E-A] V132 outputs adopted. Your existing selection.csv was preserved."
  exit 0
fi

if pgw_should_skip "$STATE" "$KEY" "$OUT/09e_evaluation/selection.csv" "$OUT/09e_evaluation/candidate_contact_sheet_rgb.png"; then pgw_skip_message "09E-A"; exit 0; fi
echo "[09E-A] This independent runner does not run or modify Stage00-09. Gaussian output is NOT rendered during curation collection."
rm -rf "$OUT/09e_evaluation"
"${PYTHON:-python}" scripts/09e_a_collect_evaluation_views.py --out "$OUT" --scene_json "$SCENE_JSON"
pgw_cache_mark "$STATE" "$KEY"
echo "[09E-A] Manual step: inspect mesh-only contact sheets and edit keep/reject_reason in $OUT/09e_evaluation/selection.csv, then run bash run_stage09e_b.sh"
