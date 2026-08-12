#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source tools/scene_context.sh
pgw_resolve_scene_context "${1:-}"
FROM="${FROM_STAGE:-00}"
TO="${TO_STAGE:-09}"
for number in $(seq $((10#$FROM)) $((10#$TO))); do
  stage=$(printf '%02d' "$number")
  script="run_stage${stage}.sh"
  bash "$script" "$OUT"
done
