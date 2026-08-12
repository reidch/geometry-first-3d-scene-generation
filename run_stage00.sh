#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source tools/scene_context.sh
source tools/stage_cache.sh
pgw_resolve_scene_context "${1:-}"
STATE="$OUT/00_validated"
KEY="$(pgw_stage_key --path "$SCENE_JSON" --path schemas --path src/core/validation.py --path scripts/00_validate_scene.py)"
if pgw_should_skip "$STATE" "$KEY" "$STATE/scene.normalized.json" "$STATE/validation_report.json"; then
  pgw_skip_message "00"; exit 0
fi
pgw_rebuild_message "00" "$STATE" "$KEY"
rm -rf "$STATE"
python scripts/00_validate_scene.py --scene "$SCENE_JSON" --out "$OUT"
pgw_cache_mark "$STATE" "$KEY"
