#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source tools/scene_context.sh
source tools/stage_cache.sh
pgw_resolve_scene_context "${1:-}"
SCENE="$OUT/00_validated/scene.normalized.json"
WORLD="$OUT/01_world_ir/world.pkl"
pgw_require_inputs "02" "Stage00/01" "$SCENE" "$WORLD" "$OUT/01_world_ir/object_registry.json"
STATE="$OUT/02_blender_scaffold"
KEY="$(pgw_stage_key --path "$SCENE" --path "$OUT/01_world_ir/object_registry.json" --path src/blender/scaffold_from_json.py --path scripts/02_build_blender_scaffold.py)"
if pgw_should_skip "$STATE" "$KEY" "$STATE/scaffold.blend"; then pgw_skip_message "02"; exit 0; fi
pgw_rebuild_message "02" "$STATE" "$KEY"
rm -rf "$STATE"
"${BLENDER_BIN:-blender}" --background --python scripts/02_build_blender_scaffold.py -- --out "$OUT" --scene "$SCENE"
pgw_cache_mark "$STATE" "$KEY"
