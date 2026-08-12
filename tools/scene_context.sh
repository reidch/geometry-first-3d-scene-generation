#!/usr/bin/env bash
# Resolve the active scene from one fixed directory and derive its output path.
# A caller may still provide SCENE_JSON/OUT explicitly for CI or archived runs.

pgw_resolve_scene_context() {
  local explicit_out="${1:-}"
  local scene_dir="${SCENE_DIR:-data/scenes/current}"
  local scene_json="${SCENE_JSON:-}"

  if [[ -z "$scene_json" ]]; then
    if [[ ! -d "$scene_dir" ]]; then
      echo "[SCENE] Missing fixed scene directory: $scene_dir" >&2
      return 2
    fi
    mapfile -t _pgw_scene_files < <(find "$scene_dir" -maxdepth 1 -type f -name '*.json' -print | sort)
    if [[ ${#_pgw_scene_files[@]} -eq 0 ]]; then
      echo "[SCENE] No scene JSON found in $scene_dir" >&2
      return 2
    fi
    if [[ ${#_pgw_scene_files[@]} -ne 1 ]]; then
      echo "[SCENE] Expected exactly one scene JSON in $scene_dir, found ${#_pgw_scene_files[@]}:" >&2
      printf '  %s\n' "${_pgw_scene_files[@]}" >&2
      return 2
    fi
    scene_json="${_pgw_scene_files[0]}"
  fi

  if [[ ! -f "$scene_json" ]]; then
    echo "[SCENE] Scene JSON does not exist: $scene_json" >&2
    return 2
  fi

  local filename stem derived_out
  filename="$(basename "$scene_json")"
  stem="${filename%.json}"
  if [[ -z "$stem" || "$stem" == "$filename" ]]; then
    echo "[SCENE] Scene file must end in .json: $scene_json" >&2
    return 2
  fi
  derived_out="outputs/$stem"

  export SCENE_JSON="$scene_json"
  export SCENE_NAME="$stem"
  if [[ -n "${OUT:-}" ]]; then
    export OUT
  elif [[ -n "$explicit_out" ]]; then
    export OUT="$explicit_out"
  else
    export OUT="$derived_out"
  fi
  echo "[SCENE] $SCENE_JSON -> $OUT"
}
