#!/usr/bin/env bash
# Shared resumable-stage helpers. Source this file from run_stage*.sh.
#
# FORCE=1 rebuilds only the stage script invoked directly by the user.
# Dependency stages are still allowed to reuse matching completed outputs.
# FORCE_ALL=1 rebuilds the full dependency chain.

_pgw_truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

pgw_stage_key() {
  # Hash only dependencies explicitly supplied by the calling stage. Avoid
  # repository-wide fingerprints, because unrelated edits must not invalidate
  # every expensive upstream stage.
  python tools/cache_key.py "$@"
}

pgw_stage_complete() {
  local stage_dir="$1"
  shift
  local args=("$stage_dir")
  local required
  for required in "$@"; do
    args+=(--require "$required")
  done
  python tools/stage_complete.py "${args[@]}"
}

pgw_should_skip() {
  local stage_dir="$1"
  local expected="$2"
  shift 2

  # FORCE applies to the current script only. FORCE_ALL applies to every stage
  # in the chain and is explicitly forwarded by pgw_run_dependency.
  _pgw_truthy "${FORCE_ALL:-0}" && return 1
  _pgw_truthy "${FORCE:-0}" && return 1
  pgw_stage_complete "$stage_dir" "$@" || return 1

  # One-time migration for outputs produced before output manifests existed.
  # Their required artifacts are already complete, so adopt the current scoped
  # fingerprint without rerunning the expensive stage.
  if [[ ! -f "$stage_dir/.output_manifest.json" ]]; then
    python tools/stage_manifest.py write "$stage_dir"
    printf '%s\n' "$expected" > "$stage_dir/.input_hash"
    return 0
  fi

  [[ -f "$stage_dir/.input_hash" ]] || return 1
  [[ "$(cat "$stage_dir/.input_hash")" == "$expected" ]]
}

# Backward-compatible name used by external wrappers.
pgw_cache_hit() {
  pgw_should_skip "$@"
}

pgw_cache_mark() {
  local stage_dir="$1"
  local key="$2"
  printf '%s\n' "$key" > "$stage_dir/.input_hash"
  python tools/stage_manifest.py write "$stage_dir"
}

pgw_run_dependency() {
  local script="$1"
  shift
  local dependency_force=0

  # A target-only FORCE must never leak into upstream stages. FORCE_ALL is the
  # explicit opt-in for rebuilding the complete chain.
  if _pgw_truthy "${FORCE_ALL:-0}"; then
    dependency_force=1
  fi

  FORCE="$dependency_force" \
  FORCE_ALL="${FORCE_ALL:-0}" \
  bash "$script" "$@"
}

pgw_rebuild_message() {
  local label="$1"
  local stage_dir="$2"
  local expected="$3"

  if _pgw_truthy "${FORCE_ALL:-0}"; then
    echo "[$label] Rebuilding because FORCE_ALL=1 was requested."
  elif _pgw_truthy "${FORCE:-0}"; then
    echo "[$label] Rebuilding because FORCE=1 was requested for this target stage."
  elif [[ ! -d "$stage_dir" ]]; then
    echo "[$label] Rebuilding because no previous stage directory exists."
  elif [[ -f "$stage_dir/.failed" ]]; then
    echo "[$label] Rebuilding because the previous run left a .failed marker."
  elif [[ ! -f "$stage_dir/.done" ]]; then
    echo "[$label] Rebuilding because the previous run did not publish .done."
  elif [[ ! -f "$stage_dir/.input_hash" ]]; then
    echo "[$label] Rebuilding because the cached input fingerprint is missing."
  elif [[ "$(cat "$stage_dir/.input_hash")" != "$expected" ]]; then
    echo "[$label] Rebuilding because the scoped input/implementation fingerprint changed."
  else
    echo "[$label] Rebuilding because one or more required outputs/manifests are incomplete."
  fi
}

pgw_skip_message() {
  local label="$1"
  echo "[$label] Complete outputs with matching inputs found; skipping. Use FORCE=1 to rebuild only this target stage, or FORCE_ALL=1 to rebuild the full chain."
}

pgw_require_inputs() {
  local stage_label="$1"
  local producer_label="$2"
  shift 2
  local missing=()
  local required
  for required in "$@"; do
    if [[ ! -e "$required" ]]; then
      missing+=("$required")
    elif [[ -f "$required" && ! -s "$required" ]]; then
      missing+=("$required (empty)")
    fi
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    return 0
  fi
  echo "[$stage_label] Missing required inputs produced by $producer_label:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  echo "[$stage_label] This runner executes only Stage$stage_label and will not invoke earlier stages." >&2
  echo "[$stage_label] Run the required producer stage explicitly, or use bash run_pipeline.sh with an appropriate --from/--to range." >&2
  return 2
}
