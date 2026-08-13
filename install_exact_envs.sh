#!/usr/bin/env bash
set -euo pipefail

# Restore the pre-packed working environments.
#
# Place this script next to:
#   world_pipeline-linux-x86_64.tar.gz
#   worldmesh-nerfstudio-linux-x86_64.tar.gz
#   SHA256SUMS
#
# Then run:
#   bash install_exact_envs.sh
#
# Use FORCE=1 only if you intentionally want to replace existing environments.

FORCE="${FORCE:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found"; exit 1; }

CONDA_BASE="$(conda info --base)"
ENV_ROOT="${ENV_ROOT:-$CONDA_BASE/envs}"

PIPE_ARCHIVE="$SCRIPT_DIR/world_pipeline-linux-x86_64.tar.gz"
NERF_ARCHIVE="$SCRIPT_DIR/worldmesh-nerfstudio-linux-x86_64.tar.gz"
CHECKSUMS="$SCRIPT_DIR/SHA256SUMS"

for f in "$PIPE_ARCHIVE" "$NERF_ARCHIVE" "$CHECKSUMS"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done

echo "[1/3] Verifying archive checksums..."
(cd "$SCRIPT_DIR" && sha256sum -c SHA256SUMS)

install_one() {
    local archive="$1"
    local name="$2"
    local dest="$ENV_ROOT/$name"

    if [[ -e "$dest" ]]; then
        if [[ "$FORCE" != "1" ]]; then
            echo "ERROR: $dest already exists."
            echo "Set FORCE=1 to replace it."
            exit 1
        fi
        rm -rf "$dest"
    fi

    mkdir -p "$dest"
    tar -xzf "$archive" -C "$dest"

    if [[ -x "$dest/bin/conda-unpack" ]]; then
        "$dest/bin/conda-unpack"
    fi
}

echo "[2/3] Restoring world_pipeline..."
install_one "$PIPE_ARCHIVE" "world_pipeline"

echo "[3/3] Restoring worldmesh-nerfstudio..."
install_one "$NERF_ARCHIVE" "worldmesh-nerfstudio"

echo
echo "Installed:"
echo "  $ENV_ROOT/world_pipeline"
echo "  $ENV_ROOT/worldmesh-nerfstudio"
echo
echo "Activate with:"
echo "  conda activate world_pipeline"
