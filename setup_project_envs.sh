#!/usr/bin/env bash
set -euo pipefail

# Standalone environment installer for:
# A Fully Automated Geometry-First Framework for Controllable Indoor 3D Scene Generation
#
# This script DOES NOT modify the project source tree.
# External repositories are stored under ~/.cache/geometry_first_3d_scene_generation.
#
# Tested host configuration used by the project:
#   Ubuntu 24.04
#   NVIDIA RTX 5080 16 GB
#   CUDA 12.8
#   Python 3.10
#
# Usage:
#   bash setup_project_envs.sh
#
# Optional clean rebuild:
#   RECREATE_ENVS=1 bash setup_project_envs.sh
#
# Optional external dependency refs:
#   PIXAL3D_REF=master TRELLIS2_REF=main bash setup_project_envs.sh

PIPELINE_ENV="${PIPELINE_ENV:-world_pipeline}"

# IMPORTANT:
# The current project config refers to "worldmesh-nerfstudio" directly for Stage 09.
# Keeping this name allows the existing project to run unchanged.
NERF_ENV="${NERF_ENV:-worldmesh-nerfstudio}"

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
RECREATE_ENVS="${RECREATE_ENVS:-0}"
CACHE_ROOT="${GEOMETRY_FIRST_CACHE:-$HOME/.cache/geometry_first_3d_scene_generation}"
DEPS_ROOT="$CACHE_ROOT/deps"

PIXAL3D_REPO="$DEPS_ROOT/Pixal3D"
TRELLIS2_REPO="$DEPS_ROOT/TRELLIS.2"
WORLDMESH_REPO="$DEPS_ROOT/worldmesh"

PIXAL3D_REF="${PIXAL3D_REF:-master}"
TRELLIS2_REF="${TRELLIS2_REF:-main}"
WORLDMESH_REF="${WORLDMESH_REF:-ee19422fbc41592130636d8d2d12a3b155d60867}"

mkdir -p "$DEPS_ROOT"

log() {
    printf '\n\033[1;36m[%s]\033[0m %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
    printf '\nERROR: %s\n' "$*" >&2
    exit 1
}

command -v conda >/dev/null 2>&1 || die "conda was not found. Install Miniconda/Anaconda first."
command -v git >/dev/null 2>&1 || die "git was not found."

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

env_exists() {
    conda env list | awk '{print $1}' | grep -qx "$1"
}

prepare_env() {
    local env_name="$1"

    if [[ "$RECREATE_ENVS" == "1" ]] && env_exists "$env_name"; then
        log "Removing existing environment: $env_name"
        conda env remove -n "$env_name" -y
    fi

    if ! env_exists "$env_name"; then
        log "Creating $env_name (Python $PYTHON_VERSION)"
        conda create -n "$env_name" -y "python=$PYTHON_VERSION" pip
    else
        log "Reusing existing environment: $env_name"
    fi

    conda run -n "$env_name" python -m pip install --upgrade pip setuptools wheel
}

clone_checkout() {
    local url="$1"
    local dest="$2"
    local ref="$3"

    if [[ ! -d "$dest/.git" ]]; then
        log "Cloning $(basename "$dest")"
        git clone --recursive "$url" "$dest"
    fi

    log "Checking out $(basename "$dest") at $ref"
    git -C "$dest" fetch --tags origin
    if git -C "$dest" rev-parse --verify "$ref^{commit}" >/dev/null 2>&1; then
        git -C "$dest" checkout "$ref"
    else
        git -C "$dest" fetch origin "$ref"
        git -C "$dest" checkout FETCH_HEAD
    fi
    git -C "$dest" submodule update --init --recursive
}

python_imports_ok() {
    local env_name="$1"
    shift
    local imports="$*"
    conda run -n "$env_name" python - "$imports" <<'PY' >/dev/null 2>&1
import importlib, sys
mods = sys.argv[1].split()
for m in mods:
    importlib.import_module(m)
PY
}

###############################################################################
# 1. world_pipeline: Stage 00-08
###############################################################################

prepare_env "$PIPELINE_ENV"

log "Installing PyTorch 2.11.0 + CUDA 12.8 runtime into $PIPELINE_ENV"
conda run -n "$PIPELINE_ENV" python -m pip install \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

log "Installing core Stage 00-08 Python dependencies"
conda run -n "$PIPELINE_ENV" python -m pip install \
    numpy==2.2.6 \
    scipy==1.15.3 \
    Pillow==12.0.0 \
    jsonschema==4.26.0 \
    opencv-python-headless==4.12.0.88 \
    diffusers==0.37.1 \
    transformers==4.57.3 \
    accelerate==1.13.0 \
    bitsandbytes==0.50.0 \
    huggingface_hub==0.36.2 \
    safetensors==0.8.0 \
    sentencepiece==0.2.2 \
    einops==0.8.2 \
    kornia==0.8.2 \
    timm==1.0.22 \
    imageio==2.37.2 \
    imageio-ffmpeg==0.6.0 \
    easydict==1.13 \
    trimesh==4.10.1 \
    plyfile==1.1.3 \
    lpips==0.1.4 \
    pandas==2.3.3 \
    gradio==6.0.1 \
    tensorboard==2.21.0 \
    tqdm==4.67.1 \
    zstandard==0.25.0 \
    ninja==1.13.0 \
    cmake \
    packaging \
    requests

###############################################################################
# Pixal3D + TRELLIS.2 runtime
###############################################################################

clone_checkout "https://github.com/TencentARC/Pixal3D.git" "$PIXAL3D_REPO" "$PIXAL3D_REF"
clone_checkout "https://github.com/microsoft/TRELLIS.2.git" "$TRELLIS2_REPO" "$TRELLIS2_REF"

log "Installing Pixal3D Python requirements"
conda run -n "$PIPELINE_ENV" python -m pip install -r "$PIXAL3D_REPO/requirements.txt"

# Restore the exact versions used by the working project environment in case
# transitive dependency resolution changed them.
conda run -n "$PIPELINE_ENV" python -m pip install \
    transformers==4.57.3 \
    diffusers==0.37.1 \
    accelerate==1.13.0 \
    timm==1.0.22 \
    kornia==0.8.2 \
    opencv-python-headless==4.12.0.88 \
    Pillow==12.0.0 \
    trimesh==4.10.1 \
    plyfile==1.1.3

log "Installing NATTEN build matching torch 2.11.0 + cu128"
conda run -n "$PIPELINE_ENV" python -m pip install \
    "natten==0.21.6+torch2110cu128" \
    -f https://whl.natten.org

# TRELLIS.2's native extensions are compiled only when missing.
# We intentionally do NOT run TRELLIS.2 --basic because it contains apt/sudo
# operations. All Python-level basic dependencies are installed above.
TRELLIS_FLAGS=()

if ! python_imports_ok "$PIPELINE_ENV" nvdiffrast; then
    TRELLIS_FLAGS+=(--nvdiffrast)
fi
if ! python_imports_ok "$PIPELINE_ENV" nvdiffrec_render; then
    TRELLIS_FLAGS+=(--nvdiffrec)
fi
if ! python_imports_ok "$PIPELINE_ENV" cumesh; then
    TRELLIS_FLAGS+=(--cumesh)
fi
if ! python_imports_ok "$PIPELINE_ENV" o_voxel; then
    TRELLIS_FLAGS+=(--o-voxel)
fi
if ! python_imports_ok "$PIPELINE_ENV" flex_gemm; then
    TRELLIS_FLAGS+=(--flexgemm)
fi

if (( ${#TRELLIS_FLAGS[@]} > 0 )); then
    log "Building missing TRELLIS.2 CUDA extensions: ${TRELLIS_FLAGS[*]}"

    # Native CUDA extensions require a working compiler toolchain.
    # The project was tested with CUDA 12.8 on Ubuntu 24.04.
    if ! command -v nvcc >/dev/null 2>&1; then
        cat >&2 <<'EOF'

CUDA nvcc was not found on PATH.
The Python environments themselves are created automatically, but the
TRELLIS.2 native extensions need a usable CUDA compiler toolchain.

Tested project host:
  Ubuntu 24.04
  NVIDIA RTX 5080 16 GB
  CUDA 12.8

Install/fix the host CUDA toolchain, then run this script again.
EOF
        exit 2
    fi

    (
        conda activate "$PIPELINE_ENV"
        export ATTN_BACKEND=sdpa
        export NATTEN_N_WORKERS="${NATTEN_N_WORKERS:-8}"

        # TRELLIS.2 uses /tmp/extensions internally. Remove stale build clones
        # so rerunning this standalone installer stays deterministic.
        rm -rf \
            /tmp/extensions/nvdiffrast \
            /tmp/extensions/nvdiffrec \
            /tmp/extensions/CuMesh \
            /tmp/extensions/FlexGEMM \
            /tmp/extensions/o-voxel

        cd "$TRELLIS2_REPO"
        # shellcheck disable=SC1091
        source ./setup.sh "${TRELLIS_FLAGS[@]}"
    )
else
    log "TRELLIS.2 native extensions are already installed"
fi

###############################################################################
# Conda activation hook for Stage 04.
# This changes the environment only; it does not touch the project source tree.
###############################################################################

PIPE_PREFIX="$(conda run -n "$PIPELINE_ENV" python -c 'import sys; print(sys.prefix)')"
mkdir -p "$PIPE_PREFIX/etc/conda/activate.d"

cat > "$PIPE_PREFIX/etc/conda/activate.d/geometry_first_project.sh" <<EOF
export PIXAL3D_REPO="$PIXAL3D_REPO"
export PIXAL3D_PYTHON="$PIPE_PREFIX/bin/python"
export ATTN_BACKEND="sdpa"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
EOF

###############################################################################
# 2. worldmesh-nerfstudio: Stage 09 + rendering/evaluation
###############################################################################

prepare_env "$NERF_ENV"

log "Installing the Stage 09 PyTorch runtime"
conda run -n "$NERF_ENV" python -m pip install \
    torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cu130

# Keep NumPy at the version from the verified working Nerfstudio environment.
conda run -n "$NERF_ENV" python -m pip install numpy==1.26.4

clone_checkout \
    "https://github.com/mschneider456/worldmesh.git" \
    "$WORLDMESH_REPO" \
    "$WORLDMESH_REF"

log "Installing the WorldMesh-compatible Nerfstudio fork"
conda run -n "$NERF_ENV" python -m pip install -e "$WORLDMESH_REPO/nerfstudio"

# Reassert the versions that are important to this fork and the existing
# Stage09/09E guard logic.
conda run -n "$NERF_ENV" python -m pip install \
    numpy==1.26.4 \
    nerfacc==0.5.2 \
    gsplat==1.4.0 \
    timm==0.6.7 \
    transformers==4.29.2 \
    viser==1.0.0

###############################################################################
# Verification
###############################################################################

log "Verifying $PIPELINE_ENV"
conda run -n "$PIPELINE_ENV" python - <<PY
import sys
import torch
import numpy
import scipy
import cv2
import PIL
import jsonschema
import diffusers
import transformers
import accelerate
import bitsandbytes
import huggingface_hub
import natten
import nvdiffrast
import nvdiffrec_render
import cumesh
import o_voxel
import flex_gemm

from diffusers import FluxControlInpaintPipeline, Flux2KleinPipeline
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

sys.path.insert(0, r"$PIXAL3D_REPO")
import pixal3d

print("world_pipeline OK")
print("  Python:", sys.version.split()[0])
print("  torch:", torch.__version__)
print("  torch CUDA:", torch.version.cuda)
print("  CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  GPU:", torch.cuda.get_device_name(0))
print("  Pixal3D repo:", r"$PIXAL3D_REPO")
print("  NATTEN:", getattr(natten, "__version__", "unknown"))
PY

log "Verifying $NERF_ENV"
conda run -n "$NERF_ENV" python - <<'PY'
import sys
import torch
import numpy
import nerfstudio
import gsplat
import nerfacc
import timm
import transformers

from nerfstudio.models.depth_splatfacto import DepthSplatfactoModelConfig
from nerfstudio.configs.method_configs import all_methods

assert "depth-splatfacto" in all_methods

print("worldmesh-nerfstudio OK")
print("  Python:", sys.version.split()[0])
print("  torch:", torch.__version__)
print("  torch CUDA:", torch.version.cuda)
print("  CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  GPU:", torch.cuda.get_device_name(0))
print("  numpy:", numpy.__version__)
print("  timm:", timm.__version__)
print("  transformers:", transformers.__version__)
PY

###############################################################################
# Record exact installed state outside the project
###############################################################################

STATE_DIR="$CACHE_ROOT/installed_state"
mkdir -p "$STATE_DIR"

conda run -n "$PIPELINE_ENV" python -m pip freeze > "$STATE_DIR/${PIPELINE_ENV}_pip_freeze.txt"
conda run -n "$NERF_ENV" python -m pip freeze > "$STATE_DIR/${NERF_ENV}_pip_freeze.txt"

{
    echo "Pixal3D $(git -C "$PIXAL3D_REPO" rev-parse HEAD)"
    echo "TRELLIS.2 $(git -C "$TRELLIS2_REPO" rev-parse HEAD)"
    echo "WorldMesh $(git -C "$WORLDMESH_REPO" rev-parse HEAD)"
} > "$STATE_DIR/external_git_revisions.txt"

log "Environment setup completed"
cat <<EOF

Created/updated:

  Stage 00-08:
    conda activate $PIPELINE_ENV

  Stage 09 + Nerfstudio evaluation:
    $NERF_ENV

The project source tree was not modified.

External repositories:
  $DEPS_ROOT

Exact installed package state:
  $STATE_DIR

Important:
  The unmodified project currently names the Stage 09 environment
  "worldmesh-nerfstudio" in configs/gaussian_pipeline.json, so this installer
  intentionally keeps that environment name.

Models such as FLUX.1, FLUX.2 and Pixal3D are NOT downloaded here.
The project stages download missing model weights when they are first needed.

EOF
