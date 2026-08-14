#!/usr/bin/env bash
set -euo pipefail

# Environment installer for the Geometry-First 3D Scene project.
#
# Copy this file and the accompanying environment/ directory into the project
# root, then run:
#
#   bash setup_project_envs.sh
#
# The installer creates:
#   world_pipeline          - Stage 00-08
#   worldmesh-nerfstudio    - Stage 09 / Nerfstudio / evaluation
#
# It does not modify project source files and does not download model weights.
# Conda subprocess output is streamed live, so each installation step is visible while it runs.
# Package-presence checks use python -c (not stdin heredocs) so missing native packages cannot be falsely skipped.
# CUDA driver linking is auto-detected; WSL libcuda.so.1 is exposed to -lcuda through a temporary build-only symlink when needed.
#
# Tested host:
#   Ubuntu 24.04
#   NVIDIA RTX 5080 16 GB
#   CUDA 12.8
#
# Default native CUDA compilation target:
#   RTX 5080 / Blackwell / sm_120
#
# Optional overrides:
#   CUDA_ARCH=8.9 BUILD_JOBS=4 bash setup_project_envs.sh
#   CUDA_ARCH=12.0 BUILD_JOBS=2 bash setup_project_envs.sh
#
# Accepted CUDA_ARCH forms:
#   12.0
#   120
#   sm_120
#
# Default:
#   CUDA_ARCH=12.0
#   BUILD_JOBS=4
#   NVCC_THREADS=1
#
# To delete and rebuild both environments:
#   RECREATE_ENVS=1 bash setup_project_envs.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$ROOT/environment"

PIPE_ENV="${PIPE_ENV:-world_pipeline}"
NERF_ENV="${NERF_ENV:-worldmesh-nerfstudio}"

CUDA_ARCH="${CUDA_ARCH:-12.0}"
BUILD_JOBS="${BUILD_JOBS:-4}"
NVCC_THREADS="${NVCC_THREADS:-1}"
RECREATE_ENVS="${RECREATE_ENVS:-0}"

WORLDMESH_COMMIT="ee19422fbc41592130636d8d2d12a3b155d60867"

PIPE_CONDA="$ENV_DIR/world_pipeline.conda.yml"
PIPE_PIP="$ENV_DIR/world_pipeline.pip.lock.txt"
NERF_CONDA="$ENV_DIR/worldmesh-nerfstudio.conda.yml"
NERF_PIP="$ENV_DIR/worldmesh-nerfstudio.pip.lock.txt"

for file in "$PIPE_CONDA" "$PIPE_PIP" "$NERF_CONDA" "$NERF_PIP"; do
    [[ -f "$file" ]] || {
        echo "ERROR: missing environment file: $file"
        exit 1
    }
done

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda is required."
    exit 1
}

command -v git >/dev/null 2>&1 || {
    echo "ERROR: git is required."
    exit 1
}

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

###############################################################################
# CUDA build configuration
###############################################################################

normalize_cuda_arch() {
    local raw="$1"
    raw="${raw#sm_}"
    raw="${raw#compute_}"

    if [[ "$raw" == *.* ]]; then
        TORCH_CUDA_ARCH="$raw"
        CUDA_ARCH_DIGITS="${raw/.}"
    else
        CUDA_ARCH_DIGITS="$raw"
        case "${#raw}" in
            2) TORCH_CUDA_ARCH="${raw:0:1}.${raw:1:1}" ;;
            3) TORCH_CUDA_ARCH="${raw:0:2}.${raw:2:1}" ;;
            *)
                echo "ERROR: invalid CUDA_ARCH: $1"
                echo "Use a value such as 12.0, 120, sm_120, 8.9 or 89."
                exit 2
                ;;
        esac
    fi
}

normalize_cuda_arch "$CUDA_ARCH"

[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: BUILD_JOBS must be a positive integer."
    exit 2
}

[[ "$NVCC_THREADS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: NVCC_THREADS must be a positive integer."
    exit 2
}

export_build_controls() {
    # PyTorch extensions
    export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH"

    # FlashAttention 2.7.3
    export FLASH_ATTN_CUDA_ARCHS="$CUDA_ARCH_DIGITS"

    # CMake/native CUDA projects
    export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH_DIGITS"
    export CUDAARCHS="$CUDA_ARCH_DIGITS"

    # Parallel build control
    export MAX_JOBS="$BUILD_JOBS"
    export CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
    export MAKEFLAGS="-j${BUILD_JOBS}"

    # Per-nvcc compilation-unit host threads
    export NVCC_THREADS="$NVCC_THREADS"
}


find_cuda_driver_lib_dir() {
    local candidates=(
        "/usr/lib/wsl/lib"
        "/usr/lib/x86_64-linux-gnu"
        "/usr/lib64/nvidia"
        "/usr/lib/nvidia"
        "/usr/local/cuda-12.8/lib64/stubs"
        "/usr/local/cuda/lib64/stubs"
    )

    local d
    for d in "${candidates[@]}"; do
        if [[ -e "$d/libcuda.so" || -e "$d/libcuda.so.1" ]]; then
            echo "$d"
            return 0
        fi
    done

    # Fall back to ldconfig when available.
    if command -v ldconfig >/dev/null 2>&1; then
        local p
        p="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so(\.1)? / {print $NF; exit}')"
        if [[ -n "$p" && -e "$p" ]]; then
            dirname "$p"
            return 0
        fi
    fi

    return 1
}

prepare_cuda_driver_link() {
    local temp_root="$1"

    CUDA_DRIVER_LIB_DIR="$(find_cuda_driver_lib_dir || true)"
    if [[ -z "${CUDA_DRIVER_LIB_DIR:-}" ]]; then
        echo "ERROR: CUDA driver library (libcuda.so/libcuda.so.1) was not found."
        echo "On WSL 2 it is normally provided by the Windows NVIDIA driver under /usr/lib/wsl/lib."
        exit 2
    fi

    CUDA_LINK_LIB_DIR="$CUDA_DRIVER_LIB_DIR"

    # Some WSL/driver setups expose only libcuda.so.1. The linker flag -lcuda
    # searches for libcuda.so, so provide a temporary build-only symlink rather
    # than changing the system installation.
    if [[ ! -e "$CUDA_DRIVER_LIB_DIR/libcuda.so" && -e "$CUDA_DRIVER_LIB_DIR/libcuda.so.1" ]]; then
        CUDA_LINK_LIB_DIR="$temp_root/libcuda-link"
        mkdir -p "$CUDA_LINK_LIB_DIR"
        ln -sf "$CUDA_DRIVER_LIB_DIR/libcuda.so.1" "$CUDA_LINK_LIB_DIR/libcuda.so"
    fi

    echo "CUDA driver library:"
    echo "  runtime: $CUDA_DRIVER_LIB_DIR"
    echo "  linker : $CUDA_LINK_LIB_DIR"
}

export_cuda_driver_link_flags() {
    export LIBRARY_PATH="$CUDA_LINK_LIB_DIR:$CUDA_DRIVER_LIB_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$CUDA_DRIVER_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LDFLAGS="-L$CUDA_LINK_LIB_DIR ${LDFLAGS:-}"
}

echo
echo "CUDA compilation:"
echo "  target      : sm_${CUDA_ARCH_DIGITS} (compute capability ${TORCH_CUDA_ARCH})"
echo "  build jobs  : ${BUILD_JOBS}"
echo "  nvcc threads: ${NVCC_THREADS}"
echo

###############################################################################
# Helpers
###############################################################################

env_exists() {
    conda env list | awk '{print $1}' | grep -qx "$1"
}

prepare_env() {
    local env_name="$1"
    local lock_file="$2"

    if [[ "$RECREATE_ENVS" == "1" ]] && env_exists "$env_name"; then
        echo "Removing existing environment: $env_name"
        conda env remove -n "$env_name" -y
    fi

    if env_exists "$env_name"; then
        echo "Reusing existing environment: $env_name"
    else
        echo "Creating environment: $env_name"
        conda env create -f "$lock_file"
    fi
}

package_version_is() {
    local env_name="$1"
    local package="$2"
    local expected="$3"

    # Do not use "python - <<HEREDOC" through conda run here. In captured mode,
    # stdin may not be forwarded as expected, which can make an empty Python
    # process exit 0 and falsely report that a missing package is installed.
    #
    # -c keeps the check independent of stdin and therefore reliable in both
    # live-output and non-live-output modes.
    conda run --no-capture-output -n "$env_name" \
        python -c '
from importlib.metadata import version
import sys
package, expected = sys.argv[1], sys.argv[2]
try:
    got = version(package)
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if got == expected else 1)
' "$package" "$expected" >/dev/null 2>&1
}

###############################################################################
# world_pipeline
###############################################################################

echo "============================================================"
echo "Creating world_pipeline"
echo "============================================================"

prepare_env "$PIPE_ENV" "$PIPE_CONDA"

echo "[1/7] PyTorch 2.11.0 + CUDA 12.8"
conda run --no-capture-output -n "$PIPE_ENV" python -m pip install \
    "torch==2.11.0+cu128" \
    "torchvision==0.26.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128

echo "[2/7] Frozen Python packages"
# --no-deps is intentional: the versions in this file came from the known
# working environment. Letting pip resolve them again can replace the required
# PyTorch/CUDA packages.
conda run --no-capture-output -n "$PIPE_ENV" python -m pip install \
    --no-deps \
    -r "$PIPE_PIP"

echo "[3/7] NATTEN 0.21.6 for torch 2.11.0 + cu128"
conda run --no-capture-output -n "$PIPE_ENV" python -m pip install \
    --no-deps \
    "NATTEN==0.21.6+torch2110cu128" \
    -f https://whl.natten.org

echo "[4/7] FlashAttention 2.7.3"

if package_version_is "$PIPE_ENV" "flash_attn" "2.7.3"; then
    echo "flash_attn==2.7.3 already installed."
else
    command -v nvcc >/dev/null 2>&1 || {
        echo "ERROR: nvcc was not found. CUDA 12.8 is used on the tested host."
        exit 2
    }

    (
        conda activate "$PIPE_ENV"
        export_build_controls
        export FLASH_ATTENTION_FORCE_BUILD="TRUE"

        python -m pip install -v \
            --no-deps \
            --no-build-isolation \
            "flash-attn==2.7.3"
    )

    if ! package_version_is "$PIPE_ENV" "flash_attn" "2.7.3"; then
        echo "ERROR: FlashAttention installation finished but flash_attn==2.7.3 is not installed."
        exit 3
    fi
fi

echo "[5/7] TRELLIS.2 native CUDA extensions"

NEED_NATIVE=0
package_version_is "$PIPE_ENV" "cumesh" "0.0.1" || NEED_NATIVE=1
package_version_is "$PIPE_ENV" "flex_gemm" "1.0.0" || NEED_NATIVE=1
package_version_is "$PIPE_ENV" "nvdiffrast" "0.4.0" || NEED_NATIVE=1
package_version_is "$PIPE_ENV" "nvdiffrec_render" "0.0.0" || NEED_NATIVE=1
package_version_is "$PIPE_ENV" "o_voxel" "0.0.1" || NEED_NATIVE=1

TMP=""

cleanup() {
    if [[ -n "${TMP:-}" && -d "$TMP" ]]; then
        rm -rf "$TMP"
    fi
}
trap cleanup EXIT

if [[ "$NEED_NATIVE" == "0" ]]; then
    echo "Native extensions already installed."
else
    command -v nvcc >/dev/null 2>&1 || {
        echo "ERROR: nvcc was not found. CUDA 12.8 is used on the tested host."
        exit 2
    }

    TMP="$(mktemp -d -t geometry-first-env-XXXXXX)"

    prepare_cuda_driver_link "$TMP"

    git clone --depth 1 --recursive \
        https://github.com/microsoft/TRELLIS.2.git \
        "$TMP/TRELLIS.2"

    # TRELLIS.2 uses these temporary paths internally.
    rm -rf \
        /tmp/extensions/nvdiffrast \
        /tmp/extensions/nvdiffrec \
        /tmp/extensions/CuMesh \
        /tmp/extensions/FlexGEMM \
        /tmp/extensions/o-voxel

    (
        conda activate "$PIPE_ENV"
        export_build_controls
        export_cuda_driver_link_flags
        export ATTN_BACKEND=sdpa

        echo "CUDA driver link flags:"
        echo "  LIBRARY_PATH=$LIBRARY_PATH"
        echo "  LDFLAGS=$LDFLAGS"

        cd "$TMP/TRELLIS.2"

        # Do not use --new-env: extensions belong in world_pipeline.
        # Do not use --basic: Python dependencies are already locked above.
        # Do not use --flash-attn: FlashAttention is installed separately with
        # an explicit single-architecture build target.
        # shellcheck disable=SC1091
        source ./setup.sh \
            --nvdiffrast \
            --nvdiffrec \
            --cumesh \
            --o-voxel \
            --flexgemm
    )

    NATIVE_BAD=0
    package_version_is "$PIPE_ENV" "cumesh" "0.0.1" || NATIVE_BAD=1
    package_version_is "$PIPE_ENV" "flex_gemm" "1.0.0" || NATIVE_BAD=1
    package_version_is "$PIPE_ENV" "nvdiffrast" "0.4.0" || NATIVE_BAD=1
    package_version_is "$PIPE_ENV" "nvdiffrec_render" "0.0.0" || NATIVE_BAD=1
    package_version_is "$PIPE_ENV" "o_voxel" "0.0.1" || NATIVE_BAD=1

    if [[ "$NATIVE_BAD" == "1" ]]; then
        echo "ERROR: one or more TRELLIS.2 native extensions were not installed correctly."
        exit 4
    fi
fi

echo "[6/7] Verifying critical world_pipeline versions"
conda run --no-capture-output -n "$PIPE_ENV" python - <<'PY'
from importlib.metadata import version

expected = {
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "numpy": "2.2.6",
    "diffusers": "0.37.1",
    "transformers": "4.57.3",
    "accelerate": "1.13.0",
    "bitsandbytes": "0.50.0",
    "NATTEN": "0.21.6+torch2110cu128",
    "flash_attn": "2.7.3",
    "cumesh": "0.0.1",
    "flex_gemm": "1.0.0",
    "nvdiffrast": "0.4.0",
    "nvdiffrec_render": "0.0.0",
    "o_voxel": "0.0.1",
    "moge": "2.0.0",
    "utils3d": "1.3",
}

mismatch = []

for package, expected_version in expected.items():
    try:
        installed = version(package)
    except Exception as exc:
        installed = f"<missing: {exc}>"

    print(f"{package}: {installed}")

    if installed != expected_version:
        mismatch.append((package, expected_version, installed))

if mismatch:
    raise SystemExit("world_pipeline version mismatch: " + repr(mismatch))
PY

echo "[7/7] Checking GPU visibility"
conda run --no-capture-output -n "$PIPE_ENV" python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0))
PY

###############################################################################
# worldmesh-nerfstudio
###############################################################################

echo
echo "============================================================"
echo "Creating worldmesh-nerfstudio"
echo "============================================================"

prepare_env "$NERF_ENV" "$NERF_CONDA"

echo "[1/4] Recorded PyTorch versions"
conda run --no-capture-output -n "$NERF_ENV" python -m pip install \
    "torch==2.13.0" \
    "torchvision==0.28.0"

echo "[2/4] Frozen Python packages"
conda run --no-capture-output -n "$NERF_ENV" python -m pip install \
    --no-deps \
    -r "$NERF_PIP"

echo "[3/4] WorldMesh-compatible Nerfstudio"
# Always reinstall the exact recorded source revision without resolving deps.
conda run --no-capture-output -n "$NERF_ENV" python -m pip install \
    --no-deps \
    --force-reinstall \
    "git+https://github.com/mschneider456/worldmesh.git@${WORLDMESH_COMMIT}#subdirectory=nerfstudio"

echo "[4/4] Verifying critical worldmesh-nerfstudio versions"
conda run --no-capture-output -n "$NERF_ENV" python - <<'PY'
from importlib.metadata import version

expected = {
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "numpy": "1.26.4",
    "nerfstudio": "1.1.5",
    "nerfacc": "0.5.2",
    "gsplat": "1.4.0",
    "timm": "0.6.7",
    "transformers": "4.29.2",
    "viser": "1.0.0",
}

mismatch = []

for package, expected_version in expected.items():
    try:
        installed = version(package)
    except Exception as exc:
        installed = f"<missing: {exc}>"

    print(f"{package}: {installed}")

    if installed != expected_version:
        mismatch.append((package, expected_version, installed))

if mismatch:
    raise SystemExit("worldmesh-nerfstudio version mismatch: " + repr(mismatch))
PY

echo
echo "============================================================"
echo "Environment setup complete"
echo "============================================================"
echo
echo "Created environments:"
echo "  $PIPE_ENV"
echo "  $NERF_ENV"
echo
echo "CUDA native build target:"
echo "  sm_${CUDA_ARCH_DIGITS}"
echo
echo "Parallel build jobs:"
echo "  ${BUILD_JOBS}"
echo
echo "The project source tree was not modified."
echo "Pretrained model weights were not downloaded."
echo
