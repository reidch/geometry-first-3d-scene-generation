#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-pgw_flux}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment $ENV_NAME already exists; updating it."
else
  conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$ROOT_DIR/requirements.txt"
python -m pip install -r "$ROOT_DIR/requirements-diffusion.txt"

python - <<'PY'
import torch
print("FLUX runtime ready")
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("archs:", torch.cuda.get_arch_list())
    if torch.cuda.get_device_capability(0) == (12, 0) and "sm_120" not in torch.cuda.get_arch_list():
        raise SystemExit("PyTorch wheel does not include sm_120 support.")
PY

python "$ROOT_DIR/tools/flux_runtime_doctor.py"
python "$ROOT_DIR/tools/depth_runtime_doctor.py"

echo
echo "Activate the main pipeline environment with:"
echo "  conda activate $ENV_NAME"
