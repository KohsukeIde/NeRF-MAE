#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_PYTHON="${BASE_PYTHON:-/groups/gag51404/.conda/bin/python}"
PROBE_ENV_PREFIX="${PROBE_ENV_PREFIX:-${ROOT_DIR}/.venv-abci3}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8/11.8.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.0}"

if [[ ! -x "${BASE_PYTHON}" ]]; then
  echo "[error] BASE_PYTHON is not executable: ${BASE_PYTHON}" >&2
  exit 1
fi

source /etc/profile.d/modules.sh
module load "${CUDA_MODULE}"

if [[ ! -x "${PROBE_ENV_PREFIX}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv "${PROBE_ENV_PREFIX}"
fi

"${PROBE_ENV_PREFIX}/bin/python" -m pip install --upgrade pip wheel setuptools
"${PROBE_ENV_PREFIX}/bin/python" -m pip install \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"
"${PROBE_ENV_PREFIX}/bin/python" -m pip install \
  "numpy==1.26.4" \
  imageio \
  configargparse \
  wandb \
  matplotlib \
  tqdm \
  pandas \
  pytransform3d \
  kornia \
  h5py \
  opencv-python \
  einops \
  torchmetrics \
  scikit-learn

"${PROBE_ENV_PREFIX}/bin/python" - <<'PY'
import importlib

mods = [
    "torch",
    "torchvision",
    "numpy",
    "wandb",
    "matplotlib",
    "tqdm",
    "pandas",
    "h5py",
    "cv2",
    "einops",
    "torchmetrics",
    "sklearn",
]
for name in mods:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "ok"))
PY

echo "[info] ABCI3 environment ready: ${PROBE_ENV_PREFIX}"
