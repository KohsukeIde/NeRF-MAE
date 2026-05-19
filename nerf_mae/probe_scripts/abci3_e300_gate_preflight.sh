#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PROBE_ENV_PREFIX="${PROBE_ENV_PREFIX:-${ROOT_DIR}/.venv-abci3}"
PRETRAIN_DATA_ROOT="${PRETRAIN_DATA_ROOT:-${ROOT_DIR}/dataset/pretrain}"
FCOS_DATA_ROOT="${FCOS_DATA_ROOT:-${ROOT_DIR}/dataset/finetune/front3d_rpn_data}"
REQUIRE_PRETRAIN_DATA="${REQUIRE_PRETRAIN_DATA:-1}"
REQUIRE_FCOS_DATA="${REQUIRE_FCOS_DATA:-1}"

failed=0

check_path() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    echo "[ok] ${label}: ${path}"
  else
    echo "[missing] ${label}: ${path}" >&2
    failed=1
  fi
}

check_cmd() {
  local cmd="$1"
  if command -v "${cmd}" >/dev/null 2>&1; then
    echo "[ok] command ${cmd}: $(command -v "${cmd}")"
  else
    echo "[missing] command ${cmd}" >&2
    failed=1
  fi
}

check_cmd qsub
check_cmd qstat
check_path "python" "${PROBE_ENV_PREFIX}/bin/python"

if [[ "${REQUIRE_PRETRAIN_DATA}" == "1" ]]; then
  check_path "pretrain features" "${PRETRAIN_DATA_ROOT}/features"
  check_path "pretrain split" "${PRETRAIN_DATA_ROOT}/nerfmae_split.npz"
fi
if [[ "${REQUIRE_FCOS_DATA}" == "1" ]]; then
  check_path "Front3D features" "${FCOS_DATA_ROOT}/features"
  check_path "Front3D boxes" "${FCOS_DATA_ROOT}/obb"
  check_path "Front3D split" "${FCOS_DATA_ROOT}/3dfront_split.npz"
fi

if [[ -x "${PROBE_ENV_PREFIX}/bin/python" ]]; then
  "${PROBE_ENV_PREFIX}/bin/python" - <<'PY' || failed=1
import importlib

required = [
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
for name in required:
    mod = importlib.import_module(name)
    print(f"[ok] python import {name}: {getattr(mod, '__version__', 'ok')}")
PY
fi

if (( failed != 0 )); then
  cat >&2 <<EOF
[error] ABCI3 e300 gate preflight failed.

Set these if the defaults are wrong:
  PROBE_ENV_PREFIX=${PROBE_ENV_PREFIX}
  PRETRAIN_DATA_ROOT=${PRETRAIN_DATA_ROOT}
  FCOS_DATA_ROOT=${FCOS_DATA_ROOT}

For a fresh env, run:
  PROBE_ENV_PREFIX='${PROBE_ENV_PREFIX}' bash '${SCRIPT_DIR}/setup_abci3_env.sh'

For data links, run:
  PRETRAIN_DATA_SRC=/path/to/pretrain \\
  FCOS_DATA_SRC=/path/to/front3d_rpn_data \\
  bash '${SCRIPT_DIR}/setup_abci3_data_links.sh'
EOF
  exit 1
fi

echo "[ok] ABCI3 e300 gate preflight passed"
