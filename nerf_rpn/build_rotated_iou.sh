#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${PROBE_ENV_PREFIX:-/home/minesawa/anaconda3/envs/nerf-mae-shortcut-probe}"
PYTHON_BIN="${PROBE_PYTHON_BIN:-${ENV_PREFIX}/bin/python}"
CC_BIN="${CC_BIN:-${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-gcc}"
CXX_BIN="${CXX_BIN:-${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-g++}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

[[ -x "${PYTHON_BIN}" ]] || { echo "[error] python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -x "${CC_BIN}" ]] || { echo "[error] gcc not found: ${CC_BIN}" >&2; exit 1; }
[[ -x "${CXX_BIN}" ]] || { echo "[error] g++ not found: ${CXX_BIN}" >&2; exit 1; }

cd "${SCRIPT_DIR}/model/rotated_iou/cuda_op"
TORCH_DONT_CHECK_COMPILER_ABI=1 \
CC="${CC_BIN}" \
CXX="${CXX_BIN}" \
CUDAHOSTCXX="${CXX_BIN}" \
CUDA_HOME="${CUDA_HOME}" \
"${PYTHON_BIN}" setup.py install
