#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NERF_MAE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export SAVE_NAME="${SAVE_NAME:-nerfmae_alpha_target_only}"
export RUN_TAG="${RUN_TAG:-${SAVE_NAME}}"
export PROBE_MODE="${PROBE_MODE:-custom}"
export PROBE_RGB_INPUT="${PROBE_RGB_INPUT:-zero}"
export PROBE_ALPHA_INPUT="${PROBE_ALPHA_INPUT:-zero}"
export PROBE_RGB_LOSS="${PROBE_RGB_LOSS:-none}"
export PROBE_ALPHA_LOSS="${PROBE_ALPHA_LOSS:-removed}"

exec "${NERF_MAE_DIR}/train_mae3d.sh"
