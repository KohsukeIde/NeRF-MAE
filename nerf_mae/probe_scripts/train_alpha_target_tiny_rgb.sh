#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NERF_MAE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TINY_RGB_WEIGHT="${TINY_RGB_WEIGHT:-0.05}"

export SAVE_NAME="${SAVE_NAME:-nerfmae_alpha_target_tiny_rgb_w${TINY_RGB_WEIGHT}}"
export RUN_TAG="${RUN_TAG:-${SAVE_NAME}}"
export PROBE_MODE="${PROBE_MODE:-custom}"
export PROBE_RGB_INPUT="${PROBE_RGB_INPUT:-zero}"
export PROBE_ALPHA_INPUT="${PROBE_ALPHA_INPUT:-zero}"
export PROBE_ALPHA_TARGET="${PROBE_ALPHA_TARGET:-keep}"
export PROBE_RGB_LOSS="${PROBE_RGB_LOSS:-removed_occupied}"
export PROBE_ALPHA_LOSS="${PROBE_ALPHA_LOSS:-removed}"
export PROBE_RGB_WEIGHT="${PROBE_RGB_WEIGHT:-${TINY_RGB_WEIGHT}}"
export PROBE_ALPHA_WEIGHT="${PROBE_ALPHA_WEIGHT:-1.0}"

exec "${NERF_MAE_DIR}/train_mae3d.sh"
