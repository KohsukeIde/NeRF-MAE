#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NERF_MAE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export SAVE_NAME="${SAVE_NAME:-nerfmae_masked_only_rgb_loss}"
export RUN_TAG="${RUN_TAG:-${SAVE_NAME}}"
export PROBE_MODE="${PROBE_MODE:-masked_only_rgb_loss}"

exec "${NERF_MAE_DIR}/train_mae3d.sh"
