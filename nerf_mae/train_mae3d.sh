#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/probe_scripts/_probe_common.sh"

DATA_ROOT="${DATA_ROOT:-../dataset/pretrain}"
dataset_name="${DATASET_NAME:-nerfmae}"
resolution="${RESOLUTION:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_EPOCHS="${NUM_EPOCHS:-2000}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
LOG_INTERVAL="${LOG_INTERVAL:-30}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
MASKING_PROB="${MASKING_PROB:-0.75}"
PERCENT_TRAIN="${PERCENT_TRAIN:-1.0}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
USE_WANDB="${USE_WANDB:-1}"
PROBE_MODE="${PROBE_MODE:-}"
PROBE_RGB_INPUT="${PROBE_RGB_INPUT:-}"
PROBE_ALPHA_INPUT="${PROBE_ALPHA_INPUT:-}"
PROBE_ALPHA_TARGET="${PROBE_ALPHA_TARGET:-}"
PROBE_RGB_LOSS="${PROBE_RGB_LOSS:-}"
PROBE_ALPHA_LOSS="${PROBE_ALPHA_LOSS:-}"
PROBE_RGB_WEIGHT="${PROBE_RGB_WEIGHT:-}"
PROBE_ALPHA_WEIGHT="${PROBE_ALPHA_WEIGHT:-}"
PROBE_ALPHA_THRESHOLD="${PROBE_ALPHA_THRESHOLD:-}"
SEED="${SEED:-}"
DETERMINISTIC="${DETERMINISTIC:-0}"

if [[ -z "${resolution}" ]]; then
  resolution=160
  if [[ "${dataset_name}" == "hypersim" ]]; then
    resolution=200
  fi
fi

num_gpus="$(probe_count_gpus "${GPU_IDS}")"
DEFAULT_BATCH_SIZE=$((num_gpus * BATCH_SIZE_PER_GPU))
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
SAVE_NAME="${SAVE_NAME:-${dataset_name}_all}"
RUN_TAG="${RUN_TAG:-${SAVE_NAME}}"
SAVE_PATH="${SAVE_PATH:-../output/nerf_mae/results/${SAVE_NAME}}"

cmd=(
  python3 -u run_swin_mae3d.py
  --mode train
  --backbone_type swin_s
  --features_path "${DATA_ROOT}/features"
  --num_epochs "${NUM_EPOCHS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --log_interval "${LOG_INTERVAL}"
  --eval_interval "${EVAL_INTERVAL}"
  --normalize_density
  --log_to_file
  --batch_size "${BATCH_SIZE}"
  --resolution "${resolution}"
  --masking_prob "${MASKING_PROB}"
  --dataset "${dataset_name}"
  --dataset_split "${DATA_ROOT}/${dataset_name}_split.npz"
  --save_path "${SAVE_PATH}"
  --gpus "${GPU_IDS}"
  --percent_train "${PERCENT_TRAIN}"
  --tags "${RUN_TAG}"
)

if [[ "${USE_WANDB}" == "1" ]]; then
  cmd+=(--wandb)
fi
if [[ -n "${PROBE_MODE}" ]]; then
  cmd+=(--probe_mode "${PROBE_MODE}")
fi
if [[ -n "${PROBE_RGB_INPUT}" ]]; then
  cmd+=(--probe_rgb_input "${PROBE_RGB_INPUT}")
fi
if [[ -n "${PROBE_ALPHA_INPUT}" ]]; then
  cmd+=(--probe_alpha_input "${PROBE_ALPHA_INPUT}")
fi
if [[ -n "${PROBE_ALPHA_TARGET}" ]]; then
  cmd+=(--probe_alpha_target "${PROBE_ALPHA_TARGET}")
fi
if [[ -n "${PROBE_RGB_LOSS}" ]]; then
  cmd+=(--probe_rgb_loss "${PROBE_RGB_LOSS}")
fi
if [[ -n "${PROBE_ALPHA_LOSS}" ]]; then
  cmd+=(--probe_alpha_loss "${PROBE_ALPHA_LOSS}")
fi
if [[ -n "${PROBE_RGB_WEIGHT}" ]]; then
  cmd+=(--probe_rgb_weight "${PROBE_RGB_WEIGHT}")
fi
if [[ -n "${PROBE_ALPHA_WEIGHT}" ]]; then
  cmd+=(--probe_alpha_weight "${PROBE_ALPHA_WEIGHT}")
fi
if [[ -n "${PROBE_ALPHA_THRESHOLD}" ]]; then
  cmd+=(--probe_alpha_threshold "${PROBE_ALPHA_THRESHOLD}")
fi
if [[ -n "${SEED}" ]]; then
  cmd+=(--seed "${SEED}")
fi
if [[ "${DETERMINISTIC}" == "1" ]]; then
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  cmd+=(--deterministic)
fi

"${cmd[@]}"
