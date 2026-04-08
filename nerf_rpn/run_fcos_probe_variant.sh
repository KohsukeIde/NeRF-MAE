#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRETRAIN_SAVE_NAME="${PRETRAIN_SAVE_NAME:?PRETRAIN_SAVE_NAME is required}"
VARIANT_NAME="${VARIANT_NAME:?VARIANT_NAME is required}"

GPU_IDS="${GPU_IDS:-0}"
DATASET_NAME="${DATASET_NAME:-front3d}"
SPLIT_NAME="${SPLIT_NAME:-3dfront}"
PERCENT_TRAIN="${PERCENT_TRAIN:-1.0}"
FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS:-100}"
FCOS_LR="${FCOS_LR:-1e-4}"
FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY:-1e-3}"
FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU:-2}"
FCOS_LOG_INTERVAL="${FCOS_LOG_INTERVAL:-10}"
FCOS_EVAL_INTERVAL="${FCOS_EVAL_INTERVAL:-10}"
USE_WANDB="${USE_WANDB:-0}"

PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-../output/nerf_mae/results/${PRETRAIN_SAVE_NAME}/model_best.pt}"
SAVE_NAME="${SAVE_NAME:-${PRETRAIN_SAVE_NAME}_fcos${FCOS_NUM_EPOCHS}}"
SAVE_PATH="${SAVE_PATH:-../output/nerf_rpn/results/${SAVE_NAME}}"
EVAL_SAVE_NAME="${EVAL_SAVE_NAME:-${SAVE_NAME}_eval}"
EVAL_SAVE_PATH="${EVAL_SAVE_PATH:-../output/nerf_rpn/results/${EVAL_SAVE_NAME}}"

cd "${SCRIPT_DIR}"

export PATH="/home/minesawa/anaconda3/envs/nerf-mae-shortcut-probe/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export DATASET_NAME
export SPLIT_NAME
export GPU_IDS
export PERCENT_TRAIN
export NUM_EPOCHS="${FCOS_NUM_EPOCHS}"
export LR="${FCOS_LR}"
export WEIGHT_DECAY="${FCOS_WEIGHT_DECAY}"
export BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU}"
export LOG_INTERVAL="${FCOS_LOG_INTERVAL}"
export EVAL_INTERVAL="${FCOS_EVAL_INTERVAL}"
export MAE_CHECKPOINT="${PRETRAIN_CHECKPOINT}"
export SAVE_NAME
export SAVE_PATH
export RUN_TAG="${SAVE_NAME}"
export USE_WANDB

bash "${SCRIPT_DIR}/train_fcos_pretrained.sh"

BEST_CHECKPOINT="$(
  find "${SAVE_PATH}" -maxdepth 1 -name 'model_best_ap50_ap25_*.pt' -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | awk '{print $2}'
)"

[[ -n "${BEST_CHECKPOINT}" ]] || {
  echo "[error] no FCOS checkpoint found in ${SAVE_PATH}" >&2
  exit 1
}

export CHECKPOINT="${BEST_CHECKPOINT}"
export SAVE_NAME="${EVAL_SAVE_NAME}"
export SAVE_PATH="${EVAL_SAVE_PATH}"

bash "${SCRIPT_DIR}/test_fcos_pretrained.sh"
