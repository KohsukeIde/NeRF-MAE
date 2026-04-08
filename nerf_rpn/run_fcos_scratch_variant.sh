#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/nerf_mae/probe_scripts/_probe_common.sh"

GPU_IDS="${GPU_IDS:-0}"
DATASET_NAME="${DATASET_NAME:-front3d}"
SPLIT_NAME="${SPLIT_NAME:-3dfront}"
RESOLUTION="${RESOLUTION:-}"
DATA_ROOT="${DATA_ROOT:-../dataset/finetune/${DATASET_NAME}_rpn_data}"
PERCENT_TRAIN="${PERCENT_TRAIN:-1.0}"
FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS:-100}"
FCOS_LR="${FCOS_LR:-1e-4}"
FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY:-1e-3}"
FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU:-2}"
FCOS_EVAL_BATCH_SIZE="${FCOS_EVAL_BATCH_SIZE:-2}"
FCOS_LOG_INTERVAL="${FCOS_LOG_INTERVAL:-10}"
FCOS_EVAL_INTERVAL="${FCOS_EVAL_INTERVAL:-10}"
CLIP_GRAD_NORM="${CLIP_GRAD_NORM:-0.1}"
NMS_THRESH="${NMS_THRESH:-0.3}"
CENTER_SAMPLING_RADIUS="${CENTER_SAMPLING_RADIUS:-1.5}"
IOU_LOSS_TYPE="${IOU_LOSS_TYPE:-iou}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-3}"
USE_WANDB="${USE_WANDB:-0}"

if [[ -z "${RESOLUTION}" ]]; then
  RESOLUTION=160
  if [[ "${DATASET_NAME}" == "hypersim" ]]; then
    RESOLUTION=200
  fi
fi

NUM_GPUS="$(probe_count_gpus "${GPU_IDS}")"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((NUM_GPUS * FCOS_BATCH_SIZE_PER_GPU))}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${FCOS_EVAL_BATCH_SIZE}}"

SAVE_NAME="${SAVE_NAME:-${DATASET_NAME}_scratch_fcos${FCOS_NUM_EPOCHS}}"
RUN_TAG="${RUN_TAG:-${SAVE_NAME}}"
SAVE_PATH="${SAVE_PATH:-../output/nerf_rpn/results/${SAVE_NAME}}"
EVAL_SAVE_NAME="${EVAL_SAVE_NAME:-${SAVE_NAME}_eval}"
EVAL_SAVE_PATH="${EVAL_SAVE_PATH:-../output/nerf_rpn/results/${EVAL_SAVE_NAME}}"

cd "${SCRIPT_DIR}"

export PATH="${PROBE_ENV_PREFIX}/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export WANDB_MODE="${WANDB_MODE:-offline}"

train_cmd=(
  python3 -u run_fcos.py
  --mode train
  --resolution "${RESOLUTION}"
  --backbone_type swin_s
  --features_path "${DATA_ROOT}/features"
  --boxes_path "${DATA_ROOT}/obb"
  --num_epochs "${FCOS_NUM_EPOCHS}"
  --lr "${FCOS_LR}"
  --weight_decay "${FCOS_WEIGHT_DECAY}"
  --clip_grad_norm "${CLIP_GRAD_NORM}"
  --log_interval "${FCOS_LOG_INTERVAL}"
  --eval_interval "${FCOS_EVAL_INTERVAL}"
  --keep_checkpoints "${KEEP_CHECKPOINTS}"
  --norm_reg_targets
  --centerness_on_reg
  --center_sampling_radius "${CENTER_SAMPLING_RADIUS}"
  --iou_loss_type "${IOU_LOSS_TYPE}"
  --rotated_bbox
  --log_to_file
  --nms_thresh "${NMS_THRESH}"
  --batch_size "${TRAIN_BATCH_SIZE}"
  --gpus "${GPU_IDS}"
  --percent_train "${PERCENT_TRAIN}"
  --normalize_density
  --tags "${RUN_TAG}"
  --dataset "${DATASET_NAME}"
  --dataset_split "${DATA_ROOT}/${SPLIT_NAME}_split.npz"
  --save_path "${SAVE_PATH}"
)

if [[ "${USE_WANDB}" == "1" ]]; then
  train_cmd+=(--wandb)
fi

"${train_cmd[@]}"

BEST_CHECKPOINT="${SAVE_PATH}/model_best.pt"
[[ -f "${BEST_CHECKPOINT}" ]] || {
  echo "[error] no scratch FCOS checkpoint found at ${BEST_CHECKPOINT}" >&2
  exit 1
}

eval_cmd=(
  python3 -u run_fcos.py
  --mode eval
  --resolution "${RESOLUTION}"
  --backbone_type swin_s
  --features_path "${DATA_ROOT}/features"
  --boxes_path "${DATA_ROOT}/obb"
  --norm_reg_targets
  --centerness_on_reg
  --rotated_bbox
  --output_proposals
  --save_level_index
  --nms_thresh "${NMS_THRESH}"
  --batch_size "${EVAL_BATCH_SIZE}"
  --gpus "${GPU_IDS}"
  --normalize_density
  --dataset "${DATASET_NAME}"
  --dataset_split "${DATA_ROOT}/${SPLIT_NAME}_split.npz"
  --save_path "${EVAL_SAVE_PATH}"
  --checkpoint "${BEST_CHECKPOINT}"
)

"${eval_cmd[@]}"
