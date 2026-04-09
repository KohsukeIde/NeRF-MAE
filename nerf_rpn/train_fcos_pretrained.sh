#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../nerf_mae/probe_scripts/_probe_common.sh"

dataset_name="${DATASET_NAME:-front3d}"
split_name="${SPLIT_NAME:-3dfront}"
resolution="${RESOLUTION:-}"
DATA_ROOT="${DATA_ROOT:-../dataset/finetune/${dataset_name}_rpn_data}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_EPOCHS="${NUM_EPOCHS:-1000}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
PERCENT_TRAIN="${PERCENT_TRAIN:-1.0}"
CLIP_GRAD_NORM="${CLIP_GRAD_NORM:-0.1}"
NMS_THRESH="${NMS_THRESH:-0.3}"
CENTER_SAMPLING_RADIUS="${CENTER_SAMPLING_RADIUS:-1.5}"
IOU_LOSS_TYPE="${IOU_LOSS_TYPE:-iou}"
USE_WANDB="${USE_WANDB:-1}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-2}"
SCRATCH_BACKBONE="${SCRATCH_BACKBONE:-0}"
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
SAVE_NAME="${SAVE_NAME:-nerfmae_all}"
RUN_TAG="${RUN_TAG:-${dataset_name}_finetune}"
SAVE_PATH="${SAVE_PATH:-../output/nerf_rpn/results/${SAVE_NAME}}"
MAE_CHECKPOINT="${MAE_CHECKPOINT:-../checkpoints/nerf_mae_pretrained.pt}"

cmd=(
  python3 -u run_fcos_pretrained.py
  --mode train
  --resolution "${resolution}"
  --backbone_type swin_s
  --features_path "${DATA_ROOT}/features"
  --boxes_path "${DATA_ROOT}/obb"
  --num_epochs "${NUM_EPOCHS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --clip_grad_norm "${CLIP_GRAD_NORM}"
  --log_interval "${LOG_INTERVAL}"
  --eval_interval "${EVAL_INTERVAL}"
  --norm_reg_targets
  --centerness_on_reg
  --center_sampling_radius "${CENTER_SAMPLING_RADIUS}"
  --iou_loss_type "${IOU_LOSS_TYPE}"
  --rotated_bbox
  --log_to_file
  --nms_thresh "${NMS_THRESH}"
  --batch_size "${BATCH_SIZE}"
  --gpus "${GPU_IDS}"
  --percent_train "${PERCENT_TRAIN}"
  --normalize_density
  --tags "${RUN_TAG}"
  --dataset "${dataset_name}"
  --dataset_split "${DATA_ROOT}/${split_name}_split.npz"
  --save_path "${SAVE_PATH}"
  --mae_checkpoint "${MAE_CHECKPOINT}"
)

if [[ "${USE_WANDB}" == "1" ]]; then
  cmd+=(--wandb)
fi
if [[ "${SCRATCH_BACKBONE}" == "1" ]]; then
  cmd+=(--scratch_backbone)
fi
if [[ -n "${SEED}" ]]; then
  cmd+=(--seed "${SEED}")
fi
if [[ "${DETERMINISTIC}" == "1" ]]; then
  cmd+=(--deterministic)
fi

"${cmd[@]}"
