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
ROTATE_PROB="${ROTATE_PROB:-0.5}"
FLIP_PROB="${FLIP_PROB:-0.5}"
ROT_SCALE_PROB="${ROT_SCALE_PROB:-0.5}"
LR_SCHEDULER="${LR_SCHEDULER:-}"
SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-}"
SCHEDULER_MIN_LR="${SCHEDULER_MIN_LR:-}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-}"
BACKBONE_LR_SCALE="${BACKBONE_LR_SCALE:-}"

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
  --rotate_prob "${ROTATE_PROB}"
  --flip_prob "${FLIP_PROB}"
  --rot_scale_prob "${ROT_SCALE_PROB}"
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
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  cmd+=(--deterministic)
fi
if [[ -n "${LR_SCHEDULER}" ]]; then
  cmd+=(--lr_scheduler "${LR_SCHEDULER}")
fi
if [[ -n "${SCHEDULER_TOTAL_STEPS}" ]]; then
  cmd+=(--scheduler_total_steps "${SCHEDULER_TOTAL_STEPS}")
fi
if [[ -n "${SCHEDULER_MIN_LR}" ]]; then
  cmd+=(--scheduler_min_lr "${SCHEDULER_MIN_LR}")
fi
if [[ -n "${FREEZE_BACKBONE_EPOCHS}" ]]; then
  cmd+=(--freeze_backbone_epochs "${FREEZE_BACKBONE_EPOCHS}")
fi
if [[ -n "${BACKBONE_LR_SCALE}" ]]; then
  cmd+=(--backbone_lr_scale "${BACKBONE_LR_SCALE}")
fi

"${cmd[@]}"
