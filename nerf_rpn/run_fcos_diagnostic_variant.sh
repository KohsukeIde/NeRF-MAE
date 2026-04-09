#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/nerf_mae/probe_scripts/_probe_common.sh"

VARIANT_NAME="${VARIANT_NAME:?VARIANT_NAME is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"

GPU_IDS="${GPU_IDS:-0}"
DATASET_NAME="${DATASET_NAME:-front3d}"
SPLIT_NAME="${SPLIT_NAME:-3dfront}"
RESOLUTION="${RESOLUTION:-160}"
DATA_ROOT="${DATA_ROOT:-../dataset/finetune/${DATASET_NAME}_rpn_data}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
NMS_THRESH="${NMS_THRESH:-0.3}"
AP_TOP_N="${AP_TOP_N:-}"
FILTER_MODE="${FILTER_MODE:-none}"
FILTER_THRESHOLD="${FILTER_THRESHOLD:-0.7}"
OUTPUT_PROPOSALS="${OUTPUT_PROPOSALS:-1}"
SAVE_LEVEL_INDEX="${SAVE_LEVEL_INDEX:-1}"
OUTPUT_VOXEL_SCORES="${OUTPUT_VOXEL_SCORES:-1}"
SEED="${SEED:-}"
DETERMINISTIC="${DETERMINISTIC:-0}"

SAVE_NAME="${SAVE_NAME:-${VARIANT_NAME}_diagnostics}"
SAVE_PATH="${SAVE_PATH:-../output/nerf_rpn/results/${SAVE_NAME}}"

cd "${SCRIPT_DIR}"

export PATH="${PROBE_ENV_PREFIX}/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export WANDB_MODE="${WANDB_MODE:-offline}"

cmd=(
  python3 -u run_fcos_pretrained.py
  --mode eval
  --resolution "${RESOLUTION}"
  --backbone_type swin_s
  --features_path "${DATA_ROOT}/features"
  --boxes_path "${DATA_ROOT}/obb"
  --norm_reg_targets
  --normalize_density
  --centerness_on_reg
  --rotated_bbox
  --nms_thresh "${NMS_THRESH}"
  --batch_size "${EVAL_BATCH_SIZE}"
  --gpus "${GPU_IDS}"
  --dataset "${DATASET_NAME}"
  --dataset_split "${DATA_ROOT}/${SPLIT_NAME}_split.npz"
  --save_path "${SAVE_PATH}"
  --checkpoint "${CHECKPOINT}"
  --filter "${FILTER_MODE}"
  --filter_threshold "${FILTER_THRESHOLD}"
)

if [[ "${OUTPUT_PROPOSALS}" == "1" ]]; then
  cmd+=(--output_proposals)
fi
if [[ "${SAVE_LEVEL_INDEX}" == "1" ]]; then
  cmd+=(--save_level_index)
fi
if [[ "${OUTPUT_VOXEL_SCORES}" == "1" ]]; then
  cmd+=(--output_voxel_scores)
fi
if [[ -n "${AP_TOP_N}" ]]; then
  cmd+=(--ap_top_n "${AP_TOP_N}")
fi
if [[ -n "${SEED}" ]]; then
  cmd+=(--seed "${SEED}")
fi
if [[ "${DETERMINISTIC}" == "1" ]]; then
  cmd+=(--deterministic)
fi

"${cmd[@]}"
