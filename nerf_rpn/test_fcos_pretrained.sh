#!/usr/bin/env bash
set -euo pipefail
set -x

dataset_name="${DATASET_NAME:-front3d}"
split_name="${SPLIT_NAME:-3dfront}"
resolution="${RESOLUTION:-160}"
DATA_ROOT="${DATA_ROOT:-../dataset/finetune/${dataset_name}_rpn_data}"
GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NMS_THRESH="${NMS_THRESH:-0.3}"
SAVE_NAME="${SAVE_NAME:-${dataset_name}_finetune}"
SAVE_PATH="${SAVE_PATH:-../output/nerf_rpn/results/${SAVE_NAME}}"
CHECKPOINT="${CHECKPOINT:-../checkpoints/front3d_obb_finetuned.pt}"
SEED="${SEED:-}"
DETERMINISTIC="${DETERMINISTIC:-0}"

cmd=(
python3 -u run_fcos_pretrained.py
--mode eval
--resolution "${resolution}"
--backbone_type swin_s
--features_path "${DATA_ROOT}/features"
--boxes_path "${DATA_ROOT}/obb"
--norm_reg_targets
--normalize_density
--centerness_on_reg
--rotated_bbox
--output_proposals
--save_level_index
--nms_thresh "${NMS_THRESH}"
--batch_size "${BATCH_SIZE}"
--gpus "${GPU_IDS}"
--dataset "${dataset_name}"
--dataset_split "${DATA_ROOT}/${split_name}_split.npz"
--save_path "${SAVE_PATH}"
--checkpoint "${CHECKPOINT}"
)

if [[ -n "${SEED}" ]]; then
  cmd+=(--seed "${SEED}")
fi
if [[ "${DETERMINISTIC}" == "1" ]]; then
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  cmd+=(--deterministic)
fi

"${cmd[@]}"
