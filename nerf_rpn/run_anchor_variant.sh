#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/nerf_mae/probe_scripts/_probe_common.sh"

MODE="${MODE:?MODE is required: probe or scratch}"
GPU_IDS="${GPU_IDS:-0}"
DATASET_NAME="${DATASET_NAME:-front3d}"
SPLIT_NAME="${SPLIT_NAME:-3dfront}"
DATA_ROOT="${DATA_ROOT:-../dataset/finetune/${DATASET_NAME}_rpn_data}"
PERCENT_TRAIN="${PERCENT_TRAIN:-1.0}"
ANCHOR_NUM_EPOCHS="${ANCHOR_NUM_EPOCHS:-200}"
ANCHOR_LR="${ANCHOR_LR:-3e-4}"
ANCHOR_WEIGHT_DECAY="${ANCHOR_WEIGHT_DECAY:-1e-3}"
ANCHOR_BATCH_SIZE="${ANCHOR_BATCH_SIZE:-2}"
ANCHOR_LOG_INTERVAL="${ANCHOR_LOG_INTERVAL:-10}"
ANCHOR_EVAL_INTERVAL="${ANCHOR_EVAL_INTERVAL:-10}"
ANCHOR_KEEP_CHECKPOINTS="${ANCHOR_KEEP_CHECKPOINTS:-1}"
ANCHOR_NMS_THRESH="${ANCHOR_NMS_THRESH:-0.3}"
ANCHOR_ROTATE_PROB="${ANCHOR_ROTATE_PROB:-0.5}"
ANCHOR_FLIP_PROB="${ANCHOR_FLIP_PROB:-0.5}"
ANCHOR_ROT_SCALE_PROB="${ANCHOR_ROT_SCALE_PROB:-0.5}"
USE_WANDB="${USE_WANDB:-0}"
SEED="${SEED:-}"
DETERMINISTIC="${DETERMINISTIC:-0}"
NORMALIZE_DENSITY="${NORMALIZE_DENSITY:-1}"
ANCHOR_MAE_BACKBONE_ARCH="${ANCHOR_MAE_BACKBONE_ARCH:-0}"
ANCHOR_BEST_CHECKPOINT_NAME="${ANCHOR_BEST_CHECKPOINT_NAME:-model_best_ap50.pt}"
ANCHOR_EVAL_ON_TRAIN="${ANCHOR_EVAL_ON_TRAIN:-0}"

PRETRAIN_SAVE_NAME="${PRETRAIN_SAVE_NAME:-}"
VARIANT_NAME="${VARIANT_NAME:-${MODE}}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-}"
SAVE_NAME="${SAVE_NAME:-${DATASET_NAME}_anchor_${VARIANT_NAME}_p${PERCENT_TRAIN}_seed${SEED:-none}}"
SAVE_PATH="${SAVE_PATH:-../output/nerf_rpn/results/${SAVE_NAME}}"
EVAL_SAVE_NAME="${EVAL_SAVE_NAME:-${SAVE_NAME}_eval}"
EVAL_SAVE_PATH="${EVAL_SAVE_PATH:-../output/nerf_rpn/results/${EVAL_SAVE_NAME}}"

if [[ "${PRETRAIN_CHECKPOINT}" != /* && -n "${PRETRAIN_CHECKPOINT}" ]]; then
  if [[ -f "${PRETRAIN_CHECKPOINT}" ]]; then
    PRETRAIN_CHECKPOINT="$(realpath "${PRETRAIN_CHECKPOINT}")"
  elif [[ -f "${ROOT_DIR}/${PRETRAIN_CHECKPOINT}" ]]; then
    PRETRAIN_CHECKPOINT="$(realpath "${ROOT_DIR}/${PRETRAIN_CHECKPOINT}")"
  elif [[ -f "${SCRIPT_DIR}/${PRETRAIN_CHECKPOINT}" ]]; then
    PRETRAIN_CHECKPOINT="$(realpath "${SCRIPT_DIR}/${PRETRAIN_CHECKPOINT}")"
  fi
fi

if [[ "${MODE}" == "probe" ]]; then
  [[ -n "${PRETRAIN_CHECKPOINT}" ]] || {
    echo "[error] PRETRAIN_CHECKPOINT is required for MODE=probe" >&2
    exit 1
  }
  [[ -f "${PRETRAIN_CHECKPOINT}" ]] || {
    echo "[error] missing PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT}" >&2
    exit 1
  }
elif [[ "${MODE}" != "scratch" ]]; then
  echo "[error] unsupported MODE=${MODE}; expected probe or scratch" >&2
  exit 1
fi

if [[ "${MODE}" == "probe" && "${ANCHOR_MAE_BACKBONE_ARCH}" != "1" ]]; then
  echo "[warn] MODE=probe requires the MAE-compatible backbone for train/eval; forcing ANCHOR_MAE_BACKBONE_ARCH=1" >&2
  ANCHOR_MAE_BACKBONE_ARCH=1
fi

cd "${SCRIPT_DIR}"

export PATH="${PROBE_ENV_PREFIX}/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}"
export WANDB_MODE="${WANDB_MODE:-offline}"

train_cmd=(
  python3 -u run_rpn.py
  --mode train
  --dataset_name "${DATASET_NAME}"
  --resolution 160
  --backbone_type swin_s
  --features_path "${DATA_ROOT}/features"
  --boxes_path "${DATA_ROOT}/obb"
  --dataset_split "${DATA_ROOT}/${SPLIT_NAME}_split.npz"
  --save_path "${SAVE_PATH}"
  --num_epochs "${ANCHOR_NUM_EPOCHS}"
  --lr "${ANCHOR_LR}"
  --weight_decay "${ANCHOR_WEIGHT_DECAY}"
  --log_interval "${ANCHOR_LOG_INTERVAL}"
  --eval_interval "${ANCHOR_EVAL_INTERVAL}"
  --keep_checkpoints "${ANCHOR_KEEP_CHECKPOINTS}"
  --rpn_nms_thresh "${ANCHOR_NMS_THRESH}"
  --rotate_prob "${ANCHOR_ROTATE_PROB}"
  --flip_prob "${ANCHOR_FLIP_PROB}"
  --rot_scale_prob "${ANCHOR_ROT_SCALE_PROB}"
  --log_to_file
  --rotated_bbox
  --batch_size "${ANCHOR_BATCH_SIZE}"
  --gpus "${GPU_IDS}"
  --percent_train "${PERCENT_TRAIN}"
)

if [[ "${NORMALIZE_DENSITY}" == "1" ]]; then
  train_cmd+=(--normalize_density)
fi
if [[ "${USE_WANDB}" == "1" ]]; then
  train_cmd+=(--wandb)
fi
if [[ -n "${SEED}" ]]; then
  train_cmd+=(--seed "${SEED}")
fi
if [[ "${DETERMINISTIC}" == "1" ]]; then
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  train_cmd+=(--deterministic)
fi
if [[ "${MODE}" == "probe" ]]; then
  train_cmd+=(--mae_checkpoint "${PRETRAIN_CHECKPOINT}")
else
  train_cmd+=(--scratch_backbone)
fi
if [[ "${ANCHOR_MAE_BACKBONE_ARCH}" == "1" ]]; then
  train_cmd+=(--mae_backbone_arch)
fi
if [[ "${ANCHOR_EVAL_ON_TRAIN}" == "1" ]]; then
  train_cmd+=(--eval_on_train)
fi

"${train_cmd[@]}"

BEST_CHECKPOINT="${SAVE_PATH}/${ANCHOR_BEST_CHECKPOINT_NAME}"
if [[ ! -f "${BEST_CHECKPOINT}" && "${ANCHOR_BEST_CHECKPOINT_NAME}" != "model_best.pt" ]]; then
  echo "[warn] no ${ANCHOR_BEST_CHECKPOINT_NAME}; falling back to model_best.pt" >&2
  BEST_CHECKPOINT="${SAVE_PATH}/model_best.pt"
fi
[[ -f "${BEST_CHECKPOINT}" ]] || {
  echo "[error] no anchor checkpoint found: ${BEST_CHECKPOINT}" >&2
  exit 1
}

eval_cmd=(
  python3 -u run_rpn.py
  --mode eval
  --dataset_name "${DATASET_NAME}"
  --resolution 160
  --backbone_type swin_s
  --features_path "${DATA_ROOT}/features"
  --boxes_path "${DATA_ROOT}/obb"
  --dataset_split "${DATA_ROOT}/${SPLIT_NAME}_split.npz"
  --save_path "${EVAL_SAVE_PATH}"
  --checkpoint "${BEST_CHECKPOINT}"
  --rpn_nms_thresh "${ANCHOR_NMS_THRESH}"
  --rotated_bbox
  --batch_size "${ANCHOR_BATCH_SIZE}"
  --gpus "${GPU_IDS}"
)

if [[ "${NORMALIZE_DENSITY}" == "1" ]]; then
  eval_cmd+=(--normalize_density)
fi
if [[ -n "${SEED}" ]]; then
  eval_cmd+=(--seed "${SEED}")
fi
if [[ "${DETERMINISTIC}" == "1" ]]; then
  eval_cmd+=(--deterministic)
fi
if [[ "${MODE}" == "probe" ]]; then
  eval_cmd+=(--mae_checkpoint "${PRETRAIN_CHECKPOINT}")
else
  eval_cmd+=(--scratch_backbone)
fi
if [[ "${ANCHOR_MAE_BACKBONE_ARCH}" == "1" ]]; then
  eval_cmd+=(--mae_backbone_arch)
fi

"${eval_cmd[@]}"

test -f "${EVAL_SAVE_PATH}/eval.json"
echo "[info] Anchor-RPN ${MODE} complete eval=${EVAL_SAVE_PATH}/eval.json"
