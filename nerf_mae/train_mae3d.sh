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
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-0}"
PROFILE_STEP_TIME="${PROFILE_STEP_TIME:-0}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"
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
PROBE_CURRICULUM="${PROBE_CURRICULUM:-}"
PROBE_CURRICULUM_EPOCHS="${PROBE_CURRICULUM_EPOCHS:-}"
PROBE_CURRICULUM_RGB_START_WEIGHT="${PROBE_CURRICULUM_RGB_START_WEIGHT:-}"
PROBE_CURRICULUM_RGB_END_WEIGHT="${PROBE_CURRICULUM_RGB_END_WEIGHT:-}"
PROBE_CURRICULUM_ALPHA_WEIGHT="${PROBE_CURRICULUM_ALPHA_WEIGHT:-}"
PROBE_DECOMP_MODE="${PROBE_DECOMP_MODE:-}"
DISABLE_ABS_POS_EMBED="${DISABLE_ABS_POS_EMBED:-0}"
DISABLE_RELATIVE_POSITION_BIAS="${DISABLE_RELATIVE_POSITION_BIAS:-0}"
ROTATE_PROB="${ROTATE_PROB:-}"
FLIP_PROB="${FLIP_PROB:-}"
ROT_SCALE_PROB="${ROT_SCALE_PROB:-}"
COORD_SHIFT_PROB="${COORD_SHIFT_PROB:-}"
COORD_SHIFT_MAX_VOXELS="${COORD_SHIFT_MAX_VOXELS:-}"
SEED="${SEED:-}"
DETERMINISTIC="${DETERMINISTIC:-0}"

normalize_empty_literal() {
  local name="$1"
  local value="${!name:-}"
  if [[ "${value}" == "''" || "${value}" == '""' ]]; then
    printf -v "${name}" '%s' ""
  fi
}

for optional_arg in ROTATE_PROB FLIP_PROB ROT_SCALE_PROB COORD_SHIFT_PROB COORD_SHIFT_MAX_VOXELS; do
  normalize_empty_literal "${optional_arg}"
done

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
CHECKPOINT="${CHECKPOINT:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
RESUME_ALLOW_PARTIAL="${RESUME_ALLOW_PARTIAL:-0}"
RESUME_START_EPOCH="${RESUME_START_EPOCH:-}"

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
  --checkpoint_interval "${CHECKPOINT_INTERVAL}"
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

if [[ "${PROFILE_STEP_TIME}" == "1" ]]; then
  cmd+=(--profile_step_time)
fi
if [[ -n "${TRAIN_NUM_WORKERS}" ]]; then
  cmd+=(--train_num_workers "${TRAIN_NUM_WORKERS}")
fi
if [[ -n "${EVAL_NUM_WORKERS}" ]]; then
  cmd+=(--eval_num_workers "${EVAL_NUM_WORKERS}")
fi
if [[ "${PERSISTENT_WORKERS}" == "1" ]]; then
  cmd+=(--persistent_workers)
fi
if [[ -n "${CHECKPOINT}" ]]; then
  cmd+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  cmd+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi
if [[ "${RESUME_ALLOW_PARTIAL}" == "1" ]]; then
  cmd+=(--resume_allow_partial)
fi
if [[ -n "${RESUME_START_EPOCH}" ]]; then
  cmd+=(--resume_start_epoch "${RESUME_START_EPOCH}")
fi
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
if [[ -n "${PROBE_CURRICULUM}" ]]; then
  cmd+=(--probe_curriculum "${PROBE_CURRICULUM}")
fi
if [[ -n "${PROBE_CURRICULUM_EPOCHS}" ]]; then
  cmd+=(--probe_curriculum_epochs "${PROBE_CURRICULUM_EPOCHS}")
fi
if [[ -n "${PROBE_CURRICULUM_RGB_START_WEIGHT}" ]]; then
  cmd+=(--probe_curriculum_rgb_start_weight "${PROBE_CURRICULUM_RGB_START_WEIGHT}")
fi
if [[ -n "${PROBE_CURRICULUM_RGB_END_WEIGHT}" ]]; then
  cmd+=(--probe_curriculum_rgb_end_weight "${PROBE_CURRICULUM_RGB_END_WEIGHT}")
fi
if [[ -n "${PROBE_CURRICULUM_ALPHA_WEIGHT}" ]]; then
  cmd+=(--probe_curriculum_alpha_weight "${PROBE_CURRICULUM_ALPHA_WEIGHT}")
fi
if [[ -n "${PROBE_DECOMP_MODE}" ]]; then
  cmd+=(--probe_decomp_mode "${PROBE_DECOMP_MODE}")
fi
if [[ "${DISABLE_ABS_POS_EMBED}" == "1" ]]; then
  cmd+=(--disable_abs_pos_embed)
fi
if [[ "${DISABLE_RELATIVE_POSITION_BIAS}" == "1" ]]; then
  cmd+=(--disable_relative_position_bias)
fi
if [[ -n "${ROTATE_PROB}" ]]; then
  cmd+=(--rotate_prob "${ROTATE_PROB}")
fi
if [[ -n "${FLIP_PROB}" ]]; then
  cmd+=(--flip_prob "${FLIP_PROB}")
fi
if [[ -n "${ROT_SCALE_PROB}" ]]; then
  cmd+=(--rot_scale_prob "${ROT_SCALE_PROB}")
fi
if [[ -n "${COORD_SHIFT_PROB}" ]]; then
  cmd+=(--coord_shift_prob "${COORD_SHIFT_PROB}")
fi
if [[ -n "${COORD_SHIFT_MAX_VOXELS}" ]]; then
  cmd+=(--coord_shift_max_voxels "${COORD_SHIFT_MAX_VOXELS}")
fi
if [[ -n "${SEED}" ]]; then
  cmd+=(--seed "${SEED}")
fi
if [[ "${DETERMINISTIC}" == "1" ]]; then
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  cmd+=(--deterministic)
fi

"${cmd[@]}"
