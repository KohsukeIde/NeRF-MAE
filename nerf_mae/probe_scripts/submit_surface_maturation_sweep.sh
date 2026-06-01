#!/usr/bin/env bash
# Submit Surface-Maturation scouts plus dependent FCOS evaluation jobs.
#
# Run from anywhere inside the repo. Defaults match the current ABCI3 scout
# protocol: one 8-GPU node, global batch 16, deterministic off, coord-jitter on.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

ABCI_GROUP="${ABCI_GROUP:-gag51404}"
PRETRAIN_QUEUE="${PRETRAIN_QUEUE:-rt_HF}"
FCOS_QUEUE="${FCOS_QUEUE:-rt_HG}"
PRETRAIN_WALLTIME="${PRETRAIN_WALLTIME:-72:00:00}"
FCOS_WALLTIME="${FCOS_WALLTIME:-72:00:00}"
PRETRAIN_NODES="${PRETRAIN_NODES:-1}"
PRETRAIN_GPU_IDS="${PRETRAIN_GPU_IDS:-0-7}"
PRETRAIN_BATCH_SIZE_PER_GPU="${PRETRAIN_BATCH_SIZE_PER_GPU:-2}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-3}"
PRETRAIN_WEIGHT_DECAY="${PRETRAIN_WEIGHT_DECAY:-0.0}"
PRETRAIN_EVAL_INTERVAL="${PRETRAIN_EVAL_INTERVAL:-300}"
PRETRAIN_CHECKPOINT_INTERVAL="${PRETRAIN_CHECKPOINT_INTERVAL:-50}"
PRETRAIN_KEEP_CHECKPOINTS="${PRETRAIN_KEEP_CHECKPOINTS:-0}"
PRETRAIN_LOG_INTERVAL="${PRETRAIN_LOG_INTERVAL:-30}"
PRETRAIN_PROFILE_STEP_TIME="${PRETRAIN_PROFILE_STEP_TIME:-0}"
PRETRAIN_TRAIN_NUM_WORKERS="${PRETRAIN_TRAIN_NUM_WORKERS:-}"
PRETRAIN_EVAL_NUM_WORKERS="${PRETRAIN_EVAL_NUM_WORKERS:-}"
PRETRAIN_PERSISTENT_WORKERS="${PRETRAIN_PERSISTENT_WORKERS:-0}"
STAGE_PRETRAIN_DATA="${STAGE_PRETRAIN_DATA:-0}"
PRETRAIN_DATA_ROOT="${PRETRAIN_DATA_ROOT:-${ROOT_DIR}/dataset/pretrain}"
PRETRAIN_DATA_SRC="${PRETRAIN_DATA_SRC:-${PRETRAIN_DATA_ROOT}}"
LOCAL_STAGE_ROOT="${LOCAL_STAGE_ROOT:-}"
STAGE_KEEP="${STAGE_KEEP:-0}"
STAGE_MIN_FREE_GB="${STAGE_MIN_FREE_GB:-80}"
FCOS_DATA_ROOT="${FCOS_DATA_ROOT:-${ROOT_DIR}/dataset/finetune/front3d_rpn_data}"
FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS:-1000}"
FCOS_GPU_IDS="${FCOS_GPU_IDS:-0}"
FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU:-2}"
FCOS_LR="${FCOS_LR:-1e-4}"
FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY:-1e-3}"
FCOS_LR_SCHEDULER="${FCOS_LR_SCHEDULER:-onecycle_epoch}"
ABCI3_CUDA_MODULE="${ABCI3_CUDA_MODULE:-cuda/11.8/11.8.0}"
PROBE_ENV_PREFIX="${PROBE_ENV_PREFIX:-${ROOT_DIR}/.venv-abci3}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
DETERMINISTIC="${DETERMINISTIC:-0}"
PRETRAIN_MASTER_PORT="${PRETRAIN_MASTER_PORT:-29500}"
RUN_SUFFIX="${RUN_SUFFIX:-abci3sm_cj_det0_1n8g}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-300}"
SUBMIT_PRETRAIN="${SUBMIT_PRETRAIN:-1}"
SUBMIT_FCOS="${SUBMIT_FCOS:-1}"
INCLUDE_INPUT_ALPHA="${INCLUDE_INPUT_ALPHA:-0}"
SUBMIT_LOG_DIR="${SUBMIT_LOG_DIR:-${ROOT_DIR}/output/launcher/surface_maturation_sweep_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${SUBMIT_LOG_DIR}"

SURFACE_JOBS=(
  "surface_maturation_tau0p3_k10_w0p05"
  "surface_maturation_tau0p5_k20_w0p05"
  "surface_maturation_tau0p7_k30_w0p05"
)
if [[ "${INCLUDE_INPUT_ALPHA}" == "1" ]]; then
  SURFACE_JOBS+=("input_alpha_curriculum")
fi

short_condition() {
  case "$1" in
    surface_maturation_tau0p3_k10_w0p05) printf "sm03" ;;
    surface_maturation_tau0p5_k20_w0p05) printf "sm05" ;;
    surface_maturation_tau0p7_k30_w0p05) printf "sm07" ;;
    input_alpha_curriculum) printf "inpa" ;;
    *) printf "%s" "$1" | tr -cd '[:alnum:]_' | cut -c1-8 ;;
  esac
}

run_or_print() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    { printf '[dry-run]'; printf ' %q' "$@"; printf '\n'; } >&2
  else
    "$@"
  fi
}

submit_pretrain() {
  local condition="$1"
  local short qsub_gpu_ids varlist cmd_output
  short="$(short_condition "${condition}")"
  qsub_gpu_ids="${PRETRAIN_GPU_IDS//,/:}"
  varlist="ROOT_DIR=${ROOT_DIR},KIND=surface_maturation,CONDITION=${condition},EPOCHS=${EPOCHS},SEED=${SEED},RUN_SUFFIX=${RUN_SUFFIX},PROBE_ENV_PREFIX=${PROBE_ENV_PREFIX},PRETRAIN_DATA_ROOT=${PRETRAIN_DATA_ROOT},PRETRAIN_DATA_SRC=${PRETRAIN_DATA_SRC},FCOS_DATA_ROOT=${FCOS_DATA_ROOT},ABCI3_CUDA_MODULE=${ABCI3_CUDA_MODULE},PRETRAIN_NODES=${PRETRAIN_NODES},PRETRAIN_GPU_IDS=${qsub_gpu_ids},PRETRAIN_BATCH_SIZE_PER_GPU=${PRETRAIN_BATCH_SIZE_PER_GPU},PRETRAIN_LR=${PRETRAIN_LR},PRETRAIN_WEIGHT_DECAY=${PRETRAIN_WEIGHT_DECAY},PRETRAIN_EVAL_INTERVAL=${PRETRAIN_EVAL_INTERVAL},PRETRAIN_CHECKPOINT_INTERVAL=${PRETRAIN_CHECKPOINT_INTERVAL},PRETRAIN_KEEP_CHECKPOINTS=${PRETRAIN_KEEP_CHECKPOINTS},PRETRAIN_LOG_INTERVAL=${PRETRAIN_LOG_INTERVAL},PRETRAIN_PROFILE_STEP_TIME=${PRETRAIN_PROFILE_STEP_TIME},PRETRAIN_TRAIN_NUM_WORKERS=${PRETRAIN_TRAIN_NUM_WORKERS},PRETRAIN_EVAL_NUM_WORKERS=${PRETRAIN_EVAL_NUM_WORKERS},PRETRAIN_PERSISTENT_WORKERS=${PRETRAIN_PERSISTENT_WORKERS},STAGE_PRETRAIN_DATA=${STAGE_PRETRAIN_DATA},PRETRAIN_DATA_SRC=${PRETRAIN_DATA_SRC},LOCAL_STAGE_ROOT=${LOCAL_STAGE_ROOT},STAGE_KEEP=${STAGE_KEEP},STAGE_MIN_FREE_GB=${STAGE_MIN_FREE_GB},USE_WANDB=${USE_WANDB},WANDB_MODE=${WANDB_MODE},DETERMINISTIC=${DETERMINISTIC},PRETRAIN_MASTER_PORT=${PRETRAIN_MASTER_PORT},SKIP_EXISTING=1"
  local cmd=(
    qsub
    -P "${ABCI_GROUP}"
    -q "${PRETRAIN_QUEUE}"
    -l "select=${PRETRAIN_NODES}"
    -l "walltime=${PRETRAIN_WALLTIME}"
    -N "e${EPOCHS}_${short}_pre"
    -j oe
    -o "${SUBMIT_LOG_DIR}/e${EPOCHS}_${short}_pre.pbs.log"
    -v "${varlist}"
    "${SCRIPT_DIR}/abci3_e300_gate_pretrain.pbs"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    run_or_print "${cmd[@]}"
    printf "DRYRUN_PRE_%s\n" "${short}"
  else
    cmd_output="$("${cmd[@]}")"
    cmd_output="${cmd_output%% *}"
    echo "[submitted] pretrain condition=${condition} job=${cmd_output}" >&2
    printf "%s\n" "${cmd_output}"
  fi
}

submit_fcos() {
  local condition="$1"
  local dependency="$2"
  local short varlist cmd_output
  short="$(short_condition "${condition}")"
  varlist="ROOT_DIR=${ROOT_DIR},KIND=surface_maturation,CONDITION=${condition},EPOCHS=${EPOCHS},SEED=${SEED},FINETUNE_SEED=${SEED},RUN_SUFFIX=${RUN_SUFFIX},PROBE_ENV_PREFIX=${PROBE_ENV_PREFIX},FCOS_DATA_ROOT=${FCOS_DATA_ROOT},ABCI3_CUDA_MODULE=${ABCI3_CUDA_MODULE},FCOS_GPU_IDS=${FCOS_GPU_IDS},FCOS_NUM_EPOCHS=${FCOS_NUM_EPOCHS},FCOS_BATCH_SIZE_PER_GPU=${FCOS_BATCH_SIZE_PER_GPU},FCOS_LR=${FCOS_LR},FCOS_WEIGHT_DECAY=${FCOS_WEIGHT_DECAY},FCOS_LR_SCHEDULER=${FCOS_LR_SCHEDULER},USE_WANDB=${USE_WANDB},WANDB_MODE=${WANDB_MODE},DETERMINISTIC=${DETERMINISTIC},SKIP_EXISTING=1"
  local cmd=(
    qsub
    -P "${ABCI_GROUP}"
    -q "${FCOS_QUEUE}"
    -l select=1
    -l "walltime=${FCOS_WALLTIME}"
    -N "e${EPOCHS}_${short}_fcos"
    -j oe
    -o "${SUBMIT_LOG_DIR}/e${EPOCHS}_${short}_fcos.pbs.log"
    -v "${varlist}"
  )
  if [[ -n "${dependency}" ]]; then
    cmd+=(-W "depend=afterok:${dependency}")
  fi
  cmd+=("${SCRIPT_DIR}/abci3_e300_gate_fcos.pbs")
  if [[ "${DRY_RUN}" == "1" ]]; then
    run_or_print "${cmd[@]}"
  else
    cmd_output="$("${cmd[@]}")"
    cmd_output="${cmd_output%% *}"
    echo "[submitted] fcos condition=${condition} job=${cmd_output} dep=${dependency:-none}" >&2
  fi
}

for condition in "${SURFACE_JOBS[@]}"; do
  pre_job=""
  if [[ "${SUBMIT_PRETRAIN}" == "1" ]]; then
    pre_job="$(submit_pretrain "${condition}")"
  fi
  if [[ "${SUBMIT_FCOS}" == "1" ]]; then
    submit_fcos "${condition}" "${pre_job}"
  fi
done

echo "[info] submission log dir=${SUBMIT_LOG_DIR}"
