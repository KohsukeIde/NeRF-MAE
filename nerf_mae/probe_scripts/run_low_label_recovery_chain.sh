#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_probe_common.sh"

CHAIN_NAME="${CHAIN_NAME:-nerfmae_low_label_recovery}"
LOG_ROOT="${LOG_ROOT:-${PROBE_LOG_ROOT_DEFAULT}}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${CHAIN_NAME}.chain.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${CHAIN_NAME}.chain.pid}"
TMUX_SESSION="${TMUX_SESSION:-${CHAIN_NAME//[^[:alnum:]_]/_}_chain}"

WAIT_BEFORE_CHAIN="${WAIT_BEFORE_CHAIN:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
DETERMINISTIC="${DETERMINISTIC:-1}"

FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS:-100}"
FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU:-2}"
FCOS_LR="${FCOS_LR:-1e-4}"
FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY:-1e-3}"
FCOS_LR_SCHEDULER="${FCOS_LR_SCHEDULER:-onecycle_epoch}"
LOW_LABEL_SEED="${LOW_LABEL_SEED:-1}"

GPU_SCRATCH_PT01="${GPU_SCRATCH_PT01:-0}"
GPU_SCRATCH_PT02="${GPU_SCRATCH_PT02:-1}"
GPU_BASELINE_PT02="${GPU_BASELINE_PT02:-2}"
GPU_ALPHA_TARGET_PT02="${GPU_ALPHA_TARGET_PT02:-3}"

mkdir -p "${LOG_ROOT}"

if [[ "${FOREGROUND:-0}" != "1" ]]; then
  if env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
    echo "[info] chain already running in tmux session=${TMUX_SESSION}"
    echo "[info] log=${LOG_FILE}"
    exit 0
  fi
  env -u LD_LIBRARY_PATH -u PYTHONPATH tmux new-session -d -s "${TMUX_SESSION}" \
    "cd '${ROOT_DIR}' && FOREGROUND=1 WAIT_BEFORE_CHAIN='${WAIT_BEFORE_CHAIN}' SKIP_EXISTING='${SKIP_EXISTING}' USE_WANDB='${USE_WANDB}' WANDB_MODE='${WANDB_MODE}' DETERMINISTIC='${DETERMINISTIC}' FCOS_NUM_EPOCHS='${FCOS_NUM_EPOCHS}' FCOS_BATCH_SIZE_PER_GPU='${FCOS_BATCH_SIZE_PER_GPU}' FCOS_LR='${FCOS_LR}' FCOS_WEIGHT_DECAY='${FCOS_WEIGHT_DECAY}' FCOS_LR_SCHEDULER='${FCOS_LR_SCHEDULER}' LOW_LABEL_SEED='${LOW_LABEL_SEED}' GPU_SCRATCH_PT01='${GPU_SCRATCH_PT01}' GPU_SCRATCH_PT02='${GPU_SCRATCH_PT02}' GPU_BASELINE_PT02='${GPU_BASELINE_PT02}' GPU_ALPHA_TARGET_PT02='${GPU_ALPHA_TARGET_PT02}' bash '${SCRIPT_DIR}/run_low_label_recovery_chain.sh'"
  echo "[info] started detached chain"
  echo "[info] session=${TMUX_SESSION}"
  echo "[info] log=${LOG_FILE}"
  exit 0
fi

touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "$$" > "${PID_FILE}"

cleanup() {
  local rc=$?
  probe_log "exit_code=${rc}"
  rm -f "${PID_FILE}"
  exit "${rc}"
}
trap cleanup EXIT

log() {
  probe_log "$*"
}

eval_exists() {
  local save_name="$1"
  [[ -f "${ROOT_DIR}/output/nerf_rpn/results/${save_name}_eval/eval.json" ]]
}

run_job_scratch() {
  local gpu_id="$1"
  local save_name="$2"
  local percent_train="$3"
  local log_file="$4"
  (
    cd "${ROOT_DIR}"
    GPU_IDS="${gpu_id}" \
    DATASET_NAME=front3d \
    SPLIT_NAME=3dfront \
    PERCENT_TRAIN="${percent_train}" \
    FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS}" \
    FCOS_LR="${FCOS_LR}" \
    FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY}" \
    FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU}" \
    USE_WANDB="${USE_WANDB}" \
    WANDB_MODE="${WANDB_MODE}" \
    SAVE_NAME="${save_name}" \
    RUN_TAG="${save_name}" \
    SEED="${LOW_LABEL_SEED}" \
    DETERMINISTIC="${DETERMINISTIC}" \
    LR_SCHEDULER="${FCOS_LR_SCHEDULER}" \
    bash nerf_rpn/run_fcos_fair_scratch_variant.sh
  ) >> "${log_file}" 2>&1 &
  RUN_VARIANT_PID=$!
}

run_job_probe() {
  local variant="$1"
  local gpu_id="$2"
  local pretrain_save="$3"
  local save_name="$4"
  local percent_train="$5"
  local log_file="$6"
  (
    cd "${ROOT_DIR}"
    VARIANT_NAME="${variant}" \
    PRETRAIN_SAVE_NAME="${pretrain_save}" \
    PRETRAIN_CHECKPOINT="../output/nerf_mae/results/${pretrain_save}/epoch_100.pt" \
    GPU_IDS="${gpu_id}" \
    DATASET_NAME=front3d \
    SPLIT_NAME=3dfront \
    PERCENT_TRAIN="${percent_train}" \
    FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS}" \
    FCOS_LR="${FCOS_LR}" \
    FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY}" \
    FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU}" \
    USE_WANDB="${USE_WANDB}" \
    WANDB_MODE="${WANDB_MODE}" \
    SAVE_NAME="${save_name}" \
    RUN_TAG="${save_name}" \
    SEED="${LOW_LABEL_SEED}" \
    DETERMINISTIC="${DETERMINISTIC}" \
    LR_SCHEDULER="${FCOS_LR_SCHEDULER}" \
    bash nerf_rpn/run_fcos_probe_variant.sh
  ) >> "${log_file}" 2>&1 &
  RUN_VARIANT_PID=$!
}

wait_for_pids() {
  local -n pids_ref=$1
  local -n names_ref=$2
  local failed=0
  local i
  for i in "${!pids_ref[@]}"; do
    if wait "${pids_ref[$i]}"; then
      log "job_done name=${names_ref[$i]}"
    else
      log "job_failed name=${names_ref[$i]}"
      failed=1
    fi
  done
  (( failed == 0 )) || probe_die "one or more recovery jobs failed"
}

jobs=(
  "scratch:${GPU_SCRATCH_PT01}:front3d_scratch_samepath_sched_epoch_pt01_seed${LOW_LABEL_SEED}_fcos${FCOS_NUM_EPOCHS}:0.1"
  "scratch:${GPU_SCRATCH_PT02}:front3d_scratch_samepath_sched_epoch_pt02_seed${LOW_LABEL_SEED}_fcos${FCOS_NUM_EPOCHS}:0.2"
  "baseline:${GPU_BASELINE_PT02}:nerfmae_all_p0.1_e100_seed1:nerfmae_all_p0.1_e100_seed1_epoch100_sched_epoch_pt02_seed${LOW_LABEL_SEED}_fcos${FCOS_NUM_EPOCHS}:0.2"
  "alpha_target_only:${GPU_ALPHA_TARGET_PT02}:nerfmae_alpha_target_only_p0.1_e100_seed1:nerfmae_alpha_target_only_p0.1_e100_seed1_epoch100_sched_epoch_pt02_seed${LOW_LABEL_SEED}_fcos${FCOS_NUM_EPOCHS}:0.2"
)

if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
  GPU_IDS="${GPU_SCRATCH_PT01},${GPU_SCRATCH_PT02},${GPU_BASELINE_PT02},${GPU_ALPHA_TARGET_PT02}" probe_wait_for_runway
fi

log "start low-label recovery"
pids=()
names=()
for job in "${jobs[@]}"; do
  IFS=':' read -r variant gpu_id a b c <<< "${job}"
  if [[ "${variant}" == "scratch" ]]; then
    save_name="${a}"
    percent_train="${b}"
    if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${save_name}"; then
      log "skip existing scratch save=${save_name}"
      continue
    fi
    run_job_scratch \
      "${gpu_id}" \
      "${save_name}" \
      "${percent_train}" \
      "${LOG_ROOT}/${CHAIN_NAME}.${save_name}.log"
    pids+=("${RUN_VARIANT_PID}")
    names+=("${save_name}")
  else
    pretrain_save="${a}"
    save_name="${b}"
    percent_train="${c}"
    if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${save_name}"; then
      log "skip existing variant=${variant} save=${save_name}"
      continue
    fi
    run_job_probe \
      "${variant}" \
      "${gpu_id}" \
      "${pretrain_save}" \
      "${save_name}" \
      "${percent_train}" \
      "${LOG_ROOT}/${CHAIN_NAME}.${save_name}.log"
    pids+=("${RUN_VARIANT_PID}")
    names+=("${save_name}")
  fi
done

if (( ${#pids[@]} > 0 )); then
  wait_for_pids pids names
fi
log "low-label recovery done"
