#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_probe_common.sh"

CHAIN_NAME="${CHAIN_NAME:-nerfmae_epoch30_followup}"
LOG_ROOT="${LOG_ROOT:-${PROBE_LOG_ROOT_DEFAULT}}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${CHAIN_NAME}.chain.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${CHAIN_NAME}.chain.pid}"
TMUX_SESSION="${TMUX_SESSION:-${CHAIN_NAME//[^[:alnum:]_]/_}_chain}"

STEPA_GPU_IDS="${STEPA_GPU_IDS:-0,1,2}"
STEPA_BASELINE_GPU="${STEPA_BASELINE_GPU:-0}"
STEPA_ALPHA_GPU="${STEPA_ALPHA_GPU:-1}"
STEPA_RADIANCE_GPU="${STEPA_RADIANCE_GPU:-2}"
STEPB_PRETRAIN_GPU_IDS="${STEPB_PRETRAIN_GPU_IDS:-0,1,2,3}"
STEPB_FCOS_GPU_IDS="${STEPB_FCOS_GPU_IDS:-0}"

FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS:-100}"
FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU:-2}"
FCOS_LR="${FCOS_LR:-1e-4}"
FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY:-1e-3}"
FCOS_PERCENT_TRAIN="${FCOS_PERCENT_TRAIN:-1.0}"

MASKED_PRETRAIN_EPOCHS="${MASKED_PRETRAIN_EPOCHS:-30}"
MASKED_PRETRAIN_PERCENT_TRAIN="${MASKED_PRETRAIN_PERCENT_TRAIN:-0.1}"
MASKED_PRETRAIN_BATCH_SIZE_PER_GPU="${MASKED_PRETRAIN_BATCH_SIZE_PER_GPU:-4}"

WAIT_BEFORE_CHAIN="${WAIT_BEFORE_CHAIN:-1}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-15}"
WAIT_STABLE_SECONDS="${WAIT_STABLE_SECONDS:-60}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "${LOG_ROOT}"

if [[ "${FOREGROUND:-0}" != "1" ]]; then
  if env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
    echo "[info] chain already running in tmux session=${TMUX_SESSION}"
    echo "[info] log=${LOG_FILE}"
    exit 0
  fi
  env -u LD_LIBRARY_PATH -u PYTHONPATH tmux new-session -d -s "${TMUX_SESSION}" \
    "cd '${ROOT_DIR}' && FOREGROUND=1 bash '${SCRIPT_DIR}/run_epoch30_followup_chain.sh'"
  echo "[info] started detached follow-up chain"
  echo "[info] session=${TMUX_SESSION}"
  echo "[info] log=${LOG_FILE}"
  exit 0
fi

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

run_transfer_job() {
  local variant="$1"
  local gpu_ids="$2"
  local pretrain_save_name="$3"
  local pretrain_checkpoint="$4"
  local save_name="$5"
  local log_file="$6"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${ROOT_DIR}/output/nerf_rpn/results/${save_name}_eval/eval.json" ]]; then
    log "skip transfer variant=${variant} existing_eval=${ROOT_DIR}/output/nerf_rpn/results/${save_name}_eval/eval.json"
    return 0
  fi

  (
    cd "${ROOT_DIR}"
    PRETRAIN_SAVE_NAME="${pretrain_save_name}" \
    VARIANT_NAME="${variant}" \
    PRETRAIN_CHECKPOINT="${pretrain_checkpoint}" \
    GPU_IDS="${gpu_ids}" \
    DATASET_NAME=front3d \
    SPLIT_NAME=3dfront \
    PERCENT_TRAIN="${FCOS_PERCENT_TRAIN}" \
    FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS}" \
    FCOS_LR="${FCOS_LR}" \
    FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY}" \
    FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU}" \
    USE_WANDB="${USE_WANDB}" \
    WANDB_MODE="${WANDB_MODE}" \
    SAVE_NAME="${save_name}" \
    RUN_TAG="${save_name}" \
    bash nerf_rpn/run_fcos_probe_variant.sh
  ) >> "${log_file}" 2>&1
}

wait_for_pids() {
  local -n pids_ref=$1
  local -n names_ref=$2
  local failed=0
  local i
  for i in "${!pids_ref[@]}"; do
    if ! wait "${pids_ref[$i]}"; then
      log "job_failed name=${names_ref[$i]}"
      failed=1
    else
      log "job_done name=${names_ref[$i]}"
    fi
  done
  (( failed == 0 )) || probe_die "one or more chained jobs failed"
}

run_step_a() {
  local pids=()
  local names=()

  log "step_a start epoch_30 checkpoint FCOS reruns"
  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${STEPA_GPU_IDS}" probe_wait_for_runway
  fi

  run_transfer_job \
    baseline \
    "${STEPA_BASELINE_GPU}" \
    "nerfmae_all_p0.1_e30" \
    "../output/nerf_mae/results/nerfmae_all_p0.1_e30/epoch_30.pt" \
    "nerfmae_all_p0.1_e30_epoch30_fcos${FCOS_NUM_EPOCHS}" \
    "${LOG_ROOT}/${CHAIN_NAME}.baseline_epoch30.log" &
  pids+=($!)
  names+=("baseline_epoch30")

  run_transfer_job \
    alpha_only \
    "${STEPA_ALPHA_GPU}" \
    "nerfmae_alpha_only_p0.1_e30" \
    "../output/nerf_mae/results/nerfmae_alpha_only_p0.1_e30/epoch_30.pt" \
    "nerfmae_alpha_only_p0.1_e30_epoch30_fcos${FCOS_NUM_EPOCHS}" \
    "${LOG_ROOT}/${CHAIN_NAME}.alpha_epoch30.log" &
  pids+=($!)
  names+=("alpha_epoch30")

  run_transfer_job \
    radiance_only \
    "${STEPA_RADIANCE_GPU}" \
    "nerfmae_radiance_only_p0.1_e30" \
    "../output/nerf_mae/results/nerfmae_radiance_only_p0.1_e30/epoch_30.pt" \
    "nerfmae_radiance_only_p0.1_e30_epoch30_fcos${FCOS_NUM_EPOCHS}" \
    "${LOG_ROOT}/${CHAIN_NAME}.radiance_epoch30.log" &
  pids+=($!)
  names+=("radiance_epoch30")

  wait_for_pids pids names
  log "step_a done"
}

run_step_b() {
  local masked_save_name="nerfmae_masked_only_rgb_loss_p0.1_e30"
  local masked_epoch30="${ROOT_DIR}/output/nerf_mae/results/${masked_save_name}/epoch_30.pt"
  local masked_pretrain_log="${LOG_ROOT}/${CHAIN_NAME}.masked_pretrain_e30.log"
  local masked_transfer_log="${LOG_ROOT}/${CHAIN_NAME}.masked_epoch30_fcos.log"
  local masked_transfer_save="nerfmae_masked_only_rgb_loss_p0.1_e30_epoch30_fcos${FCOS_NUM_EPOCHS}"

  log "step_b start masked_only_e30 pretrain + FCOS"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${masked_epoch30}" ]]; then
    log "skip masked pretrain existing_ckpt=${masked_epoch30}"
  else
    if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
      GPU_IDS="${STEPB_PRETRAIN_GPU_IDS}" probe_wait_for_runway
    fi
    (
      cd "${ROOT_DIR}/nerf_mae"
      PATH="${PROBE_ENV_PREFIX}/bin:${PATH}" \
      PYTHONPATH="${ROOT_DIR}" \
      SAVE_NAME="${masked_save_name}" \
      RUN_TAG="${masked_save_name}" \
      GPU_IDS="${STEPB_PRETRAIN_GPU_IDS}" \
      NUM_EPOCHS="${MASKED_PRETRAIN_EPOCHS}" \
      PERCENT_TRAIN="${MASKED_PRETRAIN_PERCENT_TRAIN}" \
      BATCH_SIZE_PER_GPU="${MASKED_PRETRAIN_BATCH_SIZE_PER_GPU}" \
      USE_WANDB="${USE_WANDB}" \
      WANDB_MODE="${WANDB_MODE}" \
      bash probe_scripts/train_masked_only_rgb_loss.sh
    ) >> "${masked_pretrain_log}" 2>&1
  fi

  [[ -f "${masked_epoch30}" ]] || probe_die "missing masked epoch_30 checkpoint ${masked_epoch30}"

  run_transfer_job \
    masked_only_rgb_loss \
    "${STEPB_FCOS_GPU_IDS}" \
    "${masked_save_name}" \
    "../output/nerf_mae/results/${masked_save_name}/epoch_30.pt" \
    "${masked_transfer_save}" \
    "${masked_transfer_log}"

  log "step_b done"
}

log "start follow-up chain A(epoch30 reruns) -> B(masked_only_e30)"
run_step_a
run_step_b
log "follow-up chain done"
