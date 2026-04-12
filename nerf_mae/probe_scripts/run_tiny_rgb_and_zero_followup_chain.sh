#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_probe_common.sh"

CHAIN_NAME="${CHAIN_NAME:-nerfmae_tiny_rgb_zero_followup}"
LOG_ROOT="${LOG_ROOT:-${PROBE_LOG_ROOT_DEFAULT}}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${CHAIN_NAME}.chain.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${CHAIN_NAME}.chain.pid}"
TMUX_SESSION="${TMUX_SESSION:-${CHAIN_NAME//[^[:alnum:]_]/_}_chain}"

WAIT_BEFORE_CHAIN="${WAIT_BEFORE_CHAIN:-1}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-30}"
WAIT_STABLE_SECONDS="${WAIT_STABLE_SECONDS:-240}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
DETERMINISTIC="${DETERMINISTIC:-1}"

FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS:-100}"
FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU:-2}"
FCOS_LR="${FCOS_LR:-1e-4}"
FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY:-1e-3}"
FCOS_LR_SCHEDULER="${FCOS_LR_SCHEDULER:-onecycle_epoch}"

TINY_RGB_WEIGHTS="${TINY_RGB_WEIGHTS:-0.02,0.05,0.1}"
TINY_RGB_SEED="${TINY_RGB_SEED:-1}"
TINY_RGB_PRETRAIN_GPU_IDS="${TINY_RGB_PRETRAIN_GPU_IDS:-0,1,2,3}"
TINY_RGB_PRETRAIN_EPOCHS="${TINY_RGB_PRETRAIN_EPOCHS:-30}"
TINY_RGB_PRETRAIN_PERCENT_TRAIN="${TINY_RGB_PRETRAIN_PERCENT_TRAIN:-0.1}"
TINY_RGB_PRETRAIN_BATCH_SIZE_PER_GPU="${TINY_RGB_PRETRAIN_BATCH_SIZE_PER_GPU:-4}"
TINY_RGB_FCOS_GPU_IDS="${TINY_RGB_FCOS_GPU_IDS:-0,1,2}"

ZERO_E100_SEEDS="${ZERO_E100_SEEDS:-1,2,3}"
ZERO_E100_PRETRAIN_GPU_IDS="${ZERO_E100_PRETRAIN_GPU_IDS:-0,1,2,3}"
ZERO_E100_PRETRAIN_EPOCHS="${ZERO_E100_PRETRAIN_EPOCHS:-100}"
ZERO_E100_PRETRAIN_PERCENT_TRAIN="${ZERO_E100_PRETRAIN_PERCENT_TRAIN:-0.1}"
ZERO_E100_PRETRAIN_BATCH_SIZE_PER_GPU="${ZERO_E100_PRETRAIN_BATCH_SIZE_PER_GPU:-4}"
ZERO_E100_FCOS_GPU_IDS="${ZERO_E100_FCOS_GPU_IDS:-0,1,2}"

mkdir -p "${LOG_ROOT}"

if [[ "${FOREGROUND:-0}" != "1" ]]; then
  if env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
    echo "[info] chain already running in tmux session=${TMUX_SESSION}"
    echo "[info] log=${LOG_FILE}"
    exit 0
  fi
  env -u LD_LIBRARY_PATH -u PYTHONPATH tmux new-session -d -s "${TMUX_SESSION}" \
    "cd '${ROOT_DIR}' && FOREGROUND=1 WAIT_BEFORE_CHAIN='${WAIT_BEFORE_CHAIN}' WAIT_POLL_SECONDS='${WAIT_POLL_SECONDS}' WAIT_STABLE_SECONDS='${WAIT_STABLE_SECONDS}' WAIT_TIMEOUT_SECONDS='${WAIT_TIMEOUT_SECONDS}' SKIP_EXISTING='${SKIP_EXISTING}' USE_WANDB='${USE_WANDB}' WANDB_MODE='${WANDB_MODE}' DETERMINISTIC='${DETERMINISTIC}' FCOS_NUM_EPOCHS='${FCOS_NUM_EPOCHS}' FCOS_BATCH_SIZE_PER_GPU='${FCOS_BATCH_SIZE_PER_GPU}' FCOS_LR='${FCOS_LR}' FCOS_WEIGHT_DECAY='${FCOS_WEIGHT_DECAY}' FCOS_LR_SCHEDULER='${FCOS_LR_SCHEDULER}' TINY_RGB_WEIGHTS='${TINY_RGB_WEIGHTS}' TINY_RGB_SEED='${TINY_RGB_SEED}' TINY_RGB_PRETRAIN_GPU_IDS='${TINY_RGB_PRETRAIN_GPU_IDS}' TINY_RGB_PRETRAIN_EPOCHS='${TINY_RGB_PRETRAIN_EPOCHS}' TINY_RGB_PRETRAIN_PERCENT_TRAIN='${TINY_RGB_PRETRAIN_PERCENT_TRAIN}' TINY_RGB_PRETRAIN_BATCH_SIZE_PER_GPU='${TINY_RGB_PRETRAIN_BATCH_SIZE_PER_GPU}' TINY_RGB_FCOS_GPU_IDS='${TINY_RGB_FCOS_GPU_IDS}' ZERO_E100_SEEDS='${ZERO_E100_SEEDS}' ZERO_E100_PRETRAIN_GPU_IDS='${ZERO_E100_PRETRAIN_GPU_IDS}' ZERO_E100_PRETRAIN_EPOCHS='${ZERO_E100_PRETRAIN_EPOCHS}' ZERO_E100_PRETRAIN_PERCENT_TRAIN='${ZERO_E100_PRETRAIN_PERCENT_TRAIN}' ZERO_E100_PRETRAIN_BATCH_SIZE_PER_GPU='${ZERO_E100_PRETRAIN_BATCH_SIZE_PER_GPU}' ZERO_E100_FCOS_GPU_IDS='${ZERO_E100_FCOS_GPU_IDS}' BLOCK_ON_TMUX_SESSIONS='${BLOCK_ON_TMUX_SESSIONS:-}' BLOCK_ON_PID_FILES='${BLOCK_ON_PID_FILES:-}' WAIT_MEMORY_USED_MAX_MIB='${WAIT_MEMORY_USED_MAX_MIB:-512}' WAIT_UTIL_MAX='${WAIT_UTIL_MAX:-10}' bash '${SCRIPT_DIR}/run_tiny_rgb_and_zero_followup_chain.sh'"
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

parse_csv_to_array() {
  local csv="$1"
  local -n out_ref=$2
  IFS=',' read -r -a out_ref <<< "${csv}"
  local i
  for i in "${!out_ref[@]}"; do
    out_ref[$i]="${out_ref[$i]//[[:space:]]/}"
  done
}

weight_tag() {
  local weight="$1"
  printf "w%s" "${weight//./p}"
}

eval_exists() {
  local save_name="$1"
  [[ -f "${ROOT_DIR}/output/nerf_rpn/results/${save_name}_eval/eval.json" ]]
}

pretrain_epoch_exists() {
  local save_name="$1"
  local epoch="$2"
  [[ -f "${ROOT_DIR}/output/nerf_mae/results/${save_name}/epoch_${epoch}.pt" ]]
}

run_pretrain_script() {
  local save_name="$1"
  local gpu_ids="$2"
  local num_epochs="$3"
  local percent_train="$4"
  local batch_size_per_gpu="$5"
  local seed="$6"
  local script_rel="$7"
  local log_file="$8"
  shift 8

  (
    cd "${ROOT_DIR}/nerf_mae"
    env "$@" \
      PATH="${PROBE_ENV_PREFIX}/bin:${PATH}" \
      PYTHONPATH="${ROOT_DIR}" \
      SAVE_NAME="${save_name}" \
      RUN_TAG="${save_name}" \
      GPU_IDS="${gpu_ids}" \
      NUM_EPOCHS="${num_epochs}" \
      PERCENT_TRAIN="${percent_train}" \
      BATCH_SIZE_PER_GPU="${batch_size_per_gpu}" \
      USE_WANDB="${USE_WANDB}" \
      WANDB_MODE="${WANDB_MODE}" \
      SEED="${seed}" \
      DETERMINISTIC="${DETERMINISTIC}" \
      bash "${script_rel}"
  ) >> "${log_file}" 2>&1
}

run_job_probe() {
  local variant="$1"
  local gpu_ids="$2"
  local pretrain_save_name="$3"
  local pretrain_checkpoint="$4"
  local save_name="$5"
  local seed="$6"
  local percent_train="$7"
  local log_file="$8"
  (
    cd "${ROOT_DIR}"
    PRETRAIN_SAVE_NAME="${pretrain_save_name}" \
    VARIANT_NAME="${variant}" \
    PRETRAIN_CHECKPOINT="${pretrain_checkpoint}" \
    GPU_IDS="${gpu_ids}" \
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
    SEED="${seed}" \
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
  (( failed == 0 )) || probe_die "one or more jobs failed"
}

run_parallel_probe_wave() {
  local gpu_csv="$1"
  local percent_train="$2"
  local -n jobs_ref=$3
  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${gpu_csv}" probe_wait_for_runway
  fi
  local gpus=()
  parse_csv_to_array "${gpu_csv}" gpus
  local pids=()
  local names=()
  local idx=0
  local job variant pretrain_save checkpoint save_name seed gpu_id
  for job in "${jobs_ref[@]}"; do
    IFS=':' read -r variant pretrain_save checkpoint save_name seed <<< "${job}"
    if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${save_name}"; then
      log "skip fcos variant=${variant} save=${save_name}"
      continue
    fi
    gpu_id="${gpus[$idx]}"
    idx=$((idx + 1))
    run_job_probe \
      "${variant}" \
      "${gpu_id}" \
      "${pretrain_save}" \
      "${checkpoint}" \
      "${save_name}" \
      "${seed}" \
      "${percent_train}" \
      "${LOG_ROOT}/${CHAIN_NAME}.${save_name}.log"
    pids+=("${RUN_VARIANT_PID}")
    names+=("${save_name}")
  done
  if (( ${#pids[@]} > 0 )); then
    wait_for_pids pids names
  fi
}

run_tiny_rgb_e30_sweep() {
  local weights=()
  parse_csv_to_array "${TINY_RGB_WEIGHTS}" weights
  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${TINY_RGB_PRETRAIN_GPU_IDS}" probe_wait_for_runway
  fi
  log "step_tiny_rgb_e30_pretrain start"
  local weight tag save_name
  for weight in "${weights[@]}"; do
    tag="$(weight_tag "${weight}")"
    save_name="nerfmae_alpha_target_tiny_rgb_${tag}_p0.1_e30_seed${TINY_RGB_SEED}"
    if [[ "${SKIP_EXISTING}" == "1" ]] && pretrain_epoch_exists "${save_name}" "${TINY_RGB_PRETRAIN_EPOCHS}"; then
      log "skip tiny_rgb pretrain save=${save_name}"
      continue
    fi
    run_pretrain_script \
      "${save_name}" \
      "${TINY_RGB_PRETRAIN_GPU_IDS}" \
      "${TINY_RGB_PRETRAIN_EPOCHS}" \
      "${TINY_RGB_PRETRAIN_PERCENT_TRAIN}" \
      "${TINY_RGB_PRETRAIN_BATCH_SIZE_PER_GPU}" \
      "${TINY_RGB_SEED}" \
      "probe_scripts/train_alpha_target_tiny_rgb.sh" \
      "${LOG_ROOT}/${CHAIN_NAME}.tiny_rgb.${tag}.pretrain.log" \
      "TINY_RGB_WEIGHT=${weight}"
    log "tiny_rgb pretrain done save=${save_name}"
  done

  local full_jobs=()
  local low_jobs=()
  for weight in "${weights[@]}"; do
    tag="$(weight_tag "${weight}")"
    save_name="nerfmae_alpha_target_tiny_rgb_${tag}_p0.1_e30_seed${TINY_RGB_SEED}"
    full_jobs+=("tiny_rgb_${tag}:${save_name}:../output/nerf_mae/results/${save_name}/epoch_${TINY_RGB_PRETRAIN_EPOCHS}.pt:${save_name}_epoch30_sched_epoch_seed${TINY_RGB_SEED}_fcos${FCOS_NUM_EPOCHS}:${TINY_RGB_SEED}")
    low_jobs+=("tiny_rgb_${tag}:${save_name}:../output/nerf_mae/results/${save_name}/epoch_${TINY_RGB_PRETRAIN_EPOCHS}.pt:${save_name}_epoch30_sched_epoch_pt02_seed${TINY_RGB_SEED}_fcos${FCOS_NUM_EPOCHS}:${TINY_RGB_SEED}")
  done
  log "step_tiny_rgb_e30_full_fcos start"
  run_parallel_probe_wave "${TINY_RGB_FCOS_GPU_IDS}" "1.0" full_jobs
  log "step_tiny_rgb_e30_full_fcos done"
  log "step_tiny_rgb_e30_pt02_fcos start"
  run_parallel_probe_wave "${TINY_RGB_FCOS_GPU_IDS}" "0.2" low_jobs
  log "step_tiny_rgb_e30_pt02_fcos done"
}

run_alpha_target_zero_e100() {
  local seeds=()
  parse_csv_to_array "${ZERO_E100_SEEDS}" seeds
  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${ZERO_E100_PRETRAIN_GPU_IDS}" probe_wait_for_runway
  fi
  log "step_alpha_target_zero_e100_pretrain start"
  local seed save_name
  for seed in "${seeds[@]}"; do
    save_name="nerfmae_alpha_target_zero_p0.1_e100_seed${seed}"
    if [[ "${SKIP_EXISTING}" == "1" ]] && pretrain_epoch_exists "${save_name}" "${ZERO_E100_PRETRAIN_EPOCHS}"; then
      log "skip alpha_target_zero e100 pretrain save=${save_name}"
      continue
    fi
    run_pretrain_script \
      "${save_name}" \
      "${ZERO_E100_PRETRAIN_GPU_IDS}" \
      "${ZERO_E100_PRETRAIN_EPOCHS}" \
      "${ZERO_E100_PRETRAIN_PERCENT_TRAIN}" \
      "${ZERO_E100_PRETRAIN_BATCH_SIZE_PER_GPU}" \
      "${seed}" \
      "probe_scripts/train_alpha_target_zero.sh" \
      "${LOG_ROOT}/${CHAIN_NAME}.alpha_target_zero.e100.seed${seed}.pretrain.log"
    log "alpha_target_zero e100 pretrain done save=${save_name}"
  done

  local jobs=()
  for seed in "${seeds[@]}"; do
    save_name="nerfmae_alpha_target_zero_p0.1_e100_seed${seed}"
    jobs+=("alpha_target_zero:${save_name}:../output/nerf_mae/results/${save_name}/epoch_${ZERO_E100_PRETRAIN_EPOCHS}.pt:${save_name}_epoch100_sched_epoch_seed${seed}_fcos${FCOS_NUM_EPOCHS}:${seed}")
  done
  log "step_alpha_target_zero_e100_fcos start"
  run_parallel_probe_wave "${ZERO_E100_FCOS_GPU_IDS}" "1.0" jobs
  log "step_alpha_target_zero_e100_fcos done"
}

log "start tiny_rgb_e30_sweep -> alpha_target_zero_e100"
run_tiny_rgb_e30_sweep
run_alpha_target_zero_e100
log "chain done"
