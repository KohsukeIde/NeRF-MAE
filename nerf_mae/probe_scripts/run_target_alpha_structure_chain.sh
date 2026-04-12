#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_probe_common.sh"

CHAIN_NAME="${CHAIN_NAME:-nerfmae_target_alpha_structure}"
LOG_ROOT="${LOG_ROOT:-${PROBE_LOG_ROOT_DEFAULT}}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${CHAIN_NAME}.chain.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${CHAIN_NAME}.chain.pid}"
TMUX_SESSION="${TMUX_SESSION:-${CHAIN_NAME//[^[:alnum:]_]/_}_chain}"

WAIT_BEFORE_CHAIN="${WAIT_BEFORE_CHAIN:-1}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-30}"
WAIT_STABLE_SECONDS="${WAIT_STABLE_SECONDS:-180}"
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

E100_SEEDS="${E100_SEEDS:-1,2,3}"
E100_PRETRAIN_GPU_IDS="${E100_PRETRAIN_GPU_IDS:-0,1,2,3}"
E100_PRETRAIN_EPOCHS="${E100_PRETRAIN_EPOCHS:-100}"
E100_PRETRAIN_PERCENT_TRAIN="${E100_PRETRAIN_PERCENT_TRAIN:-0.1}"
E100_PRETRAIN_BATCH_SIZE_PER_GPU="${E100_PRETRAIN_BATCH_SIZE_PER_GPU:-4}"
E100_FCOS_GPU_IDS="${E100_FCOS_GPU_IDS:-0,1,2}"

E30_SEEDS="${E30_SEEDS:-1,2,3}"
E30_PRETRAIN_GPU_IDS="${E30_PRETRAIN_GPU_IDS:-0,1,2,3}"
E30_PRETRAIN_EPOCHS="${E30_PRETRAIN_EPOCHS:-30}"
E30_PRETRAIN_PERCENT_TRAIN="${E30_PRETRAIN_PERCENT_TRAIN:-0.1}"
E30_PRETRAIN_BATCH_SIZE_PER_GPU="${E30_PRETRAIN_BATCH_SIZE_PER_GPU:-4}"
E30_FCOS_GPU_IDS="${E30_FCOS_GPU_IDS:-0,1,2}"

DIAG_GPU_IDS="${DIAG_GPU_IDS:-0,1,2,3}"
GT_BOXES_DIR="${GT_BOXES_DIR:-/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/dataset/finetune/front3d_rpn_data/obb}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/analysis}"

mkdir -p "${LOG_ROOT}" "${ANALYSIS_ROOT}"

if [[ "${FOREGROUND:-0}" != "1" ]]; then
  if env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
    echo "[info] chain already running in tmux session=${TMUX_SESSION}"
    echo "[info] log=${LOG_FILE}"
    exit 0
  fi
  env -u LD_LIBRARY_PATH -u PYTHONPATH tmux new-session -d -s "${TMUX_SESSION}" \
    "cd '${ROOT_DIR}' && FOREGROUND=1 WAIT_BEFORE_CHAIN='${WAIT_BEFORE_CHAIN}' WAIT_POLL_SECONDS='${WAIT_POLL_SECONDS}' WAIT_STABLE_SECONDS='${WAIT_STABLE_SECONDS}' WAIT_TIMEOUT_SECONDS='${WAIT_TIMEOUT_SECONDS}' SKIP_EXISTING='${SKIP_EXISTING}' USE_WANDB='${USE_WANDB}' WANDB_MODE='${WANDB_MODE}' DETERMINISTIC='${DETERMINISTIC}' FCOS_NUM_EPOCHS='${FCOS_NUM_EPOCHS}' FCOS_BATCH_SIZE_PER_GPU='${FCOS_BATCH_SIZE_PER_GPU}' FCOS_LR='${FCOS_LR}' FCOS_WEIGHT_DECAY='${FCOS_WEIGHT_DECAY}' FCOS_LR_SCHEDULER='${FCOS_LR_SCHEDULER}' E100_SEEDS='${E100_SEEDS}' E100_PRETRAIN_GPU_IDS='${E100_PRETRAIN_GPU_IDS}' E100_PRETRAIN_EPOCHS='${E100_PRETRAIN_EPOCHS}' E100_PRETRAIN_PERCENT_TRAIN='${E100_PRETRAIN_PERCENT_TRAIN}' E100_PRETRAIN_BATCH_SIZE_PER_GPU='${E100_PRETRAIN_BATCH_SIZE_PER_GPU}' E100_FCOS_GPU_IDS='${E100_FCOS_GPU_IDS}' E30_SEEDS='${E30_SEEDS}' E30_PRETRAIN_GPU_IDS='${E30_PRETRAIN_GPU_IDS}' E30_PRETRAIN_EPOCHS='${E30_PRETRAIN_EPOCHS}' E30_PRETRAIN_PERCENT_TRAIN='${E30_PRETRAIN_PERCENT_TRAIN}' E30_PRETRAIN_BATCH_SIZE_PER_GPU='${E30_PRETRAIN_BATCH_SIZE_PER_GPU}' E30_FCOS_GPU_IDS='${E30_FCOS_GPU_IDS}' DIAG_GPU_IDS='${DIAG_GPU_IDS}' GT_BOXES_DIR='${GT_BOXES_DIR}' ANALYSIS_ROOT='${ANALYSIS_ROOT}' BLOCK_ON_TMUX_SESSIONS='${BLOCK_ON_TMUX_SESSIONS:-}' BLOCK_ON_PID_FILES='${BLOCK_ON_PID_FILES:-}' WAIT_MEMORY_USED_MAX_MIB='${WAIT_MEMORY_USED_MAX_MIB:-512}' WAIT_UTIL_MAX='${WAIT_UTIL_MAX:-10}' bash '${SCRIPT_DIR}/run_target_alpha_structure_chain.sh'"
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

eval_exists() {
  local save_name="$1"
  [[ -f "${ROOT_DIR}/output/nerf_rpn/results/${save_name}_eval/eval.json" ]]
}

pretrain_epoch_exists() {
  local save_name="$1"
  local epoch="$2"
  [[ -f "${ROOT_DIR}/output/nerf_mae/results/${save_name}/epoch_${epoch}.pt" ]]
}

diag_exists() {
  local save_name="$1"
  [[ -f "${ROOT_DIR}/output/nerf_rpn/results/${save_name}/eval.json" ]]
}

find_latest_checkpoint() {
  local dir="$1"
  find "${dir}" -maxdepth 1 -name 'model_best_ap50_ap25_*.pt' -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | awk '{print $2}'
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

run_pretrain_script() {
  local save_name="$1"
  local gpu_ids="$2"
  local num_epochs="$3"
  local percent_train="$4"
  local batch_size_per_gpu="$5"
  local seed="$6"
  local script_rel="$7"
  local log_file="$8"

  (
    cd "${ROOT_DIR}/nerf_mae"
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

run_job_diagnostic() {
  local variant="$1"
  local gpu_id="$2"
  local checkpoint="$3"
  local save_name="$4"
  local log_file="$5"
  (
    cd "${ROOT_DIR}"
    GPU_IDS="${gpu_id}" \
    VARIANT_NAME="${variant}" \
    CHECKPOINT="${checkpoint}" \
    SAVE_NAME="${save_name}" \
    SEED=1 \
    DETERMINISTIC="${DETERMINISTIC}" \
    bash nerf_rpn/run_fcos_diagnostic_variant.sh
  ) >> "${log_file}" 2>&1 &
  RUN_VARIANT_PID=$!
}

run_e100_alpha_target_shuffle_pretrains() {
  local seeds=()
  parse_csv_to_array "${E100_SEEDS}" seeds
  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${E100_PRETRAIN_GPU_IDS}" probe_wait_for_runway
  fi
  log "step_e100_alpha_target_shuffle_pretrains start"
  local seed save_name
  for seed in "${seeds[@]}"; do
    save_name="nerfmae_alpha_target_shuffle_p0.1_e100_seed${seed}"
    if [[ "${SKIP_EXISTING}" == "1" ]] && pretrain_epoch_exists "${save_name}" "${E100_PRETRAIN_EPOCHS}"; then
      log "skip e100 alpha_target_shuffle pretrain save=${save_name}"
      continue
    fi
    run_pretrain_script \
      "${save_name}" \
      "${E100_PRETRAIN_GPU_IDS}" \
      "${E100_PRETRAIN_EPOCHS}" \
      "${E100_PRETRAIN_PERCENT_TRAIN}" \
      "${E100_PRETRAIN_BATCH_SIZE_PER_GPU}" \
      "${seed}" \
      "probe_scripts/train_alpha_target_shuffle.sh" \
      "${LOG_ROOT}/${CHAIN_NAME}.e100.alpha_target_shuffle.seed${seed}.pretrain.log"
    log "e100 alpha_target_shuffle pretrain done save=${save_name}"
  done
  log "step_e100_alpha_target_shuffle_pretrains done"
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

run_e100_alpha_target_shuffle_fcos() {
  local seeds=()
  local jobs=()
  local seed save_name pretrain_save
  parse_csv_to_array "${E100_SEEDS}" seeds
  for seed in "${seeds[@]}"; do
    pretrain_save="nerfmae_alpha_target_shuffle_p0.1_e100_seed${seed}"
    save_name="${pretrain_save}_epoch100_sched_epoch_seed${seed}_fcos${FCOS_NUM_EPOCHS}"
    jobs+=("alpha_target_shuffle:${pretrain_save}:../output/nerf_mae/results/${pretrain_save}/epoch_${E100_PRETRAIN_EPOCHS}.pt:${save_name}:${seed}")
  done
  log "step_e100_alpha_target_shuffle_fcos start"
  run_parallel_probe_wave "${E100_FCOS_GPU_IDS}" "1.0" jobs
  log "step_e100_alpha_target_shuffle_fcos done"
}

run_e30_alpha_target_zero_pretrains() {
  local seeds=()
  parse_csv_to_array "${E30_SEEDS}" seeds
  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${E30_PRETRAIN_GPU_IDS}" probe_wait_for_runway
  fi
  log "step_e30_alpha_target_zero_pretrains start"
  local seed save_name
  for seed in "${seeds[@]}"; do
    save_name="nerfmae_alpha_target_zero_p0.1_e30_seed${seed}"
    if [[ "${SKIP_EXISTING}" == "1" ]] && pretrain_epoch_exists "${save_name}" "${E30_PRETRAIN_EPOCHS}"; then
      log "skip e30 alpha_target_zero pretrain save=${save_name}"
      continue
    fi
    run_pretrain_script \
      "${save_name}" \
      "${E30_PRETRAIN_GPU_IDS}" \
      "${E30_PRETRAIN_EPOCHS}" \
      "${E30_PRETRAIN_PERCENT_TRAIN}" \
      "${E30_PRETRAIN_BATCH_SIZE_PER_GPU}" \
      "${seed}" \
      "probe_scripts/train_alpha_target_zero.sh" \
      "${LOG_ROOT}/${CHAIN_NAME}.e30.alpha_target_zero.seed${seed}.pretrain.log"
    log "e30 alpha_target_zero pretrain done save=${save_name}"
  done
  log "step_e30_alpha_target_zero_pretrains done"
}

run_e30_alpha_target_zero_fcos() {
  local seeds=()
  local jobs=()
  local seed save_name pretrain_save
  parse_csv_to_array "${E30_SEEDS}" seeds
  for seed in "${seeds[@]}"; do
    pretrain_save="nerfmae_alpha_target_zero_p0.1_e30_seed${seed}"
    save_name="${pretrain_save}_epoch30_sched_epoch_seed${seed}_fcos${FCOS_NUM_EPOCHS}"
    jobs+=("alpha_target_zero:${pretrain_save}:../output/nerf_mae/results/${pretrain_save}/epoch_${E30_PRETRAIN_EPOCHS}.pt:${save_name}:${seed}")
  done
  log "step_e30_alpha_target_zero_fcos start"
  run_parallel_probe_wave "${E30_FCOS_GPU_IDS}" "1.0" jobs
  log "step_e30_alpha_target_zero_fcos done"
}

run_diagnostics_and_summary() {
  local diag_jobs=(
    "baseline_e100:${ROOT_DIR}/output/nerf_rpn/results/nerfmae_all_p0.1_e100_seed1_epoch100_sched_epoch_seed1_fcos100:nerfmae_all_p0.1_e100_seed1_epoch100_sched_epoch_seed1_target_alpha_diag"
    "alpha_target_only_e100:${ROOT_DIR}/output/nerf_rpn/results/nerfmae_alpha_target_only_p0.1_e100_seed1_epoch100_sched_epoch_seed1_fcos100:nerfmae_alpha_target_only_p0.1_e100_seed1_epoch100_sched_epoch_seed1_target_alpha_diag"
    "alpha_target_shuffle_e100:${ROOT_DIR}/output/nerf_rpn/results/nerfmae_alpha_target_shuffle_p0.1_e100_seed1_epoch100_sched_epoch_seed1_fcos100:nerfmae_alpha_target_shuffle_p0.1_e100_seed1_epoch100_sched_epoch_seed1_target_alpha_diag"
    "alpha_target_zero_e30:${ROOT_DIR}/output/nerf_rpn/results/nerfmae_alpha_target_zero_p0.1_e30_seed1_epoch30_sched_epoch_seed1_fcos100:nerfmae_alpha_target_zero_p0.1_e30_seed1_epoch30_sched_epoch_seed1_target_alpha_diag"
  )
  local gpus=()
  parse_csv_to_array "${DIAG_GPU_IDS}" gpus
  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${DIAG_GPU_IDS}" probe_wait_for_runway
  fi
  log "step_target_alpha_diagnostics start"
  local pids=()
  local names=()
  local idx=0
  local item label ckpt_dir save_name checkpoint gpu_id
  for item in "${diag_jobs[@]}"; do
    IFS=':' read -r label ckpt_dir save_name <<< "${item}"
    if [[ "${SKIP_EXISTING}" == "1" ]] && diag_exists "${save_name}"; then
      log "skip diagnostic save=${save_name}"
      continue
    fi
    checkpoint="$(find_latest_checkpoint "${ckpt_dir}")"
    [[ -n "${checkpoint}" ]] || probe_die "missing checkpoint in ${ckpt_dir}"
    gpu_id="${gpus[$idx]}"
    idx=$((idx + 1))
    run_job_diagnostic \
      "${label}" \
      "${gpu_id}" \
      "${checkpoint}" \
      "${save_name}" \
      "${LOG_ROOT}/${CHAIN_NAME}.${save_name}.log"
    pids+=("${RUN_VARIANT_PID}")
    names+=("${save_name}")
  done
  if (( ${#pids[@]} > 0 )); then
    wait_for_pids pids names
  fi

  local output_json="${ANALYSIS_ROOT}/${CHAIN_NAME}_diagnostics_summary.json"
  local output_md="${ANALYSIS_ROOT}/${CHAIN_NAME}_diagnostics_summary.md"
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="${ROOT_DIR}/nerf_rpn:${ROOT_DIR}" \
  "${PROBE_PYTHON_BIN}" "${ROOT_DIR}/nerf_rpn/tools/summarize_diagnostic_dumps.py" \
    --diagnostic "baseline_e100=${ROOT_DIR}/output/nerf_rpn/results/nerfmae_all_p0.1_e100_seed1_epoch100_sched_epoch_seed1_target_alpha_diag" \
    --diagnostic "alpha_target_only_e100=${ROOT_DIR}/output/nerf_rpn/results/nerfmae_alpha_target_only_p0.1_e100_seed1_epoch100_sched_epoch_seed1_target_alpha_diag" \
    --diagnostic "alpha_target_shuffle_e100=${ROOT_DIR}/output/nerf_rpn/results/nerfmae_alpha_target_shuffle_p0.1_e100_seed1_epoch100_sched_epoch_seed1_target_alpha_diag" \
    --diagnostic "alpha_target_zero_e30=${ROOT_DIR}/output/nerf_rpn/results/nerfmae_alpha_target_zero_p0.1_e30_seed1_epoch30_sched_epoch_seed1_target_alpha_diag" \
    --gt_boxes_dir "${GT_BOXES_DIR}" \
    --output_json "${output_json}" \
    --output_md "${output_md}" \
    >> "${LOG_ROOT}/${CHAIN_NAME}.diag.summary.log" 2>&1
  log "step_target_alpha_diagnostics done output_md=${output_md}"
}

log "start e100_alpha_target_shuffle -> e30_alpha_target_zero -> diagnostics"
run_e100_alpha_target_shuffle_pretrains
run_e100_alpha_target_shuffle_fcos
run_e30_alpha_target_zero_pretrains
run_e30_alpha_target_zero_fcos
run_diagnostics_and_summary
log "chain done"
