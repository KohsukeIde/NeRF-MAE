#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_probe_common.sh"

CHAIN_NAME="${CHAIN_NAME:-nerfmae_seed_repl_alpha_shuffle}"
LOG_ROOT="${LOG_ROOT:-${PROBE_LOG_ROOT_DEFAULT}}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${CHAIN_NAME}.chain.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${CHAIN_NAME}.chain.pid}"
TMUX_SESSION="${TMUX_SESSION:-${CHAIN_NAME//[^[:alnum:]_]/_}_chain}"

SEED_LIST="${SEED_LIST:-1,2,3}"
INCLUDE_BASELINE="${INCLUDE_BASELINE:-1}"
WAIT_BEFORE_CHAIN="${WAIT_BEFORE_CHAIN:-1}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-15}"
WAIT_STABLE_SECONDS="${WAIT_STABLE_SECONDS:-60}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
DETERMINISTIC="${DETERMINISTIC:-1}"

FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS:-100}"
FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU:-2}"
FCOS_LR="${FCOS_LR:-1e-4}"
FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY:-1e-3}"
FCOS_PERCENT_TRAIN="${FCOS_PERCENT_TRAIN:-1.0}"

GPU_FAIR="${GPU_FAIR:-0}"
GPU_BASELINE="${GPU_BASELINE:-1}"
GPU_ALPHA="${GPU_ALPHA:-2}"
GPU_MASKED="${GPU_MASKED:-3}"

ALPHA_SHUFFLE_PRETRAIN_GPU_IDS="${ALPHA_SHUFFLE_PRETRAIN_GPU_IDS:-0,1,2,3}"
ALPHA_SHUFFLE_PRETRAIN_EPOCHS="${ALPHA_SHUFFLE_PRETRAIN_EPOCHS:-30}"
ALPHA_SHUFFLE_PRETRAIN_PERCENT_TRAIN="${ALPHA_SHUFFLE_PRETRAIN_PERCENT_TRAIN:-0.1}"
ALPHA_SHUFFLE_PRETRAIN_BATCH_SIZE_PER_GPU="${ALPHA_SHUFFLE_PRETRAIN_BATCH_SIZE_PER_GPU:-4}"
ALPHA_SHUFFLE_PRETRAIN_SEED="${ALPHA_SHUFFLE_PRETRAIN_SEED:-1}"
ALPHA_SHUFFLE_FCOS_GPU="${ALPHA_SHUFFLE_FCOS_GPU:-0}"
ALPHA_SHUFFLE_FCOS_SEED="${ALPHA_SHUFFLE_FCOS_SEED:-1}"

mkdir -p "${LOG_ROOT}"

if [[ "${FOREGROUND:-0}" != "1" ]]; then
  if env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
    echo "[info] chain already running in tmux session=${TMUX_SESSION}"
    echo "[info] log=${LOG_FILE}"
    exit 0
  fi
  env -u LD_LIBRARY_PATH -u PYTHONPATH tmux new-session -d -s "${TMUX_SESSION}" \
    "cd '${ROOT_DIR}' && FOREGROUND=1 bash '${SCRIPT_DIR}/run_seed_repl_and_alpha_shuffle_chain.sh'"
  echo "[info] started detached chain"
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

eval_exists() {
  local save_name="$1"
  [[ -f "${ROOT_DIR}/output/nerf_rpn/results/${save_name}_eval/eval.json" ]]
}

diagnostic_eval_exists() {
  local name="$1"
  [[ -f "${ROOT_DIR}/output/nerf_rpn/results/${name}/eval.json" ]]
}

run_diagnostics_if_needed() {
  if [[ "${SKIP_EXISTING}" == "1" ]] \
    && diagnostic_eval_exists "front3d_scratch_samepath_fcos100_diagnostics" \
    && diagnostic_eval_exists "nerfmae_all_p0.1_e30_epoch30_fcos100_diagnostics" \
    && diagnostic_eval_exists "nerfmae_alpha_only_p0.1_e30_epoch30_fcos100_diagnostics" \
    && diagnostic_eval_exists "nerfmae_masked_only_rgb_loss_p0.1_e30_epoch30_fcos100_diagnostics"; then
    log "skip diagnostics existing outputs detected"
    return 0
  fi

  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="0,1,2,3" probe_wait_for_runway
  fi
  log "step_diagnostics start"
  (
    cd "${ROOT_DIR}"
    CHAIN_NAME="nerfmae_shortcut_diagnostics" \
    bash nerf_mae/probe_scripts/run_shortcut_diagnostic_dump_chain.sh
  )
  log "step_diagnostics done"
}

run_job_probe() {
  local variant="$1"
  local gpu_ids="$2"
  local pretrain_save_name="$3"
  local pretrain_checkpoint="$4"
  local save_name="$5"
  local seed="$6"
  local log_file="$7"

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
    SEED="${seed}" \
    DETERMINISTIC="${DETERMINISTIC}" \
    bash nerf_rpn/run_fcos_probe_variant.sh
  ) >> "${log_file}" 2>&1 &
  RUN_VARIANT_PID=$!
}

run_job_fair_scratch() {
  local gpu_ids="$1"
  local save_name="$2"
  local seed="$3"
  local log_file="$4"

  (
    cd "${ROOT_DIR}"
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
    SEED="${seed}" \
    DETERMINISTIC="${DETERMINISTIC}" \
    bash nerf_rpn/run_fcos_fair_scratch_variant.sh
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

run_seed_wave() {
  local seed="$1"
  local pids=()
  local names=()
  local gpus="0,1,2,3"
  local scratch_save="front3d_scratch_samepath_seed${seed}_fcos${FCOS_NUM_EPOCHS}"
  local baseline_save="nerfmae_all_p0.1_e30_epoch30_seed${seed}_fcos${FCOS_NUM_EPOCHS}"
  local alpha_save="nerfmae_alpha_only_p0.1_e30_epoch30_seed${seed}_fcos${FCOS_NUM_EPOCHS}"
  local masked_save="nerfmae_masked_only_rgb_loss_p0.1_e30_epoch30_seed${seed}_fcos${FCOS_NUM_EPOCHS}"
  local pending=0

  log "step_seed start seed=${seed}"
  if ! { [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${scratch_save}"; }; then
    pending=1
  fi
  if [[ "${INCLUDE_BASELINE}" == "1" ]] && ! { [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${baseline_save}"; }; then
    pending=1
  fi
  if ! { [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${alpha_save}"; }; then
    pending=1
  fi
  if ! { [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${masked_save}"; }; then
    pending=1
  fi

  if (( pending == 0 )); then
    log "step_seed no pending jobs seed=${seed}"
    log "step_seed done seed=${seed}"
    return 0
  fi

  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${gpus}" probe_wait_for_runway
  fi

  if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${scratch_save}"; then
    log "skip fair_scratch_seed${seed} existing_eval=${ROOT_DIR}/output/nerf_rpn/results/${scratch_save}_eval/eval.json"
  else
    run_job_fair_scratch \
      "${GPU_FAIR}" \
      "${scratch_save}" \
      "${seed}" \
      "${LOG_ROOT}/${CHAIN_NAME}.seed${seed}.fair_scratch.log"
    pids+=("${RUN_VARIANT_PID}")
    names+=("fair_scratch_seed${seed}")
  fi

  if [[ "${INCLUDE_BASELINE}" == "1" ]]; then
    if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${baseline_save}"; then
      log "skip baseline_seed${seed} existing_eval=${ROOT_DIR}/output/nerf_rpn/results/${baseline_save}_eval/eval.json"
    else
      run_job_probe \
        baseline \
        "${GPU_BASELINE}" \
        "nerfmae_all_p0.1_e30" \
        "../output/nerf_mae/results/nerfmae_all_p0.1_e30/epoch_30.pt" \
        "${baseline_save}" \
        "${seed}" \
        "${LOG_ROOT}/${CHAIN_NAME}.seed${seed}.baseline.log"
      pids+=("${RUN_VARIANT_PID}")
      names+=("baseline_seed${seed}")
    fi
  fi

  if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${alpha_save}"; then
    log "skip alpha_only_seed${seed} existing_eval=${ROOT_DIR}/output/nerf_rpn/results/${alpha_save}_eval/eval.json"
  else
    run_job_probe \
      alpha_only \
      "${GPU_ALPHA}" \
      "nerfmae_alpha_only_p0.1_e30" \
      "../output/nerf_mae/results/nerfmae_alpha_only_p0.1_e30/epoch_30.pt" \
      "${alpha_save}" \
      "${seed}" \
      "${LOG_ROOT}/${CHAIN_NAME}.seed${seed}.alpha.log"
    pids+=("${RUN_VARIANT_PID}")
    names+=("alpha_only_seed${seed}")
  fi

  if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${masked_save}"; then
    log "skip masked_only_seed${seed} existing_eval=${ROOT_DIR}/output/nerf_rpn/results/${masked_save}_eval/eval.json"
  else
    run_job_probe \
      masked_only_rgb_loss \
      "${GPU_MASKED}" \
      "nerfmae_masked_only_rgb_loss_p0.1_e30" \
      "../output/nerf_mae/results/nerfmae_masked_only_rgb_loss_p0.1_e30/epoch_30.pt" \
      "${masked_save}" \
      "${seed}" \
      "${LOG_ROOT}/${CHAIN_NAME}.seed${seed}.masked.log"
    pids+=("${RUN_VARIANT_PID}")
    names+=("masked_only_seed${seed}")
  fi

  if (( ${#pids[@]} > 0 )); then
    wait_for_pids pids names
  else
    log "step_seed no pending jobs seed=${seed}"
  fi
  log "step_seed done seed=${seed}"
}

run_seed_replication() {
  local seed
  IFS=',' read -r -a seeds <<< "${SEED_LIST}"
  for seed in "${seeds[@]}"; do
    seed="${seed//[[:space:]]/}"
    [[ -n "${seed}" ]] || continue
    run_seed_wave "${seed}"
  done
}

run_alpha_shuffle() {
  local pretrain_save="nerfmae_alpha_shuffle_p0.1_e30_seed${ALPHA_SHUFFLE_PRETRAIN_SEED}"
  local pretrain_epoch30="${ROOT_DIR}/output/nerf_mae/results/${pretrain_save}/epoch_30.pt"
  local fcos_save="nerfmae_alpha_shuffle_p0.1_e30_seed${ALPHA_SHUFFLE_PRETRAIN_SEED}_epoch30_seed${ALPHA_SHUFFLE_FCOS_SEED}_fcos${FCOS_NUM_EPOCHS}"
  local fcos_eval="${ROOT_DIR}/output/nerf_rpn/results/${fcos_save}_eval/eval.json"
  local pretrain_log="${LOG_ROOT}/${CHAIN_NAME}.alpha_shuffle_pretrain.log"
  local fcos_log="${LOG_ROOT}/${CHAIN_NAME}.alpha_shuffle_fcos.log"

  log "step_alpha_shuffle start"
  log "step_alpha_shuffle pretrain_log=${pretrain_log}"
  log "step_alpha_shuffle fcos_log=${fcos_log}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${pretrain_epoch30}" ]]; then
    log "skip alpha_shuffle pretrain existing_ckpt=${pretrain_epoch30}"
  else
    if [[ -d "${ROOT_DIR}/output/nerf_mae/results/${pretrain_save}" ]]; then
      log "remove stale alpha_shuffle pretrain dir=${ROOT_DIR}/output/nerf_mae/results/${pretrain_save}"
      rm -rf "${ROOT_DIR}/output/nerf_mae/results/${pretrain_save}"
    fi
    if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
      GPU_IDS="${ALPHA_SHUFFLE_PRETRAIN_GPU_IDS}" probe_wait_for_runway
    fi
    (
      cd "${ROOT_DIR}/nerf_mae"
      PATH="${PROBE_ENV_PREFIX}/bin:${PATH}" \
      PYTHONPATH="${ROOT_DIR}" \
      SAVE_NAME="${pretrain_save}" \
      RUN_TAG="${pretrain_save}" \
      GPU_IDS="${ALPHA_SHUFFLE_PRETRAIN_GPU_IDS}" \
      NUM_EPOCHS="${ALPHA_SHUFFLE_PRETRAIN_EPOCHS}" \
      PERCENT_TRAIN="${ALPHA_SHUFFLE_PRETRAIN_PERCENT_TRAIN}" \
      BATCH_SIZE_PER_GPU="${ALPHA_SHUFFLE_PRETRAIN_BATCH_SIZE_PER_GPU}" \
      USE_WANDB="${USE_WANDB}" \
      WANDB_MODE="${WANDB_MODE}" \
      SEED="${ALPHA_SHUFFLE_PRETRAIN_SEED}" \
      DETERMINISTIC="${DETERMINISTIC}" \
      bash probe_scripts/train_alpha_shuffle.sh
    ) >> "${pretrain_log}" 2>&1
  fi

  [[ -f "${pretrain_epoch30}" ]] || probe_die "missing alpha_shuffle epoch_30 checkpoint ${pretrain_epoch30}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${fcos_eval}" ]]; then
    log "skip alpha_shuffle FCOS existing_eval=${fcos_eval}"
  else
    if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
      GPU_IDS="${ALPHA_SHUFFLE_FCOS_GPU}" probe_wait_for_runway
    fi
    (
      cd "${ROOT_DIR}"
      PRETRAIN_SAVE_NAME="${pretrain_save}" \
      VARIANT_NAME="alpha_shuffle" \
      PRETRAIN_CHECKPOINT="../output/nerf_mae/results/${pretrain_save}/epoch_30.pt" \
      GPU_IDS="${ALPHA_SHUFFLE_FCOS_GPU}" \
      DATASET_NAME=front3d \
      SPLIT_NAME=3dfront \
      PERCENT_TRAIN="${FCOS_PERCENT_TRAIN}" \
      FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS}" \
      FCOS_LR="${FCOS_LR}" \
      FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY}" \
      FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU}" \
      USE_WANDB="${USE_WANDB}" \
      WANDB_MODE="${WANDB_MODE}" \
      SAVE_NAME="${fcos_save}" \
      RUN_TAG="${fcos_save}" \
      SEED="${ALPHA_SHUFFLE_FCOS_SEED}" \
      DETERMINISTIC="${DETERMINISTIC}" \
      bash nerf_rpn/run_fcos_probe_variant.sh
    ) >> "${fcos_log}" 2>&1
  fi

  log "step_alpha_shuffle done"
}

log "start chained run diagnostics -> seed replication -> alpha_shuffle"
run_diagnostics_if_needed
run_seed_replication
run_alpha_shuffle
log "chain done"
