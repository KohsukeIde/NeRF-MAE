#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_probe_common.sh"

CHAIN_NAME="${CHAIN_NAME:-nerfmae_downstream_protocol_diagnosis}"
LOG_ROOT="${LOG_ROOT:-${PROBE_LOG_ROOT_DEFAULT}}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${CHAIN_NAME}.chain.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${CHAIN_NAME}.chain.pid}"
TMUX_SESSION="${TMUX_SESSION:-${CHAIN_NAME//[^[:alnum:]_]/_}_chain}"

WAIT_BEFORE_CHAIN="${WAIT_BEFORE_CHAIN:-1}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-15}"
WAIT_STABLE_SECONDS="${WAIT_STABLE_SECONDS:-60}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
DETERMINISTIC="${DETERMINISTIC:-1}"

SEED_LIST="${SEED_LIST:-1,2,3}"
NOAUG_SEED="${NOAUG_SEED:-3}"
FREEZE_SEED="${FREEZE_SEED:-3}"

FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS:-100}"
FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU:-2}"
FCOS_LR="${FCOS_LR:-1e-4}"
FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY:-1e-3}"
FCOS_PERCENT_TRAIN="${FCOS_PERCENT_TRAIN:-1.0}"
FCOS_LR_SCHEDULER="${FCOS_LR_SCHEDULER:-onecycle_epoch}"
FCOS_SCHEDULER_TOTAL_STEPS="${FCOS_SCHEDULER_TOTAL_STEPS:-}"

GPU_FAIR="${GPU_FAIR:-0}"
GPU_BASELINE="${GPU_BASELINE:-1}"
GPU_ALPHA="${GPU_ALPHA:-2}"
GPU_MASKED="${GPU_MASKED:-3}"

ALPHA_SHUFFLE_SEEDS="${ALPHA_SHUFFLE_SEEDS:-1,2,3}"
ALPHA_SHUFFLE_PRETRAIN_GPU_IDS="${ALPHA_SHUFFLE_PRETRAIN_GPU_IDS:-0,1,2,3}"
ALPHA_SHUFFLE_PRETRAIN_EPOCHS="${ALPHA_SHUFFLE_PRETRAIN_EPOCHS:-30}"
ALPHA_SHUFFLE_PRETRAIN_PERCENT_TRAIN="${ALPHA_SHUFFLE_PRETRAIN_PERCENT_TRAIN:-0.1}"
ALPHA_SHUFFLE_PRETRAIN_BATCH_SIZE_PER_GPU="${ALPHA_SHUFFLE_PRETRAIN_BATCH_SIZE_PER_GPU:-4}"
ALPHA_SHUFFLE_FCOS_GPU_IDS="${ALPHA_SHUFFLE_FCOS_GPU_IDS:-0,1,2}"

FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-10}"
FREEZE_BACKBONE_LR_SCALE="${FREEZE_BACKBONE_LR_SCALE:-1.0}"

mkdir -p "${LOG_ROOT}"

if [[ "${FOREGROUND:-0}" != "1" ]]; then
  if env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
    echo "[info] chain already running in tmux session=${TMUX_SESSION}"
    echo "[info] log=${LOG_FILE}"
    exit 0
  fi
  env -u LD_LIBRARY_PATH -u PYTHONPATH tmux new-session -d -s "${TMUX_SESSION}" \
    "cd '${ROOT_DIR}' && FOREGROUND=1 bash '${SCRIPT_DIR}/run_downstream_protocol_diagnosis_chain.sh'"
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

alpha_pretrain_exists() {
  local save_name="$1"
  [[ -f "${ROOT_DIR}/output/nerf_mae/results/${save_name}/epoch_30.pt" ]]
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

run_job_probe() {
  local variant="$1"
  local gpu_ids="$2"
  local pretrain_save_name="$3"
  local pretrain_checkpoint="$4"
  local save_name="$5"
  local seed="$6"
  local log_file="$7"
  local rotate_prob="$8"
  local flip_prob="$9"
  local rot_scale_prob="${10}"
  local lr_scheduler="${11}"
  local scheduler_total_steps="${12}"
  local freeze_backbone_epochs="${13}"
  local backbone_lr_scale="${14}"

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
    ROTATE_PROB="${rotate_prob}" \
    FLIP_PROB="${flip_prob}" \
    ROT_SCALE_PROB="${rot_scale_prob}" \
    LR_SCHEDULER="${lr_scheduler}" \
    SCHEDULER_TOTAL_STEPS="${scheduler_total_steps}" \
    FREEZE_BACKBONE_EPOCHS="${freeze_backbone_epochs}" \
    BACKBONE_LR_SCALE="${backbone_lr_scale}" \
    bash nerf_rpn/run_fcos_probe_variant.sh
  ) >> "${log_file}" 2>&1 &
  RUN_VARIANT_PID=$!
}

run_job_fair_scratch() {
  local gpu_ids="$1"
  local save_name="$2"
  local seed="$3"
  local log_file="$4"
  local rotate_prob="$5"
  local flip_prob="$6"
  local rot_scale_prob="$7"
  local lr_scheduler="$8"
  local scheduler_total_steps="$9"
  local freeze_backbone_epochs="${10}"
  local backbone_lr_scale="${11}"

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
    ROTATE_PROB="${rotate_prob}" \
    FLIP_PROB="${flip_prob}" \
    ROT_SCALE_PROB="${rot_scale_prob}" \
    LR_SCHEDULER="${lr_scheduler}" \
    SCHEDULER_TOTAL_STEPS="${scheduler_total_steps}" \
    FREEZE_BACKBONE_EPOCHS="${freeze_backbone_epochs}" \
    BACKBONE_LR_SCALE="${backbone_lr_scale}" \
    bash nerf_rpn/run_fcos_fair_scratch_variant.sh
  ) >> "${log_file}" 2>&1 &
  RUN_VARIANT_PID=$!
}

run_four_way_wave() {
  local phase_name="$1"
  local seed="$2"
  local save_tag="$3"
  local rotate_prob="$4"
  local flip_prob="$5"
  local rot_scale_prob="$6"
  local lr_scheduler="$7"
  local scheduler_total_steps="$8"
  local freeze_backbone_epochs="$9"
  local backbone_lr_scale="${10}"
  local pids=()
  local names=()
  local pending=0
  local gpus="0,1,2,3"

  local scratch_save="front3d_scratch_samepath_${save_tag}_seed${seed}_fcos${FCOS_NUM_EPOCHS}"
  local baseline_save="nerfmae_all_p0.1_e30_epoch30_${save_tag}_seed${seed}_fcos${FCOS_NUM_EPOCHS}"
  local alpha_save="nerfmae_alpha_only_p0.1_e30_epoch30_${save_tag}_seed${seed}_fcos${FCOS_NUM_EPOCHS}"
  local masked_save="nerfmae_masked_only_rgb_loss_p0.1_e30_epoch30_${save_tag}_seed${seed}_fcos${FCOS_NUM_EPOCHS}"

  log "phase=${phase_name} seed=${seed} start"
  if ! { [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${scratch_save}"; }; then
    pending=1
  fi
  if ! { [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${baseline_save}"; }; then
    pending=1
  fi
  if ! { [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${alpha_save}"; }; then
    pending=1
  fi
  if ! { [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${masked_save}"; }; then
    pending=1
  fi

  if (( pending == 0 )); then
    log "phase=${phase_name} seed=${seed} no pending jobs"
    return 0
  fi

  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${gpus}" probe_wait_for_runway
  fi

  if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${scratch_save}"; then
    log "skip ${phase_name} fair_scratch seed=${seed}"
  else
    run_job_fair_scratch \
      "${GPU_FAIR}" \
      "${scratch_save}" \
      "${seed}" \
      "${LOG_ROOT}/${CHAIN_NAME}.${save_tag}.seed${seed}.fair_scratch.log" \
      "${rotate_prob}" \
      "${flip_prob}" \
      "${rot_scale_prob}" \
      "${lr_scheduler}" \
      "${scheduler_total_steps}" \
      "${freeze_backbone_epochs}" \
      "${backbone_lr_scale}"
    pids+=("${RUN_VARIANT_PID}")
    names+=("${phase_name}_fair_scratch_seed${seed}")
  fi

  if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${baseline_save}"; then
    log "skip ${phase_name} baseline seed=${seed}"
  else
    run_job_probe \
      baseline \
      "${GPU_BASELINE}" \
      "nerfmae_all_p0.1_e30" \
      "../output/nerf_mae/results/nerfmae_all_p0.1_e30/epoch_30.pt" \
      "${baseline_save}" \
      "${seed}" \
      "${LOG_ROOT}/${CHAIN_NAME}.${save_tag}.seed${seed}.baseline.log" \
      "${rotate_prob}" \
      "${flip_prob}" \
      "${rot_scale_prob}" \
      "${lr_scheduler}" \
      "${scheduler_total_steps}" \
      "${freeze_backbone_epochs}" \
      "${backbone_lr_scale}"
    pids+=("${RUN_VARIANT_PID}")
    names+=("${phase_name}_baseline_seed${seed}")
  fi

  if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${alpha_save}"; then
    log "skip ${phase_name} alpha_only seed=${seed}"
  else
    run_job_probe \
      alpha_only \
      "${GPU_ALPHA}" \
      "nerfmae_alpha_only_p0.1_e30" \
      "../output/nerf_mae/results/nerfmae_alpha_only_p0.1_e30/epoch_30.pt" \
      "${alpha_save}" \
      "${seed}" \
      "${LOG_ROOT}/${CHAIN_NAME}.${save_tag}.seed${seed}.alpha.log" \
      "${rotate_prob}" \
      "${flip_prob}" \
      "${rot_scale_prob}" \
      "${lr_scheduler}" \
      "${scheduler_total_steps}" \
      "${freeze_backbone_epochs}" \
      "${backbone_lr_scale}"
    pids+=("${RUN_VARIANT_PID}")
    names+=("${phase_name}_alpha_only_seed${seed}")
  fi

  if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${masked_save}"; then
    log "skip ${phase_name} masked_only seed=${seed}"
  else
    run_job_probe \
      masked_only_rgb_loss \
      "${GPU_MASKED}" \
      "nerfmae_masked_only_rgb_loss_p0.1_e30" \
      "../output/nerf_mae/results/nerfmae_masked_only_rgb_loss_p0.1_e30/epoch_30.pt" \
      "${masked_save}" \
      "${seed}" \
      "${LOG_ROOT}/${CHAIN_NAME}.${save_tag}.seed${seed}.masked.log" \
      "${rotate_prob}" \
      "${flip_prob}" \
      "${rot_scale_prob}" \
      "${lr_scheduler}" \
      "${scheduler_total_steps}" \
      "${freeze_backbone_epochs}" \
      "${backbone_lr_scale}"
    pids+=("${RUN_VARIANT_PID}")
    names+=("${phase_name}_masked_only_seed${seed}")
  fi

  if (( ${#pids[@]} > 0 )); then
    wait_for_pids pids names
  fi
  log "phase=${phase_name} seed=${seed} done"
}

run_scheduler_fixed_seed_sweep() {
  local seed
  local save_tag="sched_epoch"
  IFS=',' read -r -a seeds <<< "${SEED_LIST}"
  for seed in "${seeds[@]}"; do
    seed="${seed//[[:space:]]/}"
    [[ -n "${seed}" ]] || continue
    run_four_way_wave \
      scheduler_fixed \
      "${seed}" \
      "${save_tag}" \
      0.5 \
      0.5 \
      0.5 \
      "${FCOS_LR_SCHEDULER}" \
      "${FCOS_SCHEDULER_TOTAL_STEPS}" \
      0 \
      1.0
  done
}

run_noaug_diagnostic() {
  run_four_way_wave \
    deterministic_noaug \
    "${NOAUG_SEED}" \
    "sched_epoch_noaug" \
    0.0 \
    0.0 \
    0.0 \
    "${FCOS_LR_SCHEDULER}" \
    "${FCOS_SCHEDULER_TOTAL_STEPS}" \
    0 \
    1.0
}

run_alpha_shuffle_pretrain() {
  local seed="$1"
  local pretrain_save="nerfmae_alpha_shuffle_p0.1_e30_seed${seed}"
  local pretrain_epoch30="${ROOT_DIR}/output/nerf_mae/results/${pretrain_save}/epoch_30.pt"
  local pretrain_log="${LOG_ROOT}/${CHAIN_NAME}.alpha_shuffle.seed${seed}.pretrain.log"

  if [[ "${SKIP_EXISTING}" == "1" ]] && [[ -f "${pretrain_epoch30}" ]]; then
    log "skip alpha_shuffle pretrain seed=${seed} existing_ckpt=${pretrain_epoch30}"
    return 0
  fi

  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${ALPHA_SHUFFLE_PRETRAIN_GPU_IDS}" probe_wait_for_runway
  fi

  log "alpha_shuffle pretrain seed=${seed} start"
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
    SEED="${seed}" \
    DETERMINISTIC="${DETERMINISTIC}" \
    bash probe_scripts/train_alpha_shuffle.sh
  ) >> "${pretrain_log}" 2>&1

  [[ -f "${pretrain_epoch30}" ]] || probe_die "missing alpha_shuffle epoch_30 checkpoint ${pretrain_epoch30}"
  log "alpha_shuffle pretrain seed=${seed} done"
}

run_alpha_shuffle_fcos() {
  local pids=()
  local names=()
  local seeds=()
  local gpu_tokens=()
  local i=0
  local seed
  IFS=',' read -r -a seeds <<< "${ALPHA_SHUFFLE_SEEDS}"
  IFS=',' read -r -a gpu_tokens <<< "${ALPHA_SHUFFLE_FCOS_GPU_IDS}"

  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${ALPHA_SHUFFLE_FCOS_GPU_IDS}" probe_wait_for_runway
  fi

  for seed in "${seeds[@]}"; do
    seed="${seed//[[:space:]]/}"
    [[ -n "${seed}" ]] || continue
    local gpu_id="${gpu_tokens[$i]:-}"
    [[ -n "${gpu_id}" ]] || break

    local pretrain_save="nerfmae_alpha_shuffle_p0.1_e30_seed${seed}"
    local pretrain_ckpt="../output/nerf_mae/results/${pretrain_save}/epoch_30.pt"
    local save_name="nerfmae_alpha_shuffle_p0.1_e30_seed${seed}_epoch30_sched_epoch_seed${seed}_fcos${FCOS_NUM_EPOCHS}"
    local log_file="${LOG_ROOT}/${CHAIN_NAME}.alpha_shuffle.seed${seed}.fcos.log"

    if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${save_name}"; then
      log "skip alpha_shuffle fcos seed=${seed}"
    else
      run_job_probe \
        alpha_shuffle \
        "${gpu_id}" \
        "${pretrain_save}" \
        "${pretrain_ckpt}" \
        "${save_name}" \
        "${seed}" \
        "${log_file}" \
        0.5 \
        0.5 \
        0.5 \
        "${FCOS_LR_SCHEDULER}" \
        "${FCOS_SCHEDULER_TOTAL_STEPS}" \
        0 \
        1.0
      pids+=("${RUN_VARIANT_PID}")
      names+=("alpha_shuffle_seed${seed}")
    fi
    i=$((i + 1))
  done

  if (( ${#pids[@]} > 0 )); then
    wait_for_pids pids names
  fi
}

run_alpha_shuffle_multiseed() {
  local seed
  local seeds=()
  IFS=',' read -r -a seeds <<< "${ALPHA_SHUFFLE_SEEDS}"
  for seed in "${seeds[@]}"; do
    seed="${seed//[[:space:]]/}"
    [[ -n "${seed}" ]] || continue
    run_alpha_shuffle_pretrain "${seed}"
  done
  run_alpha_shuffle_fcos
}

run_freeze_backbone_diagnostic() {
  run_four_way_wave \
    freeze_backbone \
    "${FREEZE_SEED}" \
    "sched_epoch_freeze${FREEZE_BACKBONE_EPOCHS}" \
    0.5 \
    0.5 \
    0.5 \
    "${FCOS_LR_SCHEDULER}" \
    "${FCOS_SCHEDULER_TOTAL_STEPS}" \
    "${FREEZE_BACKBONE_EPOCHS}" \
    "${FREEZE_BACKBONE_LR_SCALE}"
}

log "start scheduler_fixed -> deterministic_noaug -> alpha_shuffle_multiseed -> freeze_backbone"
run_scheduler_fixed_seed_sweep
run_noaug_diagnostic
run_alpha_shuffle_multiseed
run_freeze_backbone_diagnostic
log "chain done"
