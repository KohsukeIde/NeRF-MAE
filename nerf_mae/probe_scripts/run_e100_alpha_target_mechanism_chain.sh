#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_probe_common.sh"

CHAIN_NAME="${CHAIN_NAME:-nerfmae_e100_alpha_target_mechanism}"
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

FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS:-100}"
FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU:-2}"
FCOS_LR="${FCOS_LR:-1e-4}"
FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY:-1e-3}"
FCOS_PERCENT_TRAIN_DEFAULT="${FCOS_PERCENT_TRAIN_DEFAULT:-1.0}"
FCOS_LR_SCHEDULER="${FCOS_LR_SCHEDULER:-onecycle_epoch}"

E100_SEEDS="${E100_SEEDS:-1,2,3}"
E100_PRETRAIN_GPU_IDS="${E100_PRETRAIN_GPU_IDS:-0,1,2,3}"
E100_PRETRAIN_EPOCHS="${E100_PRETRAIN_EPOCHS:-100}"
E100_PRETRAIN_PERCENT_TRAIN="${E100_PRETRAIN_PERCENT_TRAIN:-0.1}"
E100_PRETRAIN_BATCH_SIZE_PER_GPU="${E100_PRETRAIN_BATCH_SIZE_PER_GPU:-4}"
E100_FCOS_GPU_IDS="${E100_FCOS_GPU_IDS:-0,1,2,3}"

ALPHA_TARGET_SHUFFLE_SEEDS="${ALPHA_TARGET_SHUFFLE_SEEDS:-1,2,3}"
ALPHA_TARGET_SHUFFLE_PRETRAIN_GPU_IDS="${ALPHA_TARGET_SHUFFLE_PRETRAIN_GPU_IDS:-0,1,2,3}"
ALPHA_TARGET_SHUFFLE_PRETRAIN_EPOCHS="${ALPHA_TARGET_SHUFFLE_PRETRAIN_EPOCHS:-30}"
ALPHA_TARGET_SHUFFLE_PRETRAIN_PERCENT_TRAIN="${ALPHA_TARGET_SHUFFLE_PRETRAIN_PERCENT_TRAIN:-0.1}"
ALPHA_TARGET_SHUFFLE_PRETRAIN_BATCH_SIZE_PER_GPU="${ALPHA_TARGET_SHUFFLE_PRETRAIN_BATCH_SIZE_PER_GPU:-4}"
ALPHA_TARGET_SHUFFLE_FCOS_GPU_IDS="${ALPHA_TARGET_SHUFFLE_FCOS_GPU_IDS:-0,1,2}"

DIAG_GPU_IDS="${DIAG_GPU_IDS:-0,1,2,3}"
GT_BOXES_DIR="${GT_BOXES_DIR:-/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/dataset/finetune/front3d_rpn_data/obb}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/analysis}"

LOW_LABEL_SEED="${LOW_LABEL_SEED:-1}"
LOW_LABEL_PERCENTS="${LOW_LABEL_PERCENTS:-0.1,0.2}"
LOW_LABEL_GPU_IDS="${LOW_LABEL_GPU_IDS:-0,1,2,3}"

mkdir -p "${LOG_ROOT}" "${ANALYSIS_ROOT}"

if [[ "${FOREGROUND:-0}" != "1" ]]; then
  if env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
    echo "[info] chain already running in tmux session=${TMUX_SESSION}"
    echo "[info] log=${LOG_FILE}"
    exit 0
  fi
  env -u LD_LIBRARY_PATH -u PYTHONPATH tmux new-session -d -s "${TMUX_SESSION}" \
    "cd '${ROOT_DIR}' && FOREGROUND=1 bash '${SCRIPT_DIR}/run_e100_alpha_target_mechanism_chain.sh'"
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

run_job_scratch() {
  local gpu_ids="$1"
  local save_name="$2"
  local seed="$3"
  local percent_train="$4"
  local log_file="$5"
  (
    cd "${ROOT_DIR}"
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
    bash nerf_rpn/run_fcos_fair_scratch_variant.sh
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

parse_csv_to_array() {
  local csv="$1"
  local -n out_ref=$2
  IFS=',' read -r -a out_ref <<< "${csv}"
  local i
  for i in "${!out_ref[@]}"; do
    out_ref[$i]="${out_ref[$i]//[[:space:]]/}"
  done
}

percent_tag() {
  printf "pt%s" "${1//./}"
}

run_e100_pretrains() {
  local items=()
  local seeds=()
  parse_csv_to_array "${E100_SEEDS}" seeds
  local seed
  for seed in "${seeds[@]}"; do
    items+=("baseline:nerfmae_all_p0.1_e100_seed${seed}:train_mae3d.sh")
    items+=("alpha_only:nerfmae_alpha_only_p0.1_e100_seed${seed}:probe_scripts/train_alpha_only.sh")
    items+=("alpha_target_only:nerfmae_alpha_target_only_p0.1_e100_seed${seed}:probe_scripts/train_alpha_target_only.sh")
  done

  local item variant save_name script_rel
  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${E100_PRETRAIN_GPU_IDS}" probe_wait_for_runway
  fi
  log "step_e100_pretrains start"
  for item in "${items[@]}"; do
    IFS=':' read -r variant save_name script_rel <<< "${item}"
    if [[ "${SKIP_EXISTING}" == "1" ]] && pretrain_epoch_exists "${save_name}" 100; then
      log "skip e100 pretrain variant=${variant} save=${save_name}"
      continue
    fi
    log "e100 pretrain start variant=${variant} save=${save_name}"
    run_pretrain_script \
      "${save_name}" \
      "${E100_PRETRAIN_GPU_IDS}" \
      "${E100_PRETRAIN_EPOCHS}" \
      "${E100_PRETRAIN_PERCENT_TRAIN}" \
      "${E100_PRETRAIN_BATCH_SIZE_PER_GPU}" \
      "${save_name##*_seed}" \
      "${script_rel}" \
      "${LOG_ROOT}/${CHAIN_NAME}.e100.${variant}.${save_name##*_seed}.pretrain.log"
    log "e100 pretrain done variant=${variant} save=${save_name}"
  done
  log "step_e100_pretrains done"
}

run_e100_fcos() {
  local gpus=()
  parse_csv_to_array "${E100_FCOS_GPU_IDS}" gpus
  local jobs=()
  local seeds=()
  parse_csv_to_array "${E100_SEEDS}" seeds
  local seed
  for seed in "${seeds[@]}"; do
    jobs+=("baseline:nerfmae_all_p0.1_e100_seed${seed}:../output/nerf_mae/results/nerfmae_all_p0.1_e100_seed${seed}/epoch_100.pt:nerfmae_all_p0.1_e100_seed${seed}_epoch100_sched_epoch_seed${seed}_fcos${FCOS_NUM_EPOCHS}")
    jobs+=("alpha_only:nerfmae_alpha_only_p0.1_e100_seed${seed}:../output/nerf_mae/results/nerfmae_alpha_only_p0.1_e100_seed${seed}/epoch_100.pt:nerfmae_alpha_only_p0.1_e100_seed${seed}_epoch100_sched_epoch_seed${seed}_fcos${FCOS_NUM_EPOCHS}")
    jobs+=("alpha_target_only:nerfmae_alpha_target_only_p0.1_e100_seed${seed}:../output/nerf_mae/results/nerfmae_alpha_target_only_p0.1_e100_seed${seed}/epoch_100.pt:nerfmae_alpha_target_only_p0.1_e100_seed${seed}_epoch100_sched_epoch_seed${seed}_fcos${FCOS_NUM_EPOCHS}")
  done

  local pids=()
  local names=()
  local active=0
  local max_active="${#gpus[@]}"
  local gpu_idx=0
  local job variant gpu_id pretrain_save ckpt save_name seed_num
  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${E100_FCOS_GPU_IDS}" probe_wait_for_runway
  fi
  log "step_e100_fcos start"
  for job in "${jobs[@]}"; do
    IFS=':' read -r variant pretrain_save ckpt save_name <<< "${job}"
    if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${save_name}"; then
      log "skip e100 fcos variant=${variant} save=${save_name}"
      continue
    fi
    gpu_id="${gpus[$gpu_idx]}"
    gpu_idx=$(((gpu_idx + 1) % max_active))
    seed_num="${pretrain_save##*_seed}"
    run_job_probe \
      "${variant}" \
      "${gpu_id}" \
      "${pretrain_save}" \
      "${ckpt}" \
      "${save_name}" \
      "${seed_num}" \
      "${FCOS_PERCENT_TRAIN_DEFAULT}" \
      "${LOG_ROOT}/${CHAIN_NAME}.e100.${variant}.${seed_num}.fcos.log"
    pids+=("${RUN_VARIANT_PID}")
    names+=("e100_${variant}_seed${seed_num}")
    active=$((active + 1))
    if (( active >= max_active )); then
      wait_for_pids pids names
      pids=()
      names=()
      active=0
    fi
  done
  if (( ${#pids[@]} > 0 )); then
    wait_for_pids pids names
  fi
  log "step_e100_fcos done"
}

run_alpha_target_shuffle_pretrains() {
  local seeds=()
  parse_csv_to_array "${ALPHA_TARGET_SHUFFLE_SEEDS}" seeds
  local seed save_name
  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${ALPHA_TARGET_SHUFFLE_PRETRAIN_GPU_IDS}" probe_wait_for_runway
  fi
  log "step_alpha_target_shuffle_pretrains start"
  for seed in "${seeds[@]}"; do
    save_name="nerfmae_alpha_target_shuffle_p0.1_e30_seed${seed}"
    if [[ "${SKIP_EXISTING}" == "1" ]] && pretrain_epoch_exists "${save_name}" 30; then
      log "skip alpha_target_shuffle pretrain seed=${seed}"
      continue
    fi
    log "alpha_target_shuffle pretrain start seed=${seed}"
    run_pretrain_script \
      "${save_name}" \
      "${ALPHA_TARGET_SHUFFLE_PRETRAIN_GPU_IDS}" \
      "${ALPHA_TARGET_SHUFFLE_PRETRAIN_EPOCHS}" \
      "${ALPHA_TARGET_SHUFFLE_PRETRAIN_PERCENT_TRAIN}" \
      "${ALPHA_TARGET_SHUFFLE_PRETRAIN_BATCH_SIZE_PER_GPU}" \
      "${seed}" \
      "probe_scripts/train_alpha_target_shuffle.sh" \
      "${LOG_ROOT}/${CHAIN_NAME}.alpha_target_shuffle.${seed}.pretrain.log"
    log "alpha_target_shuffle pretrain done seed=${seed}"
  done
  log "step_alpha_target_shuffle_pretrains done"
}

run_alpha_target_shuffle_fcos_and_diagnostics() {
  local fcos_gpus=()
  parse_csv_to_array "${ALPHA_TARGET_SHUFFLE_FCOS_GPU_IDS}" fcos_gpus
  local diag_gpus=()
  parse_csv_to_array "${DIAG_GPU_IDS}" diag_gpus
  local pids=()
  local names=()
  local seed save_name pretrain_save ckpt

  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${DIAG_GPU_IDS}" probe_wait_for_runway
  fi
  log "step_alpha_target_shuffle_fcos_and_diagnostics start"

  for seed in 1 2 3; do
    pretrain_save="nerfmae_alpha_target_shuffle_p0.1_e30_seed${seed}"
    save_name="${pretrain_save}_epoch30_sched_epoch_seed${seed}_fcos${FCOS_NUM_EPOCHS}"
    if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${save_name}"; then
      log "skip alpha_target_shuffle fcos seed=${seed}"
      continue
    fi
    run_job_probe \
      alpha_target_shuffle \
      "${fcos_gpus[$((seed - 1))]}" \
      "${pretrain_save}" \
      "../output/nerf_mae/results/${pretrain_save}/epoch_30.pt" \
      "${save_name}" \
      "${seed}" \
      "${FCOS_PERCENT_TRAIN_DEFAULT}" \
      "${LOG_ROOT}/${CHAIN_NAME}.alpha_target_shuffle.${seed}.fcos.log"
    pids+=("${RUN_VARIANT_PID}")
    names+=("alpha_target_shuffle_seed${seed}")
  done

  if ! { [[ "${SKIP_EXISTING}" == "1" ]] && diag_exists "nerfmae_all_p0.1_e100_seed1_epoch100_sched_epoch_seed1_diagnostics"; }; then
    ckpt="$(find_latest_checkpoint "${ROOT_DIR}/output/nerf_rpn/results/nerfmae_all_p0.1_e100_seed1_epoch100_sched_epoch_seed1_fcos100")"
    run_job_diagnostic \
      baseline_e100_seed1 \
      "${diag_gpus[3]:-${diag_gpus[0]}}" \
      "${ckpt}" \
      "nerfmae_all_p0.1_e100_seed1_epoch100_sched_epoch_seed1_diagnostics" \
      "${LOG_ROOT}/${CHAIN_NAME}.diag.baseline_e100_seed1.log"
    pids+=("${RUN_VARIANT_PID}")
    names+=("diag_baseline_e100_seed1")
  fi

  if (( ${#pids[@]} > 0 )); then
    wait_for_pids pids names
  fi
  log "step_alpha_target_shuffle_fcos_and_diagnostics done"
}

run_remaining_diagnostics_and_summary() {
  local pids=()
  local names=()
  local ckpt

  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${DIAG_GPU_IDS}" probe_wait_for_runway
  fi
  log "step_remaining_diagnostics start"

  if ! { [[ "${SKIP_EXISTING}" == "1" ]] && diag_exists "nerfmae_alpha_target_only_p0.1_e100_seed1_epoch100_sched_epoch_seed1_diagnostics"; }; then
    ckpt="$(find_latest_checkpoint "${ROOT_DIR}/output/nerf_rpn/results/nerfmae_alpha_target_only_p0.1_e100_seed1_epoch100_sched_epoch_seed1_fcos100")"
    run_job_diagnostic \
      alpha_target_only_e100_seed1 \
      "0" \
      "${ckpt}" \
      "nerfmae_alpha_target_only_p0.1_e100_seed1_epoch100_sched_epoch_seed1_diagnostics" \
      "${LOG_ROOT}/${CHAIN_NAME}.diag.alpha_target_e100_seed1.log"
    pids+=("${RUN_VARIANT_PID}")
    names+=("diag_alpha_target_e100_seed1")
  fi

  if ! { [[ "${SKIP_EXISTING}" == "1" ]] && diag_exists "nerfmae_alpha_target_shuffle_p0.1_e30_seed1_epoch30_sched_epoch_seed1_diagnostics"; }; then
    ckpt="$(find_latest_checkpoint "${ROOT_DIR}/output/nerf_rpn/results/nerfmae_alpha_target_shuffle_p0.1_e30_seed1_epoch30_sched_epoch_seed1_fcos100")"
    run_job_diagnostic \
      alpha_target_shuffle_e30_seed1 \
      "1" \
      "${ckpt}" \
      "nerfmae_alpha_target_shuffle_p0.1_e30_seed1_epoch30_sched_epoch_seed1_diagnostics" \
      "${LOG_ROOT}/${CHAIN_NAME}.diag.alpha_target_shuffle_seed1.log"
    pids+=("${RUN_VARIANT_PID}")
    names+=("diag_alpha_target_shuffle_seed1")
  fi

  if (( ${#pids[@]} > 0 )); then
    wait_for_pids pids names
  fi

  local output_json="${ANALYSIS_ROOT}/${CHAIN_NAME}_diagnostics_summary.json"
  local output_md="${ANALYSIS_ROOT}/${CHAIN_NAME}_diagnostics_summary.md"
  if [[ "${SKIP_EXISTING}" == "1" ]] && [[ -f "${output_json}" ]] && [[ -f "${output_md}" ]]; then
    log "skip diagnostic summary existing outputs detected"
  else
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="${ROOT_DIR}/nerf_rpn:${ROOT_DIR}" \
    "${PROBE_PYTHON_BIN}" "${ROOT_DIR}/nerf_rpn/tools/summarize_diagnostic_dumps.py" \
      --diagnostic "scratch_e30_seed1=${ROOT_DIR}/output/nerf_rpn/results/front3d_scratch_samepath_sched_epoch_seed1_diagnostics" \
      --diagnostic "baseline_e100_seed1=${ROOT_DIR}/output/nerf_rpn/results/nerfmae_all_p0.1_e100_seed1_epoch100_sched_epoch_seed1_diagnostics" \
      --diagnostic "alpha_target_e100_seed1=${ROOT_DIR}/output/nerf_rpn/results/nerfmae_alpha_target_only_p0.1_e100_seed1_epoch100_sched_epoch_seed1_diagnostics" \
      --diagnostic "alpha_target_shuffle_e30_seed1=${ROOT_DIR}/output/nerf_rpn/results/nerfmae_alpha_target_shuffle_p0.1_e30_seed1_epoch30_sched_epoch_seed1_diagnostics" \
      --gt_boxes_dir "${GT_BOXES_DIR}" \
      --output_json "${output_json}" \
      --output_md "${output_md}" \
      >> "${LOG_ROOT}/${CHAIN_NAME}.diag.summary.log" 2>&1
    log "diagnostic summary done output_md=${output_md}"
  fi
  log "step_remaining_diagnostics done"
}

run_low_label_fcos() {
  local percents=()
  local gpus=()
  parse_csv_to_array "${LOW_LABEL_PERCENTS}" percents
  parse_csv_to_array "${LOW_LABEL_GPU_IDS}" gpus
  local jobs=()
  local percent tag
  for percent in "${percents[@]}"; do
    tag="$(percent_tag "${percent}")"
    jobs+=("scratch::front3d_scratch_samepath_sched_epoch_${tag}_seed${LOW_LABEL_SEED}_fcos${FCOS_NUM_EPOCHS}:${percent}")
    jobs+=("baseline:nerfmae_all_p0.1_e100_seed1:nerfmae_all_p0.1_e100_seed1_epoch100_sched_epoch_${tag}_seed${LOW_LABEL_SEED}_fcos${FCOS_NUM_EPOCHS}:${percent}")
    jobs+=("alpha_target_only:nerfmae_alpha_target_only_p0.1_e100_seed1:nerfmae_alpha_target_only_p0.1_e100_seed1_epoch100_sched_epoch_${tag}_seed${LOW_LABEL_SEED}_fcos${FCOS_NUM_EPOCHS}:${percent}")
  done

  local pids=()
  local names=()
  local active=0
  local max_active="${#gpus[@]}"
  local gpu_idx=0
  local job variant gpu_id pretrain_save save_name percent

  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${LOW_LABEL_GPU_IDS}" probe_wait_for_runway
  fi
  log "step_low_label_fcos start"
  for job in "${jobs[@]}"; do
    IFS=':' read -r variant pretrain_save save_name percent <<< "${job}"
    if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${save_name}"; then
      log "skip low_label variant=${variant} save=${save_name}"
      continue
    fi
    gpu_id="${gpus[$gpu_idx]}"
    gpu_idx=$(((gpu_idx + 1) % max_active))
    if [[ "${variant}" == "scratch" ]]; then
      run_job_scratch \
        "${gpu_id}" \
        "${save_name}" \
        "${LOW_LABEL_SEED}" \
        "${percent}" \
        "${LOG_ROOT}/${CHAIN_NAME}.lowlabel.${variant}.$(percent_tag "${percent}").log"
    else
      run_job_probe \
        "${variant}" \
        "${gpu_id}" \
        "${pretrain_save}" \
        "../output/nerf_mae/results/${pretrain_save}/epoch_100.pt" \
        "${save_name}" \
        "${LOW_LABEL_SEED}" \
        "${percent}" \
        "${LOG_ROOT}/${CHAIN_NAME}.lowlabel.${variant}.$(percent_tag "${percent}").log"
    fi
    pids+=("${RUN_VARIANT_PID}")
    names+=("lowlabel_${variant}_$(percent_tag "${percent}")")
    active=$((active + 1))
    if (( active >= max_active )); then
      wait_for_pids pids names
      pids=()
      names=()
      active=0
    fi
  done
  if (( ${#pids[@]} > 0 )); then
    wait_for_pids pids names
  fi
  log "step_low_label_fcos done"
}

log "start e100_multi_seed -> alpha_target_shuffle -> diagnostics -> low_label"
run_e100_pretrains
run_e100_fcos
run_alpha_target_shuffle_pretrains
run_alpha_target_shuffle_fcos_and_diagnostics
run_remaining_diagnostics_and_summary
run_low_label_fcos
log "chain done"
