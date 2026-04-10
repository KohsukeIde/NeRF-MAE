#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_probe_common.sh"

CHAIN_NAME="${CHAIN_NAME:-nerfmae_alpha_target_followup}"
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
FCOS_PERCENT_TRAIN="${FCOS_PERCENT_TRAIN:-1.0}"
FCOS_LR_SCHEDULER="${FCOS_LR_SCHEDULER:-onecycle_epoch}"

DIAG_GPU_IDS="${DIAG_GPU_IDS:-0,1,2,3}"
DIAG_SUMMARY_GPU="${DIAG_SUMMARY_GPU:-0}"
GT_BOXES_DIR="${GT_BOXES_DIR:-/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/dataset/finetune/front3d_rpn_data/obb}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/analysis}"

ALPHA_TARGET_SEEDS="${ALPHA_TARGET_SEEDS:-1,2,3}"
ALPHA_TARGET_PRETRAIN_GPU_IDS="${ALPHA_TARGET_PRETRAIN_GPU_IDS:-0,1,2,3}"
ALPHA_TARGET_PRETRAIN_GPU_IDS_MULTISEED="${ALPHA_TARGET_PRETRAIN_GPU_IDS_MULTISEED:-0}"
ALPHA_TARGET_PRETRAIN_EPOCHS="${ALPHA_TARGET_PRETRAIN_EPOCHS:-30}"
ALPHA_TARGET_PRETRAIN_PERCENT_TRAIN="${ALPHA_TARGET_PRETRAIN_PERCENT_TRAIN:-0.1}"
ALPHA_TARGET_PRETRAIN_BATCH_SIZE_PER_GPU="${ALPHA_TARGET_PRETRAIN_BATCH_SIZE_PER_GPU:-4}"
ALPHA_TARGET_MULTI_SEED_POLICY="${ALPHA_TARGET_MULTI_SEED_POLICY:-auto}"
ALPHA_TARGET_AUTO_MARGIN="${ALPHA_TARGET_AUTO_MARGIN:-0.03}"
ALPHA_TARGET_FCOS_GPU_SEED1="${ALPHA_TARGET_FCOS_GPU_SEED1:-0}"
ALPHA_TARGET_FCOS_GPU_SEED2="${ALPHA_TARGET_FCOS_GPU_SEED2:-0}"
ALPHA_TARGET_FCOS_GPU_SEED3="${ALPHA_TARGET_FCOS_GPU_SEED3:-1}"

HEAVY_PRETRAIN_EPOCHS="${HEAVY_PRETRAIN_EPOCHS:-100}"
HEAVY_PRETRAIN_PERCENT_TRAIN="${HEAVY_PRETRAIN_PERCENT_TRAIN:-0.1}"
HEAVY_PRETRAIN_BATCH_SIZE_PER_GPU="${HEAVY_PRETRAIN_BATCH_SIZE_PER_GPU:-4}"
HEAVY_PRETRAIN_GPU_IDS="${HEAVY_PRETRAIN_GPU_IDS:-0,1,2,3}"
HEAVY_PRETRAIN_GPU_IDS_OVERLAP="${HEAVY_PRETRAIN_GPU_IDS_OVERLAP:-1,2,3}"
HEAVY_SEED="${HEAVY_SEED:-1}"

SCANNET_DATA_ROOT="${SCANNET_DATA_ROOT:-/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/dataset/finetune/scannet_rpn_data}"

mkdir -p "${LOG_ROOT}" "${ANALYSIS_ROOT}"

HEAVY_PRETRAIN_BG_PID=""
HEAVY_PRETRAIN_BG_STARTED=0

if [[ "${FOREGROUND:-0}" != "1" ]]; then
  if env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
    echo "[info] chain already running in tmux session=${TMUX_SESSION}"
    echo "[info] log=${LOG_FILE}"
    exit 0
  fi
  env -u LD_LIBRARY_PATH -u PYTHONPATH tmux new-session -d -s "${TMUX_SESSION}" \
    "cd '${ROOT_DIR}' && FOREGROUND=1 bash '${SCRIPT_DIR}/run_alpha_target_followup_chain.sh'"
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

find_latest_checkpoint() {
  local dir="$1"
  find "${dir}" -maxdepth 1 -name 'model_best_ap50_ap25_*.pt' -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | awk '{print $2}'
}

eval_exists() {
  local save_name="$1"
  [[ -f "${ROOT_DIR}/output/nerf_rpn/results/${save_name}_eval/eval.json" ]]
}

pretrain_epoch_exists() {
  local save_name="$1"
  local epoch="${2:-30}"
  [[ -f "${ROOT_DIR}/output/nerf_mae/results/${save_name}/epoch_${epoch}.pt" ]]
}

diag_exists() {
  local save_name="$1"
  [[ -f "${ROOT_DIR}/output/nerf_rpn/results/${save_name}/eval.json" ]]
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
    LR_SCHEDULER="${FCOS_LR_SCHEDULER}" \
    bash nerf_rpn/run_fcos_probe_variant.sh
  ) >> "${log_file}" 2>&1 &
  RUN_VARIANT_PID=$!
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

run_scheduler_fixed_diagnostics() {
  local pids=()
  local names=()
  local gpu_array=()
  IFS=',' read -r -a gpu_array <<< "${DIAG_GPU_IDS}"
  (( ${#gpu_array[@]} >= 4 )) || probe_die "DIAG_GPU_IDS must provide at least 4 GPUs"

  local scratch_dir="${ROOT_DIR}/output/nerf_rpn/results/front3d_scratch_samepath_sched_epoch_seed1_fcos100"
  local baseline_dir="${ROOT_DIR}/output/nerf_rpn/results/nerfmae_all_p0.1_e30_epoch30_sched_epoch_seed1_fcos100"
  local alpha_dir="${ROOT_DIR}/output/nerf_rpn/results/nerfmae_alpha_only_p0.1_e30_epoch30_sched_epoch_seed1_fcos100"
  local shuffle_dir="${ROOT_DIR}/output/nerf_rpn/results/nerfmae_alpha_shuffle_p0.1_e30_seed1_epoch30_sched_epoch_seed1_fcos100"
  local masked_dir="${ROOT_DIR}/output/nerf_rpn/results/nerfmae_masked_only_rgb_loss_p0.1_e30_epoch30_sched_epoch_seed1_fcos100"

  local scratch_diag="front3d_scratch_samepath_sched_epoch_seed1_diagnostics"
  local baseline_diag="nerfmae_all_p0.1_e30_epoch30_sched_epoch_seed1_diagnostics"
  local alpha_diag="nerfmae_alpha_only_p0.1_e30_epoch30_sched_epoch_seed1_diagnostics"
  local shuffle_diag="nerfmae_alpha_shuffle_p0.1_e30_seed1_epoch30_sched_epoch_seed1_diagnostics"
  local masked_diag="nerfmae_masked_only_rgb_loss_p0.1_e30_epoch30_sched_epoch_seed1_diagnostics"

  if [[ "${SKIP_EXISTING}" == "1" ]] \
    && diag_exists "${scratch_diag}" \
    && diag_exists "${baseline_diag}" \
    && diag_exists "${alpha_diag}" \
    && diag_exists "${shuffle_diag}" \
    && diag_exists "${masked_diag}"; then
    log "skip scheduler-fixed diagnostics existing outputs detected"
    return 0
  fi

  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${DIAG_GPU_IDS}" probe_wait_for_runway
  fi

  log "step_diagnostics_sched_fixed start"

  if ! { [[ "${SKIP_EXISTING}" == "1" ]] && diag_exists "${scratch_diag}"; }; then
    run_job_diagnostic \
      scratch_sched_epoch_seed1 \
      "${gpu_array[0]//[[:space:]]/}" \
      "$(find_latest_checkpoint "${scratch_dir}")" \
      "${scratch_diag}" \
      "${LOG_ROOT}/${CHAIN_NAME}.diag.scratch.log"
    pids+=("${RUN_VARIANT_PID}")
    names+=("diag_scratch")
  fi
  if ! { [[ "${SKIP_EXISTING}" == "1" ]] && diag_exists "${baseline_diag}"; }; then
    run_job_diagnostic \
      baseline_sched_epoch_seed1 \
      "${gpu_array[1]//[[:space:]]/}" \
      "$(find_latest_checkpoint "${baseline_dir}")" \
      "${baseline_diag}" \
      "${LOG_ROOT}/${CHAIN_NAME}.diag.baseline.log"
    pids+=("${RUN_VARIANT_PID}")
    names+=("diag_baseline")
  fi
  if ! { [[ "${SKIP_EXISTING}" == "1" ]] && diag_exists "${alpha_diag}"; }; then
    run_job_diagnostic \
      alpha_only_sched_epoch_seed1 \
      "${gpu_array[2]//[[:space:]]/}" \
      "$(find_latest_checkpoint "${alpha_dir}")" \
      "${alpha_diag}" \
      "${LOG_ROOT}/${CHAIN_NAME}.diag.alpha.log"
    pids+=("${RUN_VARIANT_PID}")
    names+=("diag_alpha")
  fi
  if ! { [[ "${SKIP_EXISTING}" == "1" ]] && diag_exists "${shuffle_diag}"; }; then
    run_job_diagnostic \
      alpha_shuffle_sched_epoch_seed1 \
      "${gpu_array[3]//[[:space:]]/}" \
      "$(find_latest_checkpoint "${shuffle_dir}")" \
      "${shuffle_diag}" \
      "${LOG_ROOT}/${CHAIN_NAME}.diag.shuffle.log"
    pids+=("${RUN_VARIANT_PID}")
    names+=("diag_shuffle")
  fi

  if (( ${#pids[@]} > 0 )); then
    wait_for_pids pids names
  fi

  if ! { [[ "${SKIP_EXISTING}" == "1" ]] && diag_exists "${masked_diag}"; }; then
    if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
      GPU_IDS="${gpu_array[0]//[[:space:]]/}" probe_wait_for_runway
    fi
    run_job_diagnostic \
      masked_only_sched_epoch_seed1 \
      "${gpu_array[0]//[[:space:]]/}" \
      "$(find_latest_checkpoint "${masked_dir}")" \
      "${masked_diag}" \
      "${LOG_ROOT}/${CHAIN_NAME}.diag.masked.log"
    local masked_pid="${RUN_VARIANT_PID}"
    if wait "${masked_pid}"; then
      log "job_done name=diag_masked"
    else
      probe_die "diagnostic masked_only failed"
    fi
  fi

  log "step_diagnostics_sched_fixed done"
}

run_diagnostic_summary() {
  local output_json="${ANALYSIS_ROOT}/sched_epoch_seed1_diagnostics_summary.json"
  local output_md="${ANALYSIS_ROOT}/sched_epoch_seed1_diagnostics_summary.md"
  if [[ "${SKIP_EXISTING}" == "1" ]] && [[ -f "${output_json}" ]] && [[ -f "${output_md}" ]]; then
    log "skip diagnostic summary existing outputs detected"
    return 0
  fi
  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${DIAG_SUMMARY_GPU}" probe_wait_for_runway
  fi
  log "step_diagnostic_summary start"
  (
    cd "${ROOT_DIR}"
    CUDA_VISIBLE_DEVICES="${DIAG_SUMMARY_GPU}" \
    PATH="${PROBE_ENV_PREFIX}/bin:${PATH}" \
    PYTHONPATH="${ROOT_DIR}/nerf_rpn:${ROOT_DIR}" \
    python nerf_rpn/tools/summarize_diagnostic_dumps.py \
      --diagnostic "fair_scratch=${ROOT_DIR}/output/nerf_rpn/results/front3d_scratch_samepath_sched_epoch_seed1_diagnostics" \
      --diagnostic "baseline=${ROOT_DIR}/output/nerf_rpn/results/nerfmae_all_p0.1_e30_epoch30_sched_epoch_seed1_diagnostics" \
      --diagnostic "alpha_only=${ROOT_DIR}/output/nerf_rpn/results/nerfmae_alpha_only_p0.1_e30_epoch30_sched_epoch_seed1_diagnostics" \
      --diagnostic "alpha_shuffle=${ROOT_DIR}/output/nerf_rpn/results/nerfmae_alpha_shuffle_p0.1_e30_seed1_epoch30_sched_epoch_seed1_diagnostics" \
      --diagnostic "masked_only=${ROOT_DIR}/output/nerf_rpn/results/nerfmae_masked_only_rgb_loss_p0.1_e30_epoch30_sched_epoch_seed1_diagnostics" \
      --gt_boxes_dir "${GT_BOXES_DIR}" \
      --output_json "${output_json}" \
      --output_md "${output_md}" \
      --device cuda:0
  ) >> "${LOG_ROOT}/${CHAIN_NAME}.diag.summary.log" 2>&1
  log "step_diagnostic_summary done output_md=${output_md}"
}

run_alpha_target_pretrain_seed() {
  local seed="$1"
  local gpu_ids="${2:-${ALPHA_TARGET_PRETRAIN_GPU_IDS}}"
  local save_name="nerfmae_alpha_target_only_p0.1_e30_seed${seed}"
  local log_file="${LOG_ROOT}/${CHAIN_NAME}.alpha_target.seed${seed}.pretrain.log"
  if [[ "${SKIP_EXISTING}" == "1" ]] && pretrain_epoch_exists "${save_name}" 30; then
    log "skip alpha_target pretrain seed=${seed}"
    return 0
  fi
  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${gpu_ids}" probe_wait_for_runway
  fi
  log "alpha_target pretrain seed=${seed} start"
  run_pretrain_script \
    "${save_name}" \
    "${gpu_ids}" \
    "${ALPHA_TARGET_PRETRAIN_EPOCHS}" \
    "${ALPHA_TARGET_PRETRAIN_PERCENT_TRAIN}" \
    "${ALPHA_TARGET_PRETRAIN_BATCH_SIZE_PER_GPU}" \
    "${seed}" \
    "probe_scripts/train_alpha_target_only.sh" \
    "${log_file}"
  log "alpha_target pretrain seed=${seed} done"
}

run_alpha_target_fcos_seed() {
  local seed="$1"
  local gpu_id="$2"
  local pretrain_save="nerfmae_alpha_target_only_p0.1_e30_seed${seed}"
  local save_name="${pretrain_save}_epoch30_sched_epoch_seed${seed}_fcos${FCOS_NUM_EPOCHS}"
  local log_file="${LOG_ROOT}/${CHAIN_NAME}.alpha_target.seed${seed}.fcos.log"
  if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${save_name}"; then
    log "skip alpha_target fcos seed=${seed}"
    return 0
  fi
  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="${gpu_id}" probe_wait_for_runway
  fi
  run_job_probe \
    alpha_target_only \
    "${gpu_id}" \
    "${pretrain_save}" \
    "../output/nerf_mae/results/${pretrain_save}/epoch_30.pt" \
    "${save_name}" \
    "${seed}" \
    "${log_file}"
  local pid="${RUN_VARIANT_PID}"
  if wait "${pid}"; then
    log "job_done name=alpha_target_fcos_seed${seed}"
  else
    probe_die "alpha_target fcos failed seed=${seed}"
  fi
}

should_run_alpha_target_multiseed() {
  case "${ALPHA_TARGET_MULTI_SEED_POLICY}" in
    always)
      return 0
      ;;
    never)
      return 1
      ;;
    auto)
      ;;
    *)
      probe_die "unknown ALPHA_TARGET_MULTI_SEED_POLICY=${ALPHA_TARGET_MULTI_SEED_POLICY}"
      ;;
  esac

  local alpha_target_eval="${ROOT_DIR}/output/nerf_rpn/results/nerfmae_alpha_target_only_p0.1_e30_seed1_epoch30_sched_epoch_seed1_fcos${FCOS_NUM_EPOCHS}_eval/eval.json"
  local scratch_eval="${ROOT_DIR}/output/nerf_rpn/results/front3d_scratch_samepath_sched_epoch_seed1_fcos${FCOS_NUM_EPOCHS}_eval/eval.json"
  local alpha_eval="${ROOT_DIR}/output/nerf_rpn/results/nerfmae_alpha_only_p0.1_e30_epoch30_sched_epoch_seed1_fcos${FCOS_NUM_EPOCHS}_eval/eval.json"

  [[ -f "${alpha_target_eval}" ]] || probe_die "missing alpha_target seed1 eval"
  [[ -f "${scratch_eval}" ]] || probe_die "missing scratch seed1 eval"
  [[ -f "${alpha_eval}" ]] || probe_die "missing alpha_only seed1 eval"

  local decision
  decision="$(
    python - <<PY
import json
margin = float(${ALPHA_TARGET_AUTO_MARGIN})
paths = {
    "alpha_target": "${alpha_target_eval}",
    "scratch": "${scratch_eval}",
    "alpha_only": "${alpha_eval}",
}
vals = {}
for key, path in paths.items():
    with open(path) as f:
        vals[key] = json.load(f)["ap_50"]["ap"]
reference = max(vals["scratch"], vals["alpha_only"])
print("run" if vals["alpha_target"] >= reference - margin else "skip")
print(f"alpha_target={vals['alpha_target']:.4f} reference={reference:.4f} margin={margin:.4f}")
PY
  )"
  local first_line="${decision%%$'\n'*}"
  local second_line="${decision#*$'\n'}"
  log "alpha_target multi-seed auto decision ${second_line}"
  [[ "${first_line}" == "run" ]]
}

run_alpha_target_sequence() {
  log "step_alpha_target start"
  run_alpha_target_pretrain_seed 1
  start_overlapped_heavy_pretrains
  run_alpha_target_fcos_seed 1 "${ALPHA_TARGET_FCOS_GPU_SEED1}"

  if should_run_alpha_target_multiseed; then
    run_alpha_target_pretrain_seed 2 "${ALPHA_TARGET_PRETRAIN_GPU_IDS_MULTISEED}"
    run_alpha_target_pretrain_seed 3 "${ALPHA_TARGET_PRETRAIN_GPU_IDS_MULTISEED}"

    local pids=()
    local names=()
    for seed in 2 3; do
      local gpu_var="ALPHA_TARGET_FCOS_GPU_SEED${seed}"
      local gpu_id="${!gpu_var}"
      local pretrain_save="nerfmae_alpha_target_only_p0.1_e30_seed${seed}"
      local save_name="${pretrain_save}_epoch30_sched_epoch_seed${seed}_fcos${FCOS_NUM_EPOCHS}"
      local log_file="${LOG_ROOT}/${CHAIN_NAME}.alpha_target.seed${seed}.fcos.log"
      if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${save_name}"; then
        log "skip alpha_target fcos seed=${seed}"
        continue
      fi
      run_job_probe \
        alpha_target_only \
        "${gpu_id}" \
        "${pretrain_save}" \
        "../output/nerf_mae/results/${pretrain_save}/epoch_30.pt" \
        "${save_name}" \
        "${seed}" \
        "${log_file}"
      pids+=("${RUN_VARIANT_PID}")
      names+=("alpha_target_fcos_seed${seed}")
    done
    if (( ${#pids[@]} > 0 )); then
      wait_for_pids pids names
    fi
  else
    log "alpha_target multi-seed skipped by policy=${ALPHA_TARGET_MULTI_SEED_POLICY}"
  fi
  log "step_alpha_target done"
}

run_heavy_pretrains() {
  local items=(
    "baseline:nerfmae_all_p0.1_e100_seed${HEAVY_SEED}:train_mae3d.sh"
    "alpha_only:nerfmae_alpha_only_p0.1_e100_seed${HEAVY_SEED}:probe_scripts/train_alpha_only.sh"
    "alpha_shuffle:nerfmae_alpha_shuffle_p0.1_e100_seed${HEAVY_SEED}:probe_scripts/train_alpha_shuffle.sh"
    "alpha_target_only:nerfmae_alpha_target_only_p0.1_e100_seed${HEAVY_SEED}:probe_scripts/train_alpha_target_only.sh"
  )
  local item variant save_name script_rel
  for item in "${items[@]}"; do
    IFS=':' read -r variant save_name script_rel <<< "${item}"
    if [[ "${SKIP_EXISTING}" == "1" ]] && pretrain_epoch_exists "${save_name}" 100; then
      log "skip heavy pretrain variant=${variant} save=${save_name}"
      continue
    fi
    if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
      GPU_IDS="${HEAVY_PRETRAIN_GPU_IDS}" probe_wait_for_runway
    fi
    log "heavy pretrain start variant=${variant} save=${save_name}"
    run_pretrain_script \
      "${save_name}" \
      "${HEAVY_PRETRAIN_GPU_IDS}" \
      "${HEAVY_PRETRAIN_EPOCHS}" \
      "${HEAVY_PRETRAIN_PERCENT_TRAIN}" \
      "${HEAVY_PRETRAIN_BATCH_SIZE_PER_GPU}" \
      "${HEAVY_SEED}" \
      "${script_rel}" \
      "${LOG_ROOT}/${CHAIN_NAME}.heavy.${variant}.pretrain.log"
    log "heavy pretrain done variant=${variant} save=${save_name}"
  done
}

start_overlapped_heavy_pretrains() {
  if (( HEAVY_PRETRAIN_BG_STARTED == 1 )); then
    return 0
  fi
  HEAVY_PRETRAIN_BG_STARTED=1
  (
    HEAVY_PRETRAIN_GPU_IDS="${HEAVY_PRETRAIN_GPU_IDS_OVERLAP}"
    run_heavy_pretrains
  ) >> "${LOG_ROOT}/${CHAIN_NAME}.heavy.pretrains.overlap.log" 2>&1 &
  HEAVY_PRETRAIN_BG_PID=$!
  log "heavy pretrains overlap started pid=${HEAVY_PRETRAIN_BG_PID} gpus=${HEAVY_PRETRAIN_GPU_IDS_OVERLAP}"
}

wait_for_overlapped_heavy_pretrains() {
  if (( HEAVY_PRETRAIN_BG_STARTED == 0 )); then
    return 0
  fi
  if [[ -z "${HEAVY_PRETRAIN_BG_PID}" ]]; then
    return 0
  fi
  if wait "${HEAVY_PRETRAIN_BG_PID}"; then
    log "heavy pretrains overlap done"
  else
    probe_die "overlapped heavy pretrains failed"
  fi
  HEAVY_PRETRAIN_BG_PID=""
}

run_heavy_fcos() {
  local pids=()
  local names=()
  local jobs=(
    "baseline:1:nerfmae_all_p0.1_e100_seed${HEAVY_SEED}:../output/nerf_mae/results/nerfmae_all_p0.1_e100_seed${HEAVY_SEED}/epoch_100.pt:nerfmae_all_p0.1_e100_seed${HEAVY_SEED}_epoch100_sched_epoch_seed${HEAVY_SEED}_fcos${FCOS_NUM_EPOCHS}"
    "alpha_only:2:nerfmae_alpha_only_p0.1_e100_seed${HEAVY_SEED}:../output/nerf_mae/results/nerfmae_alpha_only_p0.1_e100_seed${HEAVY_SEED}/epoch_100.pt:nerfmae_alpha_only_p0.1_e100_seed${HEAVY_SEED}_epoch100_sched_epoch_seed${HEAVY_SEED}_fcos${FCOS_NUM_EPOCHS}"
    "alpha_shuffle:3:nerfmae_alpha_shuffle_p0.1_e100_seed${HEAVY_SEED}:../output/nerf_mae/results/nerfmae_alpha_shuffle_p0.1_e100_seed${HEAVY_SEED}/epoch_100.pt:nerfmae_alpha_shuffle_p0.1_e100_seed${HEAVY_SEED}_epoch100_sched_epoch_seed${HEAVY_SEED}_fcos${FCOS_NUM_EPOCHS}"
    "alpha_target_only:0:nerfmae_alpha_target_only_p0.1_e100_seed${HEAVY_SEED}:../output/nerf_mae/results/nerfmae_alpha_target_only_p0.1_e100_seed${HEAVY_SEED}/epoch_100.pt:nerfmae_alpha_target_only_p0.1_e100_seed${HEAVY_SEED}_epoch100_sched_epoch_seed${HEAVY_SEED}_fcos${FCOS_NUM_EPOCHS}"
  )
  local job variant gpu_id pretrain_save ckpt save_name
  if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
    GPU_IDS="0,1,2,3" probe_wait_for_runway
  fi
  log "heavy FCOS note: scratch reuses existing scheduler-fixed result front3d_scratch_samepath_sched_epoch_seed${HEAVY_SEED}_fcos${FCOS_NUM_EPOCHS}"
  for job in "${jobs[@]}"; do
    IFS=':' read -r variant gpu_id pretrain_save ckpt save_name <<< "${job}"
    if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${save_name}"; then
      log "skip heavy fcos variant=${variant}"
      continue
    fi
    run_job_probe \
      "${variant}" \
      "${gpu_id}" \
      "${pretrain_save}" \
      "${ckpt}" \
      "${save_name}" \
      "${HEAVY_SEED}" \
      "${LOG_ROOT}/${CHAIN_NAME}.heavy.${variant}.fcos.log"
    pids+=("${RUN_VARIANT_PID}")
    names+=("heavy_${variant}")
  done
  if (( ${#pids[@]} > 0 )); then
    wait_for_pids pids names
  fi
}

run_heavy_budget() {
  log "step_heavy_budget start"
  if (( HEAVY_PRETRAIN_BG_STARTED == 1 )); then
    wait_for_overlapped_heavy_pretrains
  else
    run_heavy_pretrains
  fi
  run_heavy_fcos
  log "step_heavy_budget done"
}

run_scannet_if_available() {
  if [[ ! -d "${SCANNET_DATA_ROOT}/features" ]] || [[ ! -d "${SCANNET_DATA_ROOT}/obb" ]] || [[ ! -f "${SCANNET_DATA_ROOT}/scannet_split.npz" ]]; then
    log "step_scannet skip missing dataset root=${SCANNET_DATA_ROOT}"
    return 0
  fi
  log "step_scannet dataset present but no launcher implemented yet"
}

log "start diagnostics -> alpha_target_only -> heavy_budget -> scannet_if_available"
run_scheduler_fixed_diagnostics
run_diagnostic_summary
run_alpha_target_sequence
run_heavy_budget
run_scannet_if_available
log "chain done"
