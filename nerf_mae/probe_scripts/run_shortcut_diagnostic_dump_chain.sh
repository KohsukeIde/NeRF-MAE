#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_probe_common.sh"

CHAIN_NAME="${CHAIN_NAME:-nerfmae_shortcut_diagnostics}"
LOG_ROOT="${LOG_ROOT:-${PROBE_LOG_ROOT_DEFAULT}}"
mkdir -p "${LOG_ROOT}"

CHAIN_LOG="${CHAIN_LOG:-${LOG_ROOT}/${CHAIN_NAME}.chain.log}"
GPU_IDS_AVAILABLE="${GPU_IDS_AVAILABLE:-0,1,2,3}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
NMS_THRESH="${NMS_THRESH:-0.3}"
OUTPUT_VOXEL_SCORES="${OUTPUT_VOXEL_SCORES:-1}"
OUTPUT_PROPOSALS="${OUTPUT_PROPOSALS:-1}"
SAVE_LEVEL_INDEX="${SAVE_LEVEL_INDEX:-1}"
FILTER_MODE="${FILTER_MODE:-none}"
FILTER_THRESHOLD="${FILTER_THRESHOLD:-0.7}"
SEED="${SEED:-}"
DETERMINISTIC="${DETERMINISTIC:-0}"

find_latest_checkpoint() {
  local dir="$1"
  find "${dir}" -maxdepth 1 -name 'model_best_ap50_ap25_*.pt' -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | awk '{print $2}'
}

log() {
  probe_log "$*" | tee -a "${CHAIN_LOG}"
}

run_variant() {
  local variant="$1"
  local gpu="$2"
  local checkpoint="$3"
  local save_name="$4"
  local log_file="${LOG_ROOT}/${CHAIN_NAME}.${variant}.log"

  (
    cd "${ROOT_DIR}/nerf_rpn"
    export GPU_IDS="${gpu}"
    export VARIANT_NAME="${variant}"
    export CHECKPOINT="${checkpoint}"
    export SAVE_NAME="${save_name}"
    export EVAL_BATCH_SIZE
    export NMS_THRESH
    export OUTPUT_VOXEL_SCORES
    export OUTPUT_PROPOSALS
    export SAVE_LEVEL_INDEX
    export FILTER_MODE
    export FILTER_THRESHOLD
    export SEED
    export DETERMINISTIC
    bash "${ROOT_DIR}/nerf_rpn/run_fcos_diagnostic_variant.sh"
  ) >"${log_file}" 2>&1 &
  RUN_VARIANT_PID=$!
}

IFS=',' read -r -a gpu_array <<< "${GPU_IDS_AVAILABLE}"
if (( ${#gpu_array[@]} < 4 )); then
  probe_die "run_shortcut_diagnostic_dump_chain.sh requires 4 GPU ids in GPU_IDS_AVAILABLE"
fi

FAIR_SCRATCH_CKPT="$(find_latest_checkpoint "${ROOT_DIR}/output/nerf_rpn/results/front3d_scratch_samepath_fcos100")"
BASELINE_CKPT="$(find_latest_checkpoint "${ROOT_DIR}/output/nerf_rpn/results/nerfmae_all_p0.1_e30_epoch30_fcos100")"
ALPHA_CKPT="$(find_latest_checkpoint "${ROOT_DIR}/output/nerf_rpn/results/nerfmae_alpha_only_p0.1_e30_epoch30_fcos100")"
MASKED_CKPT="$(find_latest_checkpoint "${ROOT_DIR}/output/nerf_rpn/results/nerfmae_masked_only_rgb_loss_p0.1_e30_epoch30_fcos100")"

[[ -n "${FAIR_SCRATCH_CKPT}" ]] || probe_die "missing fair scratch checkpoint"
[[ -n "${BASELINE_CKPT}" ]] || probe_die "missing baseline checkpoint"
[[ -n "${ALPHA_CKPT}" ]] || probe_die "missing alpha checkpoint"
[[ -n "${MASKED_CKPT}" ]] || probe_die "missing masked checkpoint"

log "start diagnostic dump chain"

run_variant fair_scratch "${gpu_array[0]//[[:space:]]/}" "${FAIR_SCRATCH_CKPT}" "front3d_scratch_samepath_fcos100_diagnostics"
pid_fair="${RUN_VARIANT_PID}"
run_variant baseline_e30_epoch30 "${gpu_array[1]//[[:space:]]/}" "${BASELINE_CKPT}" "nerfmae_all_p0.1_e30_epoch30_fcos100_diagnostics"
pid_base="${RUN_VARIANT_PID}"
run_variant alpha_only_e30_epoch30 "${gpu_array[2]//[[:space:]]/}" "${ALPHA_CKPT}" "nerfmae_alpha_only_p0.1_e30_epoch30_fcos100_diagnostics"
pid_alpha="${RUN_VARIANT_PID}"
run_variant masked_only_e30_epoch30 "${gpu_array[3]//[[:space:]]/}" "${MASKED_CKPT}" "nerfmae_masked_only_rgb_loss_p0.1_e30_epoch30_fcos100_diagnostics"
pid_masked="${RUN_VARIANT_PID}"

for item in \
  "fair_scratch:${pid_fair}" \
  "baseline_e30_epoch30:${pid_base}" \
  "alpha_only_e30_epoch30:${pid_alpha}" \
  "masked_only_e30_epoch30:${pid_masked}"; do
  name="${item%%:*}"
  pid="${item##*:}"
  if wait "${pid}"; then
    log "job_done name=${name}"
  else
    log "job_failed name=${name}"
    exit 1
  fi
done

log "diagnostic dump chain done"
