#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ABCI_GROUP="${ABCI_GROUP:-gag51404}"
PRETRAIN_QUEUE="${PRETRAIN_QUEUE:-rt_HF}"
FCOS_QUEUE="${FCOS_QUEUE:-rt_HG}"
PRETRAIN_WALLTIME="${PRETRAIN_WALLTIME:-72:00:00}"
FCOS_WALLTIME="${FCOS_WALLTIME:-72:00:00}"
PRETRAIN_NODES="${PRETRAIN_NODES:-1}"
PRETRAIN_GPU_IDS="${PRETRAIN_GPU_IDS:-0-7}"
PRETRAIN_BATCH_SIZE_PER_GPU="${PRETRAIN_BATCH_SIZE_PER_GPU:-}"
PRETRAIN_EVAL_INTERVAL="${PRETRAIN_EVAL_INTERVAL:-300}"
PRETRAIN_LOG_INTERVAL="${PRETRAIN_LOG_INTERVAL:-30}"
AUTO_RESUME_PRETRAIN="${AUTO_RESUME_PRETRAIN:-1}"
if [[ -z "${PRETRAIN_BATCH_SIZE_PER_GPU}" ]]; then
  if (( PRETRAIN_NODES > 1 )); then
    PRETRAIN_BATCH_SIZE_PER_GPU=1
  else
    PRETRAIN_BATCH_SIZE_PER_GPU=2
  fi
fi
RUN_SUFFIX="${RUN_SUFFIX:-abci3clean}"
PROBE_ENV_PREFIX="${PROBE_ENV_PREFIX:-${ROOT_DIR}/.venv-abci3}"
PRETRAIN_DATA_ROOT="${PRETRAIN_DATA_ROOT:-${ROOT_DIR}/dataset/pretrain}"
FCOS_DATA_ROOT="${FCOS_DATA_ROOT:-${ROOT_DIR}/dataset/finetune/front3d_rpn_data}"
SUBMIT_LOG_DIR="${SUBMIT_LOG_DIR:-${ROOT_DIR}/output/launcher/abci3_e300_gate}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
SUBMIT_PRETRAIN="${SUBMIT_PRETRAIN:-1}"
SUBMIT_FCOS="${SUBMIT_FCOS:-1}"

# Format: kind:condition:epochs:seed
FULL_GATE_JOBS="baseline:baseline:300:1 baseline:baseline:300:2 baseline:baseline:300:3 curriculum:cosine_ramp:300:1 curriculum:cosine_ramp:300:2 curriculum:cosine_ramp:300:3 curriculum:cosine_ramp_alpha_shuffle:300:1 curriculum:cosine_ramp_alpha_shuffle:300:2 curriculum:cosine_ramp_alpha_shuffle:300:3"
MISSING_PRETRAIN_JOBS="baseline:baseline:300:3 curriculum:cosine_ramp:300:2 curriculum:cosine_ramp:300:3 curriculum:cosine_ramp_alpha_shuffle:300:2 curriculum:cosine_ramp_alpha_shuffle:300:3"
MINIMAL_PAPER_JOBS="baseline:baseline:300:1 curriculum:cosine_ramp:300:1 curriculum:cosine_ramp_alpha_shuffle:300:1"
GATE_JOBS="${GATE_JOBS:-${FULL_GATE_JOBS}}"

mkdir -p "${SUBMIT_LOG_DIR}"

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  PROBE_ENV_PREFIX="${PROBE_ENV_PREFIX}" \
  PRETRAIN_DATA_ROOT="${PRETRAIN_DATA_ROOT}" \
  FCOS_DATA_ROOT="${FCOS_DATA_ROOT}" \
  REQUIRE_PRETRAIN_DATA="${SUBMIT_PRETRAIN}" \
  REQUIRE_FCOS_DATA="${SUBMIT_FCOS}" \
  bash "${SCRIPT_DIR}/abci3_e300_gate_preflight.sh"
fi

short_condition() {
  case "$1" in
    baseline) printf "base" ;;
    cosine_ramp) printf "cos" ;;
    cosine_ramp_alpha_shuffle) printf "shuf" ;;
    *) printf "%s" "$1" | tr -cd '[:alnum:]_' | cut -c1-8 ;;
  esac
}

submit_one() {
  local kind="$1"
  local condition="$2"
  local epochs="$3"
  local seed="$4"
  local short
  short="$(short_condition "${condition}")"

  local pre_name="e${epochs}_${short}_s${seed}_pre"
  local fcos_name="e${epochs}_${short}_s${seed}_fcos"
  local qsub_gpu_ids="${PRETRAIN_GPU_IDS//,/:}"
  local varlist="KIND=${kind},CONDITION=${condition},EPOCHS=${epochs},SEED=${seed},RUN_SUFFIX=${RUN_SUFFIX},PROBE_ENV_PREFIX=${PROBE_ENV_PREFIX},PRETRAIN_DATA_ROOT=${PRETRAIN_DATA_ROOT},FCOS_DATA_ROOT=${FCOS_DATA_ROOT},PRETRAIN_NODES=${PRETRAIN_NODES},PRETRAIN_GPU_IDS=${qsub_gpu_ids},PRETRAIN_BATCH_SIZE_PER_GPU=${PRETRAIN_BATCH_SIZE_PER_GPU},PRETRAIN_EVAL_INTERVAL=${PRETRAIN_EVAL_INTERVAL},PRETRAIN_LOG_INTERVAL=${PRETRAIN_LOG_INTERVAL},AUTO_RESUME_PRETRAIN=${AUTO_RESUME_PRETRAIN},SKIP_EXISTING=1"
  local pre_jobid=""

  if [[ "${SUBMIT_PRETRAIN}" == "1" ]]; then
    local pre_cmd=(
      qsub
      -P "${ABCI_GROUP}"
      -q "${PRETRAIN_QUEUE}"
      -l "select=${PRETRAIN_NODES}"
      -l "walltime=${PRETRAIN_WALLTIME}"
      -N "${pre_name}"
      -j oe
      -o "${SUBMIT_LOG_DIR}/${pre_name}.pbs.log"
      -v "${varlist}"
      "${SCRIPT_DIR}/abci3_e300_gate_pretrain.pbs"
    )
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '[dry-run]'; printf ' %q' "${pre_cmd[@]}"; printf '\n'
      pre_jobid="DRYRUN"
    else
      pre_jobid="$("${pre_cmd[@]}")"
      pre_jobid="${pre_jobid%% *}"
      echo "[submitted] pretrain ${kind}:${condition}:e${epochs}:seed${seed} job=${pre_jobid}"
    fi
  fi

  if [[ "${SUBMIT_FCOS}" == "1" ]]; then
    local fcos_cmd=(
      qsub
      -P "${ABCI_GROUP}"
      -q "${FCOS_QUEUE}"
      -l select=1
      -l "walltime=${FCOS_WALLTIME}"
      -N "${fcos_name}"
      -j oe
      -o "${SUBMIT_LOG_DIR}/${fcos_name}.pbs.log"
      -v "${varlist}"
    )
    if [[ "${SUBMIT_PRETRAIN}" == "1" ]]; then
      fcos_cmd+=(-W "depend=afterok:${pre_jobid}")
    fi
    fcos_cmd+=("${SCRIPT_DIR}/abci3_e300_gate_fcos.pbs")

    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '[dry-run]'; printf ' %q' "${fcos_cmd[@]}"; printf '\n'
    else
      local fcos_jobid
      fcos_jobid="$("${fcos_cmd[@]}")"
      fcos_jobid="${fcos_jobid%% *}"
      echo "[submitted] fcos ${kind}:${condition}:e${epochs}:seed${seed} job=${fcos_jobid} dep=${pre_jobid:-none}"
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${kind}" "${condition}" "${epochs}" "${seed}" "${pre_jobid:-none}" "${fcos_jobid}" >> "${SUBMIT_LOG_DIR}/submitted.tsv"
    fi
  fi
}

if [[ "${DRY_RUN}" != "1" ]]; then
  : > "${SUBMIT_LOG_DIR}/submitted.tsv"
fi

cd "${ROOT_DIR}"
for job in ${GATE_JOBS}; do
  IFS=':' read -r kind condition epochs seed <<< "${job}"
  submit_one "${kind}" "${condition}" "${epochs}" "${seed}"
done

echo "[info] submit log dir: ${SUBMIT_LOG_DIR}"
