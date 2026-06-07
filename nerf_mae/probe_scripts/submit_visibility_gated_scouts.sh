#!/usr/bin/env bash
# Submit Visibility-Gated V0/V1 scouts with dependent FCOS eval.
#
# This branch is allowed only after the encoder participation gate is positive.
# It intentionally submits V0/V1 only:
# - visibility_feature_reset: reset masked stage features before later stages.
# - visibility_skip_gate: gate decoder skip features at masked locations.
# - visibility_cosine_skip_gate: same gate plus cosine RGB-ramp curriculum.
#
# Attention KV-gating is not included here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

ABCI_GROUP="${ABCI_GROUP:-gag51404}"
PRETRAIN_QUEUE="${PRETRAIN_QUEUE:-rt_HF}"
FCOS_QUEUE="${FCOS_QUEUE:-rt_HG}"
PRETRAIN_WALLTIME="${PRETRAIN_WALLTIME:-72:00:00}"
FCOS_WALLTIME="${FCOS_WALLTIME:-72:00:00}"
PRETRAIN_NODES="${PRETRAIN_NODES:-1}"
PRETRAIN_GPU_IDS="${PRETRAIN_GPU_IDS:-0-7}"
PRETRAIN_BATCH_SIZE_PER_GPU="${PRETRAIN_BATCH_SIZE_PER_GPU:-2}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-3}"
PRETRAIN_WEIGHT_DECAY="${PRETRAIN_WEIGHT_DECAY:-0.0}"
PRETRAIN_EVAL_INTERVAL="${PRETRAIN_EVAL_INTERVAL:-}"
PRETRAIN_CHECKPOINT_INTERVAL="${PRETRAIN_CHECKPOINT_INTERVAL:-50}"
PRETRAIN_KEEP_CHECKPOINTS="${PRETRAIN_KEEP_CHECKPOINTS:-0}"
PRETRAIN_LOG_INTERVAL="${PRETRAIN_LOG_INTERVAL:-30}"
PRETRAIN_PROFILE_STEP_TIME="${PRETRAIN_PROFILE_STEP_TIME:-0}"
PRETRAIN_TRAIN_NUM_WORKERS="${PRETRAIN_TRAIN_NUM_WORKERS:-}"
PRETRAIN_EVAL_NUM_WORKERS="${PRETRAIN_EVAL_NUM_WORKERS:-}"
PRETRAIN_PERSISTENT_WORKERS="${PRETRAIN_PERSISTENT_WORKERS:-0}"
STAGE_PRETRAIN_DATA="${STAGE_PRETRAIN_DATA:-0}"
PRETRAIN_DATA_ROOT="${PRETRAIN_DATA_ROOT:-${ROOT_DIR}/dataset/pretrain}"
PRETRAIN_DATA_SRC="${PRETRAIN_DATA_SRC:-${PRETRAIN_DATA_ROOT}}"
LOCAL_STAGE_ROOT="${LOCAL_STAGE_ROOT:-}"
STAGE_KEEP="${STAGE_KEEP:-0}"
STAGE_MIN_FREE_GB="${STAGE_MIN_FREE_GB:-80}"
FCOS_DATA_ROOT="${FCOS_DATA_ROOT:-${ROOT_DIR}/dataset/finetune/front3d_rpn_data}"
FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS:-1000}"
FCOS_GPU_IDS="${FCOS_GPU_IDS:-0}"
FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU:-2}"
FCOS_LR="${FCOS_LR:-1e-4}"
FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY:-1e-3}"
FCOS_LR_SCHEDULER="${FCOS_LR_SCHEDULER:-onecycle_epoch}"
ABCI3_CUDA_MODULE="${ABCI3_CUDA_MODULE:-cuda/11.8/11.8.0}"
PROBE_ENV_PREFIX="${PROBE_ENV_PREFIX:-${ROOT_DIR}/.venv-abci3}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
DETERMINISTIC="${DETERMINISTIC:-0}"
PRETRAIN_MASTER_PORT_BASE="${PRETRAIN_MASTER_PORT_BASE:-29620}"
EPOCHS="${EPOCHS:-100}"
SEEDS="${SEEDS:-1}"
CONDITIONS="${CONDITIONS:-visibility_skip_gate visibility_feature_reset}"
VISGATE_RESET_STAGES="${VISGATE_RESET_STAGES:-0,1,2}"
VISGATE_SKIP_STAGES="${VISGATE_SKIP_STAGES:-0,1,2}"
VISGATE_LOG_STATS="${VISGATE_LOG_STATS:-1}"
SUBMIT_PRETRAIN="${SUBMIT_PRETRAIN:-1}"
SUBMIT_FCOS="${SUBMIT_FCOS:-1}"
RUN_SUFFIX="${RUN_SUFFIX:-abci3vis_e${EPOCHS}_$(date +%Y%m%d_%H%M%S)}"
SUBMIT_LOG_DIR="${SUBMIT_LOG_DIR:-${ROOT_DIR}/output/launcher/visibility_gated_${RUN_SUFFIX}}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${SUBMIT_LOG_DIR}"
submitted_tsv="${SUBMIT_LOG_DIR}/submitted.tsv"
printf "kind\tcondition\tepochs\tseed\tvisgate_mode\tpretrain_job\tfcos_job\n" > "${submitted_tsv}"

run_or_print() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    { printf '[dry-run]'; printf ' %q' "$@"; printf '\n'; } >&2
  else
    "$@"
  fi
}

condition_mode() {
  case "$1" in
    visibility_feature_reset) printf "feature_reset" ;;
    visibility_skip_gate) printf "skip_gate" ;;
    visibility_cosine_skip_gate) printf "skip_gate" ;;
    visibility_reset_skip) printf "reset_skip" ;;
    *) echo "[error] unsupported condition=$1" >&2; exit 1 ;;
  esac
}

condition_offset() {
  case "$1" in
    visibility_skip_gate) printf "1" ;;
    visibility_feature_reset) printf "2" ;;
    visibility_cosine_skip_gate) printf "3" ;;
    visibility_reset_skip) printf "4" ;;
    *) printf "9" ;;
  esac
}

short_condition() {
  case "$1" in
    visibility_skip_gate) printf "vskip" ;;
    visibility_feature_reset) printf "vreset" ;;
    visibility_cosine_skip_gate) printf "vcskip" ;;
    visibility_reset_skip) printf "vboth" ;;
    *) printf "%s" "$1" | tr -cd '[:alnum:]_' | cut -c1-8 ;;
  esac
}

submit_one() {
  local condition="$1"
  local seed="$2"
  local mode short qsub_gpu_ids eval_interval master_port varlist pre_job fcos_job reset_stages_q skip_stages_q
  mode="$(condition_mode "${condition}")"
  short="$(short_condition "${condition}")"
  qsub_gpu_ids="${PRETRAIN_GPU_IDS//,/:}"
  fcos_gpu_ids_q="${FCOS_GPU_IDS//,/:}"
  reset_stages_q="${VISGATE_RESET_STAGES//,/:}"
  skip_stages_q="${VISGATE_SKIP_STAGES//,/:}"
  eval_interval="${PRETRAIN_EVAL_INTERVAL:-${EPOCHS}}"
  master_port="$((PRETRAIN_MASTER_PORT_BASE + seed * 10 + $(condition_offset "${condition}")))"

  varlist="ROOT_DIR=${ROOT_DIR},KIND=visibility,CONDITION=${condition},EPOCHS=${EPOCHS},SEED=${seed},FINETUNE_SEED=${seed},RUN_SUFFIX=${RUN_SUFFIX},PROBE_ENV_PREFIX=${PROBE_ENV_PREFIX},PRETRAIN_DATA_ROOT=${PRETRAIN_DATA_ROOT},PRETRAIN_DATA_SRC=${PRETRAIN_DATA_SRC},FCOS_DATA_ROOT=${FCOS_DATA_ROOT},ABCI3_CUDA_MODULE=${ABCI3_CUDA_MODULE},PRETRAIN_NODES=${PRETRAIN_NODES},PRETRAIN_GPU_IDS=${qsub_gpu_ids},PRETRAIN_BATCH_SIZE_PER_GPU=${PRETRAIN_BATCH_SIZE_PER_GPU},PRETRAIN_LR=${PRETRAIN_LR},PRETRAIN_WEIGHT_DECAY=${PRETRAIN_WEIGHT_DECAY},PRETRAIN_EVAL_INTERVAL=${eval_interval},PRETRAIN_CHECKPOINT_INTERVAL=${PRETRAIN_CHECKPOINT_INTERVAL},PRETRAIN_KEEP_CHECKPOINTS=${PRETRAIN_KEEP_CHECKPOINTS},PRETRAIN_LOG_INTERVAL=${PRETRAIN_LOG_INTERVAL},PRETRAIN_PROFILE_STEP_TIME=${PRETRAIN_PROFILE_STEP_TIME},PRETRAIN_TRAIN_NUM_WORKERS=${PRETRAIN_TRAIN_NUM_WORKERS},PRETRAIN_EVAL_NUM_WORKERS=${PRETRAIN_EVAL_NUM_WORKERS},PRETRAIN_PERSISTENT_WORKERS=${PRETRAIN_PERSISTENT_WORKERS},STAGE_PRETRAIN_DATA=${STAGE_PRETRAIN_DATA},LOCAL_STAGE_ROOT=${LOCAL_STAGE_ROOT},STAGE_KEEP=${STAGE_KEEP},STAGE_MIN_FREE_GB=${STAGE_MIN_FREE_GB},USE_WANDB=${USE_WANDB},WANDB_MODE=${WANDB_MODE},DETERMINISTIC=${DETERMINISTIC},PRETRAIN_MASTER_PORT=${master_port},VISGATE_MODE=${mode},VISGATE_RESET_STAGES=${reset_stages_q},VISGATE_SKIP_STAGES=${skip_stages_q},VISGATE_LOG_STATS=${VISGATE_LOG_STATS},VISGATE_PROBE_CURRICULUM=${VISGATE_PROBE_CURRICULUM:-},VISGATE_PROBE_CURRICULUM_EPOCHS=${VISGATE_PROBE_CURRICULUM_EPOCHS:-},VISGATE_PROBE_CURRICULUM_RGB_START_WEIGHT=${VISGATE_PROBE_CURRICULUM_RGB_START_WEIGHT:-},VISGATE_PROBE_CURRICULUM_RGB_END_WEIGHT=${VISGATE_PROBE_CURRICULUM_RGB_END_WEIGHT:-},VISGATE_PROBE_CURRICULUM_ALPHA_WEIGHT=${VISGATE_PROBE_CURRICULUM_ALPHA_WEIGHT:-},AUTO_RESUME_PRETRAIN=0,SKIP_EXISTING=1"

  pre_job=""
  if [[ "${SUBMIT_PRETRAIN}" == "1" ]]; then
    local pre_cmd=(
      qsub
      -P "${ABCI_GROUP}"
      -q "${PRETRAIN_QUEUE}"
      -l "select=${PRETRAIN_NODES}"
      -l "walltime=${PRETRAIN_WALLTIME}"
      -N "${short}e${EPOCHS}s${seed}p"
      -j oe
      -o "${SUBMIT_LOG_DIR}/${short}_e${EPOCHS}_s${seed}_pre.pbs.log"
      -v "${varlist}"
      "${SCRIPT_DIR}/abci3_e300_gate_pretrain.pbs"
    )
    if [[ "${DRY_RUN}" == "1" ]]; then
      run_or_print "${pre_cmd[@]}"
      pre_job="DRYRUN_PRE_${short}_${EPOCHS}_s${seed}"
    else
      pre_job="$("${pre_cmd[@]}")"
      pre_job="${pre_job%% *}"
      echo "[submitted] pretrain condition=${condition} seed=${seed} mode=${mode} job=${pre_job}"
    fi
  fi

  fcos_job=""
  if [[ "${SUBMIT_FCOS}" == "1" ]]; then
    local fcos_varlist="${varlist},FCOS_GPU_IDS=${fcos_gpu_ids_q},FCOS_NUM_EPOCHS=${FCOS_NUM_EPOCHS},FCOS_BATCH_SIZE_PER_GPU=${FCOS_BATCH_SIZE_PER_GPU},FCOS_LR=${FCOS_LR},FCOS_WEIGHT_DECAY=${FCOS_WEIGHT_DECAY},FCOS_LR_SCHEDULER=${FCOS_LR_SCHEDULER}"
    local fcos_cmd=(
      qsub
      -P "${ABCI_GROUP}"
      -q "${FCOS_QUEUE}"
      -l select=1
      -l "walltime=${FCOS_WALLTIME}"
      -N "${short}e${EPOCHS}s${seed}f"
      -j oe
      -o "${SUBMIT_LOG_DIR}/${short}_e${EPOCHS}_s${seed}_fcos.pbs.log"
      -v "${fcos_varlist}"
    )
    if [[ -n "${pre_job}" ]]; then
      fcos_cmd+=(-W "depend=afterok:${pre_job}")
    fi
    fcos_cmd+=("${SCRIPT_DIR}/abci3_e300_gate_fcos.pbs")
    if [[ "${DRY_RUN}" == "1" ]]; then
      run_or_print "${fcos_cmd[@]}"
      fcos_job="DRYRUN_FCOS_${short}_${EPOCHS}_s${seed}"
    else
      fcos_job="$("${fcos_cmd[@]}")"
      fcos_job="${fcos_job%% *}"
      echo "[submitted] fcos condition=${condition} seed=${seed} job=${fcos_job} dep=${pre_job:-none}"
    fi
  fi

  printf "visibility\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${condition}" "${EPOCHS}" "${seed}" "${mode}" "${pre_job}" "${fcos_job}" >> "${submitted_tsv}"
}

for seed in ${SEEDS}; do
  for condition in ${CONDITIONS}; do
    submit_one "${condition}" "${seed}"
  done
done

echo "[info] submitted manifest=${submitted_tsv}"
