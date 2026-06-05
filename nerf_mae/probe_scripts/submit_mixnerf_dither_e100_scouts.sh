#!/usr/bin/env bash
# Submit e100 masked-objective dither / mask-token-free MixNeRF scouts.
#
# This is intentionally separate from the first MixNeRF scout launcher.  The
# goal here is to test whether the e30 same-scene shuffle result survives a
# longer budget, and whether it is distribution matching rather than merely a
# non-zero filler effect.
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
PRETRAIN_MASTER_PORT_BASE="${PRETRAIN_MASTER_PORT_BASE:-29540}"
MIXNERF_MASK_RATIO="${MIXNERF_MASK_RATIO:-0.75}"
MIXNERF_PATCH_SIZE="${MIXNERF_PATCH_SIZE:-4}"
MIXNERF_PARTNER="${MIXNERF_PARTNER:-roll}"
EPOCHS="${EPOCHS:-100}"
SEEDS="${SEEDS:-1 2}"
MEAN_SEEDS="${MEAN_SEEDS:-1}"
SUBMIT_PRETRAIN="${SUBMIT_PRETRAIN:-1}"
SUBMIT_FCOS="${SUBMIT_FCOS:-1}"
SUBMIT_LOG_DIR="${SUBMIT_LOG_DIR:-${ROOT_DIR}/output/launcher/mixnerf_dither_e100_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${SUBMIT_LOG_DIR}"
submitted_tsv="${SUBMIT_LOG_DIR}/submitted.tsv"
printf "kind\tcondition\tepochs\tseed\tfill\tprobe_mode\trgb_loss\talpha_loss\tpretrain_job\tfcos_job\n" > "${submitted_tsv}"

run_or_print() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    { printf '[dry-run]'; printf ' %q' "$@"; printf '\n'; } >&2
  else
    "$@"
  fi
}

short_condition() {
  case "$1" in
    mixnerf_lite_shuffle_visible_masked) printf "mvis" ;;
    mixnerf_lite_zeros_masked) printf "mzero" ;;
    mixnerf_lite_mean_masked) printf "mmean" ;;
    *) printf "%s" "$1" | tr -cd '[:alnum:]_' | cut -c1-8 ;;
  esac
}

condition_port_offset() {
  case "$1" in
    mixnerf_lite_shuffle_visible_masked) printf "1" ;;
    mixnerf_lite_zeros_masked) printf "2" ;;
    mixnerf_lite_mean_masked) printf "3" ;;
    *) printf "9" ;;
  esac
}

port_for_job() {
  local seed="$1"
  local offset="$2"
  printf "%d" $((PRETRAIN_MASTER_PORT_BASE + seed * 10 + offset))
}

submit_one() {
  local condition="$1"
  local epochs="$2"
  local seed="$3"
  local fill="$4"
  local run_suffix="$5"
  local short qsub_gpu_ids eval_interval master_port varlist pre_job fcos_job
  short="$(short_condition "${condition}")"
  qsub_gpu_ids="${PRETRAIN_GPU_IDS//,/:}"
  eval_interval="${PRETRAIN_EVAL_INTERVAL:-${epochs}}"
  master_port="$(port_for_job "${seed}" "$(condition_port_offset "${condition}")")"

  varlist="ROOT_DIR=${ROOT_DIR},KIND=mixnerf,CONDITION=${condition},EPOCHS=${epochs},SEED=${seed},FINETUNE_SEED=${seed},RUN_SUFFIX=${run_suffix},PROBE_ENV_PREFIX=${PROBE_ENV_PREFIX},PRETRAIN_DATA_ROOT=${PRETRAIN_DATA_ROOT},PRETRAIN_DATA_SRC=${PRETRAIN_DATA_SRC},FCOS_DATA_ROOT=${FCOS_DATA_ROOT},ABCI3_CUDA_MODULE=${ABCI3_CUDA_MODULE},PRETRAIN_NODES=${PRETRAIN_NODES},PRETRAIN_GPU_IDS=${qsub_gpu_ids},PRETRAIN_BATCH_SIZE_PER_GPU=${PRETRAIN_BATCH_SIZE_PER_GPU},PRETRAIN_LR=${PRETRAIN_LR},PRETRAIN_WEIGHT_DECAY=${PRETRAIN_WEIGHT_DECAY},PRETRAIN_EVAL_INTERVAL=${eval_interval},PRETRAIN_CHECKPOINT_INTERVAL=${PRETRAIN_CHECKPOINT_INTERVAL},PRETRAIN_KEEP_CHECKPOINTS=${PRETRAIN_KEEP_CHECKPOINTS},PRETRAIN_LOG_INTERVAL=${PRETRAIN_LOG_INTERVAL},PRETRAIN_PROFILE_STEP_TIME=${PRETRAIN_PROFILE_STEP_TIME},PRETRAIN_TRAIN_NUM_WORKERS=${PRETRAIN_TRAIN_NUM_WORKERS},PRETRAIN_EVAL_NUM_WORKERS=${PRETRAIN_EVAL_NUM_WORKERS},PRETRAIN_PERSISTENT_WORKERS=${PRETRAIN_PERSISTENT_WORKERS},STAGE_PRETRAIN_DATA=${STAGE_PRETRAIN_DATA},LOCAL_STAGE_ROOT=${LOCAL_STAGE_ROOT},STAGE_KEEP=${STAGE_KEEP},STAGE_MIN_FREE_GB=${STAGE_MIN_FREE_GB},USE_WANDB=${USE_WANDB},WANDB_MODE=${WANDB_MODE},DETERMINISTIC=${DETERMINISTIC},PRETRAIN_MASTER_PORT=${master_port},MIXNERF_MODE=mix,MIXNERF_MASK_RATIO=${MIXNERF_MASK_RATIO},MIXNERF_PARTNER=${MIXNERF_PARTNER},MIXNERF_FILL_MODE=${fill},MIXNERF_PATCH_SIZE=${MIXNERF_PATCH_SIZE},MIXNERF_DISABLE_INTERNAL_MASK=1,MIXNERF_LOG_STATS=1,MIXNERF_PROBE_MODE=custom,MIXNERF_PROBE_RGB_INPUT=keep,MIXNERF_PROBE_ALPHA_INPUT=keep,MIXNERF_PROBE_ALPHA_TARGET=keep,MIXNERF_PROBE_RGB_LOSS=removed_occupied,MIXNERF_PROBE_ALPHA_LOSS=removed,SKIP_EXISTING=1"

  pre_job=""
  if [[ "${SUBMIT_PRETRAIN}" == "1" ]]; then
    local pre_cmd=(
      qsub
      -P "${ABCI_GROUP}"
      -q "${PRETRAIN_QUEUE}"
      -l "select=${PRETRAIN_NODES}"
      -l "walltime=${PRETRAIN_WALLTIME}"
      -N "dith${short}e${epochs}s${seed}p"
      -j oe
      -o "${SUBMIT_LOG_DIR}/dither_${short}_e${epochs}_s${seed}_pre.pbs.log"
      -v "${varlist}"
      "${SCRIPT_DIR}/abci3_e300_gate_pretrain.pbs"
    )
    if [[ "${DRY_RUN}" == "1" ]]; then
      run_or_print "${pre_cmd[@]}"
      pre_job="DRYRUN_PRE_${short}_${epochs}_s${seed}"
    else
      pre_job="$("${pre_cmd[@]}")"
      pre_job="${pre_job%% *}"
      echo "[submitted] pretrain condition=${condition} epochs=${epochs} seed=${seed} fill=${fill} job=${pre_job}"
    fi
  fi

  fcos_job=""
  if [[ "${SUBMIT_FCOS}" == "1" ]]; then
    local fcos_varlist="${varlist},FCOS_GPU_IDS=${FCOS_GPU_IDS},FCOS_NUM_EPOCHS=${FCOS_NUM_EPOCHS},FCOS_BATCH_SIZE_PER_GPU=${FCOS_BATCH_SIZE_PER_GPU},FCOS_LR=${FCOS_LR},FCOS_WEIGHT_DECAY=${FCOS_WEIGHT_DECAY},FCOS_LR_SCHEDULER=${FCOS_LR_SCHEDULER}"
    local fcos_cmd=(
      qsub
      -P "${ABCI_GROUP}"
      -q "${FCOS_QUEUE}"
      -l select=1
      -l "walltime=${FCOS_WALLTIME}"
      -N "dith${short}e${epochs}s${seed}f"
      -j oe
      -o "${SUBMIT_LOG_DIR}/dither_${short}_e${epochs}_s${seed}_fcos.pbs.log"
      -v "${fcos_varlist}"
    )
    if [[ -n "${pre_job}" ]]; then
      fcos_cmd+=(-W "depend=afterok:${pre_job}")
    fi
    fcos_cmd+=("${SCRIPT_DIR}/abci3_e300_gate_fcos.pbs")
    if [[ "${DRY_RUN}" == "1" ]]; then
      run_or_print "${fcos_cmd[@]}"
      fcos_job="DRYRUN_FCOS_${short}_${epochs}_s${seed}"
    else
      fcos_job="$("${fcos_cmd[@]}")"
      fcos_job="${fcos_job%% *}"
      echo "[submitted] fcos condition=${condition} epochs=${epochs} seed=${seed} job=${fcos_job} dep=${pre_job:-none}"
    fi
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    mixnerf "${condition}" "${epochs}" "${seed}" "${fill}" custom removed_occupied removed "${pre_job}" "${fcos_job}" >> "${submitted_tsv}"
}

for seed in ${SEEDS}; do
  submit_one mixnerf_lite_shuffle_visible_masked "${EPOCHS}" "${seed}" shuffle_visible abci3dither_e100
  submit_one mixnerf_lite_zeros_masked "${EPOCHS}" "${seed}" zeros abci3dither_e100
done

for seed in ${MEAN_SEEDS}; do
  submit_one mixnerf_lite_mean_masked "${EPOCHS}" "${seed}" mean abci3dither_e100
done

echo "[info] submitted manifest=${submitted_tsv}"
