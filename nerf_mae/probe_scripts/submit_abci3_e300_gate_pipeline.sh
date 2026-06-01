#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ABCI_GROUP="${ABCI_GROUP:-gag51404}"
PRETRAIN_QUEUE="${PRETRAIN_QUEUE:-rt_HF}"
FCOS_QUEUE="${FCOS_QUEUE:-rt_HG}"
PRETRAIN_WALLTIME="${PRETRAIN_WALLTIME:-72:00:00}"
FCOS_WALLTIME="${FCOS_WALLTIME:-72:00:00}"
PRETRAIN_NODES="${PRETRAIN_NODES:-2}"
PRETRAIN_SLOTS="${PRETRAIN_SLOTS:-3}"
PRETRAIN_GPU_IDS="${PRETRAIN_GPU_IDS:-0-7}"
ABCI3_CUDA_MODULE="${ABCI3_CUDA_MODULE:-cuda/11.8/11.8.0}"
PROBE_ENV_PREFIX="${PROBE_ENV_PREFIX:-${ROOT_DIR}/.venv-abci3}"
PRETRAIN_DATA_ROOT="${PRETRAIN_DATA_ROOT:-${ROOT_DIR}/dataset/pretrain}"
FCOS_DATA_ROOT="${FCOS_DATA_ROOT:-${ROOT_DIR}/dataset/finetune/front3d_rpn_data}"
PRETRAIN_BATCH_SIZE_PER_GPU="${PRETRAIN_BATCH_SIZE_PER_GPU:-}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-3}"
PRETRAIN_WEIGHT_DECAY="${PRETRAIN_WEIGHT_DECAY:-0.0}"
PRETRAIN_EVAL_INTERVAL="${PRETRAIN_EVAL_INTERVAL:-300}"
PRETRAIN_CHECKPOINT_INTERVAL="${PRETRAIN_CHECKPOINT_INTERVAL:-50}"
PRETRAIN_KEEP_CHECKPOINTS="${PRETRAIN_KEEP_CHECKPOINTS:-0}"
PRETRAIN_LOG_INTERVAL="${PRETRAIN_LOG_INTERVAL:-30}"
PRETRAIN_PROFILE_STEP_TIME="${PRETRAIN_PROFILE_STEP_TIME:-0}"
PRETRAIN_TRAIN_NUM_WORKERS="${PRETRAIN_TRAIN_NUM_WORKERS:-}"
PRETRAIN_EVAL_NUM_WORKERS="${PRETRAIN_EVAL_NUM_WORKERS:-}"
PRETRAIN_PERSISTENT_WORKERS="${PRETRAIN_PERSISTENT_WORKERS:-0}"
STAGE_PRETRAIN_DATA="${STAGE_PRETRAIN_DATA:-0}"
PRETRAIN_DATA_SRC="${PRETRAIN_DATA_SRC:-${PRETRAIN_DATA_ROOT}}"
LOCAL_STAGE_ROOT="${LOCAL_STAGE_ROOT:-}"
STAGE_KEEP="${STAGE_KEEP:-0}"
STAGE_MIN_FREE_GB="${STAGE_MIN_FREE_GB:-80}"
AUTO_RESUME_PRETRAIN="${AUTO_RESUME_PRETRAIN:-1}"
RESUME_ALLOW_PARTIAL="${RESUME_ALLOW_PARTIAL:-1}"
FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS:-1000}"
FCOS_GPU_IDS="${FCOS_GPU_IDS:-0}"
FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU:-2}"
FCOS_LR="${FCOS_LR:-1e-4}"
FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY:-1e-3}"
FCOS_LR_SCHEDULER="${FCOS_LR_SCHEDULER:-onecycle_epoch}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
DETERMINISTIC="${DETERMINISTIC:-1}"
PRETRAIN_MASTER_PORT="${PRETRAIN_MASTER_PORT:-29500}"
DISABLE_ABS_POS_EMBED="${DISABLE_ABS_POS_EMBED:-0}"
DISABLE_RELATIVE_POSITION_BIAS="${DISABLE_RELATIVE_POSITION_BIAS:-0}"
ROTATE_PROB="${ROTATE_PROB:-}"
FLIP_PROB="${FLIP_PROB:-}"
ROT_SCALE_PROB="${ROT_SCALE_PROB:-}"
COORD_SHIFT_PROB="${COORD_SHIFT_PROB:-}"
COORD_SHIFT_MAX_VOXELS="${COORD_SHIFT_MAX_VOXELS:-}"
if [[ -z "${PRETRAIN_BATCH_SIZE_PER_GPU}" ]]; then
  if (( PRETRAIN_NODES > 1 )); then
    PRETRAIN_BATCH_SIZE_PER_GPU=1
  else
    PRETRAIN_BATCH_SIZE_PER_GPU=2
  fi
fi
RUN_SUFFIX="${RUN_SUFFIX:-abci3clean}"
SUBMIT_LOG_DIR="${SUBMIT_LOG_DIR:-${ROOT_DIR}/output/launcher/abci3_e300_gate_pipeline}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
SUBMIT_PRETRAIN="${SUBMIT_PRETRAIN:-1}"
SUBMIT_FCOS="${SUBMIT_FCOS:-1}"
GLOBAL_DEPENDENCY="${GLOBAL_DEPENDENCY:-}"

# Paired order gives usable seed-wise comparisons as early as possible.
FULL_GATE_JOBS="baseline:baseline:300:1 baseline:baseline:300:2 baseline:baseline:300:3 curriculum:cosine_ramp:300:1 curriculum:cosine_ramp:300:2 curriculum:cosine_ramp:300:3 curriculum:cosine_ramp_alpha_shuffle:300:1 curriculum:cosine_ramp_alpha_shuffle:300:2 curriculum:cosine_ramp_alpha_shuffle:300:3"
MISSING_PRETRAIN_JOBS="baseline:baseline:300:3 curriculum:cosine_ramp:300:2 curriculum:cosine_ramp:300:3 curriculum:cosine_ramp_alpha_shuffle:300:2 curriculum:cosine_ramp_alpha_shuffle:300:3"
MINIMAL_PAPER_JOBS="baseline:baseline:300:1 curriculum:cosine_ramp:300:1 curriculum:cosine_ramp_alpha_shuffle:300:1"
GATE_JOBS="${GATE_JOBS:-${FULL_GATE_JOBS}}"

if (( PRETRAIN_NODES < 1 )); then
  echo "[error] PRETRAIN_NODES must be >= 1" >&2
  exit 1
fi
if (( PRETRAIN_SLOTS < 1 )); then
  echo "[error] PRETRAIN_SLOTS must be >= 1" >&2
  exit 1
fi

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
    alpha_target_only) printf "ato" ;;
    alpha_target_only_no_pos) printf "atnopos" ;;
    alpha_target_only_coord_jitter) printf "atjit" ;;
    baseline_no_pos) printf "basenop" ;;
    baseline_coord_jitter) printf "basejit" ;;
    cosine_no_pos) printf "cosnop" ;;
    cosine_coord_jitter) printf "cosjit" ;;
    shuffle_coord_jitter) printf "shufjit" ;;
    dmae_target_alpha_gated_rgb) printf "dmaegate" ;;
    dmae_hier_concat) printf "dmaecat" ;;
    dmae_hier_concat_coord_jitter) printf "dmaecj" ;;
    dmae_hier_film) printf "dmaefilm" ;;
    *) printf "%s" "$1" | tr -cd '[:alnum:]_' | cut -c1-8 ;;
  esac
}

pretrain_save_name() {
  local kind="$1"
  local condition="$2"
  local epochs="$3"
  local seed="$4"
  local suffix_part=""
  if [[ -n "${RUN_SUFFIX}" ]]; then
    suffix_part="_${RUN_SUFFIX}"
  fi

  case "${kind}:${condition}" in
    baseline:baseline)
      printf "nerfmae_all_p1.0_e%s_seed%s%s\n" "${epochs}" "${seed}" "${suffix_part}"
      ;;
    curriculum:cosine_ramp|curriculum:cosine_ramp_alpha_shuffle)
      printf "nerfmae_alpha_rgba_curr_%s_p1.0_e%s_seed%s%s\n" "${condition}" "${epochs}" "${seed}" "${suffix_part}"
      ;;
    diagnostic:alpha_target_only|diagnostic:alpha_target_only_no_pos|diagnostic:alpha_target_only_coord_jitter|diagnostic:baseline_no_pos|diagnostic:baseline_coord_jitter|diagnostic:cosine_no_pos|diagnostic:cosine_coord_jitter|diagnostic:shuffle_coord_jitter|diagnostic:dmae_target_alpha_gated_rgb|diagnostic:dmae_hier_concat|diagnostic:dmae_hier_concat_coord_jitter|diagnostic:dmae_hier_film)
      printf "nerfmae_%s_p1.0_e%s_seed%s%s\n" "${condition}" "${epochs}" "${seed}" "${suffix_part}"
      ;;
    *)
      echo "[error] unknown KIND:CONDITION=${kind}:${condition}" >&2
      exit 1
      ;;
  esac
}

pretrain_checkpoint_path() {
  local save_name
  save_name="$(pretrain_save_name "$@")"
  printf "%s/output/nerf_mae/results/%s/epoch_%s.pt\n" "${ROOT_DIR}" "${save_name}" "$3"
}

run_or_print() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    { printf '[dry-run]'; printf ' %q' "$@"; printf '\n'; } >&2
  else
    "$@"
  fi
}

combine_dependency() {
  local a="${1:-}"
  local b="${2:-}"
  if [[ -n "${a}" && -n "${b}" ]]; then
    printf "%s:%s\n" "${a}" "${b}"
  elif [[ -n "${a}" ]]; then
    printf "%s\n" "${a}"
  else
    printf "%s\n" "${b}"
  fi
}

submit_pretrain() {
  local idx="$1"
  local slot="$2"
  local dependency="$3"
  local kind="$4"
  local condition="$5"
  local epochs="$6"
  local seed="$7"
  local short pre_name qsub_gpu_ids varlist cmd_output

  short="$(short_condition "${condition}")"
  pre_name="e${epochs}_${short}_s${seed}_pre"
  qsub_gpu_ids="${PRETRAIN_GPU_IDS//,/:}"
  varlist="KIND=${kind},CONDITION=${condition},EPOCHS=${epochs},SEED=${seed},RUN_SUFFIX=${RUN_SUFFIX},PROBE_ENV_PREFIX=${PROBE_ENV_PREFIX},PRETRAIN_DATA_ROOT=${PRETRAIN_DATA_ROOT},PRETRAIN_DATA_SRC=${PRETRAIN_DATA_SRC},FCOS_DATA_ROOT=${FCOS_DATA_ROOT},ABCI3_CUDA_MODULE=${ABCI3_CUDA_MODULE},PRETRAIN_NODES=${PRETRAIN_NODES},PRETRAIN_GPU_IDS=${qsub_gpu_ids},PRETRAIN_BATCH_SIZE_PER_GPU=${PRETRAIN_BATCH_SIZE_PER_GPU},PRETRAIN_LR=${PRETRAIN_LR},PRETRAIN_WEIGHT_DECAY=${PRETRAIN_WEIGHT_DECAY},PRETRAIN_EVAL_INTERVAL=${PRETRAIN_EVAL_INTERVAL},PRETRAIN_CHECKPOINT_INTERVAL=${PRETRAIN_CHECKPOINT_INTERVAL},PRETRAIN_KEEP_CHECKPOINTS=${PRETRAIN_KEEP_CHECKPOINTS},PRETRAIN_LOG_INTERVAL=${PRETRAIN_LOG_INTERVAL},PRETRAIN_PROFILE_STEP_TIME=${PRETRAIN_PROFILE_STEP_TIME},PRETRAIN_TRAIN_NUM_WORKERS=${PRETRAIN_TRAIN_NUM_WORKERS},PRETRAIN_EVAL_NUM_WORKERS=${PRETRAIN_EVAL_NUM_WORKERS},PRETRAIN_PERSISTENT_WORKERS=${PRETRAIN_PERSISTENT_WORKERS},STAGE_PRETRAIN_DATA=${STAGE_PRETRAIN_DATA},LOCAL_STAGE_ROOT=${LOCAL_STAGE_ROOT},STAGE_KEEP=${STAGE_KEEP},STAGE_MIN_FREE_GB=${STAGE_MIN_FREE_GB},AUTO_RESUME_PRETRAIN=${AUTO_RESUME_PRETRAIN},RESUME_ALLOW_PARTIAL=${RESUME_ALLOW_PARTIAL},USE_WANDB=${USE_WANDB},WANDB_MODE=${WANDB_MODE},DETERMINISTIC=${DETERMINISTIC},PRETRAIN_MASTER_PORT=${PRETRAIN_MASTER_PORT},DISABLE_ABS_POS_EMBED=${DISABLE_ABS_POS_EMBED},DISABLE_RELATIVE_POSITION_BIAS=${DISABLE_RELATIVE_POSITION_BIAS},ROTATE_PROB=${ROTATE_PROB},FLIP_PROB=${FLIP_PROB},ROT_SCALE_PROB=${ROT_SCALE_PROB},COORD_SHIFT_PROB=${COORD_SHIFT_PROB},COORD_SHIFT_MAX_VOXELS=${COORD_SHIFT_MAX_VOXELS},SKIP_EXISTING=1"

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
  )
  if [[ -n "${dependency}" ]]; then
    pre_cmd+=(-W "depend=afterok:${dependency}")
  fi
  pre_cmd+=("${SCRIPT_DIR}/abci3_e300_gate_pretrain.pbs")

  if [[ "${DRY_RUN}" == "1" ]]; then
    run_or_print "${pre_cmd[@]}"
    printf "DRYRUN_PRE_%02d\n" "${idx}"
  else
    cmd_output="$("${pre_cmd[@]}")"
    cmd_output="${cmd_output%% *}"
    echo "[submitted] slot=${slot} pretrain ${kind}:${condition}:e${epochs}:seed${seed} job=${cmd_output} dep=${dependency:-none}" >&2
    printf "%s\n" "${cmd_output}"
  fi
}

submit_fcos() {
  local idx="$1"
  local pre_jobid="$2"
  local kind="$3"
  local condition="$4"
  local epochs="$5"
  local seed="$6"
  local finetune_seed="$7"
  local short fcos_name qsub_gpu_ids varlist cmd_output

  short="$(short_condition "${condition}")"
  if [[ "${finetune_seed}" == "${seed}" ]]; then
    fcos_name="e${epochs}_${short}_s${seed}_fcos"
  else
    fcos_name="e${epochs}_${short}_pre${seed}_ft${finetune_seed}_fcos"
  fi
  qsub_gpu_ids="${PRETRAIN_GPU_IDS//,/:}"
  fcos_gpu_ids="${FCOS_GPU_IDS:-0}"
  varlist="KIND=${kind},CONDITION=${condition},EPOCHS=${epochs},SEED=${seed},FINETUNE_SEED=${finetune_seed},RUN_SUFFIX=${RUN_SUFFIX},PROBE_ENV_PREFIX=${PROBE_ENV_PREFIX},PRETRAIN_DATA_ROOT=${PRETRAIN_DATA_ROOT},FCOS_DATA_ROOT=${FCOS_DATA_ROOT},ABCI3_CUDA_MODULE=${ABCI3_CUDA_MODULE},PRETRAIN_NODES=${PRETRAIN_NODES},PRETRAIN_GPU_IDS=${qsub_gpu_ids},PRETRAIN_BATCH_SIZE_PER_GPU=${PRETRAIN_BATCH_SIZE_PER_GPU},PRETRAIN_EVAL_INTERVAL=${PRETRAIN_EVAL_INTERVAL},PRETRAIN_CHECKPOINT_INTERVAL=${PRETRAIN_CHECKPOINT_INTERVAL},PRETRAIN_KEEP_CHECKPOINTS=${PRETRAIN_KEEP_CHECKPOINTS},PRETRAIN_LOG_INTERVAL=${PRETRAIN_LOG_INTERVAL},PRETRAIN_PROFILE_STEP_TIME=${PRETRAIN_PROFILE_STEP_TIME},FCOS_GPU_IDS=${fcos_gpu_ids},FCOS_NUM_EPOCHS=${FCOS_NUM_EPOCHS},FCOS_BATCH_SIZE_PER_GPU=${FCOS_BATCH_SIZE_PER_GPU},FCOS_LR=${FCOS_LR},FCOS_WEIGHT_DECAY=${FCOS_WEIGHT_DECAY},FCOS_LR_SCHEDULER=${FCOS_LR_SCHEDULER},USE_WANDB=${USE_WANDB},WANDB_MODE=${WANDB_MODE},DETERMINISTIC=${DETERMINISTIC},SKIP_EXISTING=1"

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
  if [[ -n "${pre_jobid}" ]]; then
    fcos_cmd+=(-W "depend=afterok:${pre_jobid}")
  fi
  fcos_cmd+=("${SCRIPT_DIR}/abci3_e300_gate_fcos.pbs")

  if [[ "${DRY_RUN}" == "1" ]]; then
    run_or_print "${fcos_cmd[@]}"
    printf "DRYRUN_FCOS_%02d\n" "${idx}"
  else
    cmd_output="$("${fcos_cmd[@]}")"
    cmd_output="${cmd_output%% *}"
    echo "[submitted] fcos ${kind}:${condition}:e${epochs}:preseed${seed}:ftseed${finetune_seed} job=${cmd_output} dep=${pre_jobid:-none}" >&2
    printf "%s\n" "${cmd_output}"
  fi
}

cd "${ROOT_DIR}"
if [[ "${DRY_RUN}" != "1" ]]; then
  printf "idx\tslot\tkind\tcondition\tepochs\tpretrain_seed\tfinetune_seed\tpretrain_job\tfcos_job\n" > "${SUBMIT_LOG_DIR}/submitted.tsv"
fi

declare -a slot_tail
for slot in $(seq 0 $((PRETRAIN_SLOTS - 1))); do
  slot_tail["${slot}"]=""
done

idx=0
pre_idx=0
for job in ${GATE_JOBS}; do
  IFS=':' read -r kind condition epochs seed finetune_seed <<< "${job}"
  finetune_seed="${finetune_seed:-${seed}}"
  slot=""
  pre_jobid=""
  fcos_jobid=""

  if [[ "${SUBMIT_PRETRAIN}" == "1" ]]; then
    final_checkpoint="$(pretrain_checkpoint_path "${kind}" "${condition}" "${epochs}" "${seed}")"
    if [[ -f "${final_checkpoint}" ]]; then
      echo "[info] skip pretrain submit; existing checkpoint=${final_checkpoint}" >&2
      slot="skip"
    else
      slot=$((pre_idx % PRETRAIN_SLOTS))
      pre_dependency="$(combine_dependency "${GLOBAL_DEPENDENCY}" "${slot_tail[${slot}]}")"
      pre_jobid="$(submit_pretrain "${idx}" "${slot}" "${pre_dependency}" "${kind}" "${condition}" "${epochs}" "${seed}")"
      slot_tail["${slot}"]="${pre_jobid}"
      pre_idx=$((pre_idx + 1))
    fi
  fi

  if [[ "${SUBMIT_FCOS}" == "1" ]]; then
    fcos_dependency="${pre_jobid}"
    if [[ -z "${fcos_dependency}" ]]; then
      fcos_dependency="${GLOBAL_DEPENDENCY}"
    fi
    fcos_jobid="$(submit_fcos "${idx}" "${fcos_dependency}" "${kind}" "${condition}" "${epochs}" "${seed}" "${finetune_seed}")"
  fi

  if [[ "${DRY_RUN}" != "1" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${idx}" "${slot}" "${kind}" "${condition}" "${epochs}" "${seed}" "${finetune_seed}" \
      "${pre_jobid:-none}" "${fcos_jobid:-none}" >> "${SUBMIT_LOG_DIR}/submitted.tsv"
  fi
  idx=$((idx + 1))
done

echo "[info] pipeline submit log dir: ${SUBMIT_LOG_DIR}"
if [[ "${SUBMIT_PRETRAIN}" == "1" ]]; then
  echo "[info] pretrain slots=${PRETRAIN_SLOTS} nodes_per_pretrain=${PRETRAIN_NODES} total_pretrain_nodes_if_full=$((PRETRAIN_SLOTS * PRETRAIN_NODES))"
else
  echo "[info] pretrain disabled; submitted FCOS-only jobs"
fi
