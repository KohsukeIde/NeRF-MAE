#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_probe_common.sh"

LOG_ROOT="${LOG_ROOT:-${PROBE_LOG_ROOT_DEFAULT}}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/nerfmae_shortcut_probe.chain.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/nerfmae_shortcut_probe.chain.pid}"
CHAIN_VARIANTS="${CHAIN_VARIANTS:-baseline,alpha_only,radiance_only,masked_only_rgb_loss}"
RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
RUN_FCOS_FINETUNE="${RUN_FCOS_FINETUNE:-0}"
RUN_FCOS_EVAL="${RUN_FCOS_EVAL:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
WAIT_BEFORE_CHAIN="${WAIT_BEFORE_CHAIN:-1}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
ALLOWED_GPU_IDS="${ALLOWED_GPU_IDS:-${GPU_IDS}}"
TRANSFER_GPU_COUNT="${TRANSFER_GPU_COUNT:-1}"
DATASET_NAME="${DATASET_NAME:-nerfmae}"
FCOS_DATASET_NAME="${FCOS_DATASET_NAME:-front3d}"
FCOS_SPLIT_NAME="${FCOS_SPLIT_NAME:-3dfront}"
FCOS_PERCENT_TRAIN="${FCOS_PERCENT_TRAIN:-1.0}"
FCOS_LR="${FCOS_LR:-1e-4}"
FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY:-1e-3}"
FCOS_LOG_INTERVAL="${FCOS_LOG_INTERVAL:-10}"
FCOS_EVAL_INTERVAL="${FCOS_EVAL_INTERVAL:-10}"
RESULT_SUFFIX="${RESULT_SUFFIX:-p${PERCENT_TRAIN:-0.1}_e${NUM_EPOCHS:-10}}"
PRETRAIN_RESULT_SUFFIX="${PRETRAIN_RESULT_SUFFIX:-${RESULT_SUFFIX}}"
FCOS_RESULT_SUFFIX="${FCOS_RESULT_SUFFIX:-fcos${FCOS_NUM_EPOCHS:-100}}"
SUMMARY_ROWS_TSV="${SUMMARY_ROWS_TSV:-${LOG_ROOT}/${CHAIN_NAME:-nerfmae_shortcut_probe}.rows.tsv}"
SUMMARY_JSON="${SUMMARY_JSON:-${LOG_ROOT}/${CHAIN_NAME:-nerfmae_shortcut_probe}.json}"
SUMMARY_CSV="${SUMMARY_CSV:-${LOG_ROOT}/${CHAIN_NAME:-nerfmae_shortcut_probe}.csv}"

cleanup() {
  local rc=$?
  probe_log "exit_code=${rc}" >> "${LOG_FILE}"
  rm -f "${PID_FILE}"
  exit "${rc}"
}
trap cleanup EXIT

mkdir -p "${LOG_ROOT}"
echo "$$" > "${PID_FILE}"

log() {
  probe_log "$*" | tee -a "${LOG_FILE}"
}

pretrain_save_name() {
  local variant="$1"
  case "${variant}" in
    baseline) echo "${DATASET_NAME}_all_${PRETRAIN_RESULT_SUFFIX}" ;;
    alpha_only) echo "${DATASET_NAME}_alpha_only_${PRETRAIN_RESULT_SUFFIX}" ;;
    radiance_only) echo "${DATASET_NAME}_radiance_only_${PRETRAIN_RESULT_SUFFIX}" ;;
    masked_only_rgb_loss) echo "${DATASET_NAME}_masked_only_rgb_loss_${PRETRAIN_RESULT_SUFFIX}" ;;
    *) probe_die "unknown variant=${variant}" ;;
  esac
}

fcos_save_name() {
  local variant="$1"
  case "${variant}" in
    baseline) echo "${DATASET_NAME}_all_${PRETRAIN_RESULT_SUFFIX}_${FCOS_RESULT_SUFFIX}" ;;
    alpha_only) echo "${DATASET_NAME}_alpha_only_${PRETRAIN_RESULT_SUFFIX}_${FCOS_RESULT_SUFFIX}" ;;
    radiance_only) echo "${DATASET_NAME}_radiance_only_${PRETRAIN_RESULT_SUFFIX}_${FCOS_RESULT_SUFFIX}" ;;
    masked_only_rgb_loss) echo "${DATASET_NAME}_masked_only_rgb_loss_${PRETRAIN_RESULT_SUFFIX}_${FCOS_RESULT_SUFFIX}" ;;
    *) probe_die "unknown variant=${variant}" ;;
  esac
}

pretrain_script() {
  local variant="$1"
  case "${variant}" in
    baseline) echo "${ROOT_DIR}/nerf_mae/train_mae3d.sh" ;;
    alpha_only) echo "${ROOT_DIR}/nerf_mae/probe_scripts/train_alpha_only.sh" ;;
    radiance_only) echo "${ROOT_DIR}/nerf_mae/probe_scripts/train_radiance_only.sh" ;;
    masked_only_rgb_loss) echo "${ROOT_DIR}/nerf_mae/probe_scripts/train_masked_only_rgb_loss.sh" ;;
    *) probe_die "unknown variant=${variant}" ;;
  esac
}

latest_fcos_checkpoint() {
  local save_dir="$1"
  [[ -d "${save_dir}" ]] || return 1
  find "${save_dir}" -maxdepth 1 -name 'model_best_ap50_ap25_*.pt' -printf '%T@ %p\n' \
    2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | awk '{print $2}'
}

append_summary_row() {
  local variant="$1"
  local pretrain_ckpt="$2"
  local fcos_ckpt="$3"
  local status="$4"
  local eval_json="$5"
  printf '%s\t%s\t%s\t%s\t%s\n' "${variant}" "${pretrain_ckpt}" "${fcos_ckpt}" "${status}" "${eval_json}" >> "${SUMMARY_ROWS_TSV}"
}

refresh_transfer_summary() {
  "${PROBE_PYTHON_BIN}" "${ROOT_DIR}/nerf_mae/tools/write_transfer_summary.py" \
    --rows-tsv "${SUMMARY_ROWS_TSV}" \
    --json-out "${SUMMARY_JSON}" \
    --csv-out "${SUMMARY_CSV}" \
    >> "${LOG_FILE}" 2>&1
}

ensure_transfer_preflight() {
  local variant ckpt

  log "preflight import sort_vertices"
  if ! (
    cd "${ROOT_DIR}/nerf_rpn"
    export PYTHONPATH="${ROOT_DIR}"
    "${PROBE_PYTHON_BIN}" - <<'PY'
import torch
import sort_vertices
print("sort_vertices import ok")
PY
  ) >> "${LOG_FILE}" 2>&1; then
    log "sort_vertices import missing; building rotated_iou"
    (
      cd "${ROOT_DIR}/nerf_rpn"
      export PROBE_ENV_PREFIX
      export PROBE_PYTHON_BIN
      bash "${ROOT_DIR}/nerf_rpn/build_rotated_iou.sh"
    ) >> "${LOG_FILE}" 2>&1
  fi

  log "preflight import run_fcos_pretrained"
  (
    cd "${ROOT_DIR}/nerf_rpn"
    export PYTHONPATH="${ROOT_DIR}"
    "${PROBE_PYTHON_BIN}" - <<'PY'
import run_fcos_pretrained
print("run_fcos_pretrained import ok")
PY
  ) >> "${LOG_FILE}" 2>&1

  IFS=',' read -r -a variants <<< "${CHAIN_VARIANTS}"
  for variant in "${variants[@]}"; do
    variant="${variant//[[:space:]]/}"
    [[ -z "${variant}" ]] && continue
    ckpt="${ROOT_DIR}/output/nerf_mae/results/$(pretrain_save_name "${variant}")/model_best.pt"
    if [[ -f "${ckpt}" ]]; then
      log "preflight pretrain ckpt ok variant=${variant} ckpt=${ckpt}"
    else
      log "preflight pretrain ckpt missing variant=${variant} ckpt=${ckpt}"
    fi
  done
}

run_pretrain_variant() {
  local variant="$1"
  local save_name save_path ckpt script

  save_name="$(pretrain_save_name "${variant}")"
  save_path="${ROOT_DIR}/output/nerf_mae/results/${save_name}"
  ckpt="${save_path}/model_best.pt"
  script="$(pretrain_script "${variant}")"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${ckpt}" ]]; then
    log "skip pretrain variant=${variant} existing_ckpt=${ckpt}"
    return 0
  fi

  log "start pretrain variant=${variant} save_name=${save_name}"
  (
    cd "${ROOT_DIR}/nerf_mae"
    export PATH="${PROBE_ENV_PREFIX}/bin:${PATH}"
    export PYTHONPATH="${ROOT_DIR}"
    export SAVE_NAME="${save_name}"
    export RUN_TAG="${save_name}"
    export GPU_IDS
    bash "${script}"
  ) >> "${LOG_FILE}" 2>&1

  [[ -f "${ckpt}" ]] || probe_die "missing pretrain checkpoint ${ckpt}"
  log "done pretrain variant=${variant} ckpt=${ckpt}"
}

run_fcos_variant() {
  local variant="$1"
  local save_name pretrain_ckpt transfer_save_name fcos_save_path best_ckpt eval_save_path
  local transfer_gpu status="completed"

  save_name="$(pretrain_save_name "${variant}")"
  pretrain_ckpt="${ROOT_DIR}/output/nerf_mae/results/${save_name}/model_best.pt"
  if [[ ! -f "${pretrain_ckpt}" ]]; then
    log "missing pretrain checkpoint variant=${variant} ckpt=${pretrain_ckpt}"
    append_summary_row "${variant}" "${pretrain_ckpt}" "" "failed_missing_pretrain" ""
    refresh_transfer_summary
    return 0
  fi

  transfer_save_name="$(fcos_save_name "${variant}")"
  fcos_save_path="${ROOT_DIR}/output/nerf_rpn/results/${transfer_save_name}"
  eval_save_path="${ROOT_DIR}/output/nerf_rpn/results/${transfer_save_name}_eval"

  transfer_gpu="$(
    probe_wait_for_idle_gpu_selection "${ALLOWED_GPU_IDS}" "${TRANSFER_GPU_COUNT}" \
      2> >(tee -a "${LOG_FILE}" >&2)
  )"
  log "start transfer variant=${variant} gpu=${transfer_gpu} save_name=${transfer_save_name}"

  if [[ "${SKIP_EXISTING}" == "1" ]] && latest_fcos_checkpoint "${fcos_save_path}" >/dev/null 2>&1; then
    log "skip fcos train variant=${variant} existing_dir=${fcos_save_path}"
  else
    if ! (
      cd "${ROOT_DIR}/nerf_rpn"
      export PATH="${PROBE_ENV_PREFIX}/bin:${PATH}"
      export PYTHONPATH="${ROOT_DIR}"
      export GPU_IDS="${transfer_gpu}"
      export DATASET_NAME="${FCOS_DATASET_NAME}"
      export SPLIT_NAME="${FCOS_SPLIT_NAME}"
      export DATA_ROOT="../dataset/finetune/${FCOS_DATASET_NAME}_rpn_data"
      export NUM_EPOCHS="${FCOS_NUM_EPOCHS}"
      export LR="${FCOS_LR}"
      export WEIGHT_DECAY="${FCOS_WEIGHT_DECAY}"
      export LOG_INTERVAL="${FCOS_LOG_INTERVAL}"
      export EVAL_INTERVAL="${FCOS_EVAL_INTERVAL}"
      export PERCENT_TRAIN="${FCOS_PERCENT_TRAIN}"
      export BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU}"
      export SAVE_NAME="${transfer_save_name}"
      export SAVE_PATH="../output/nerf_rpn/results/${transfer_save_name}"
      export RUN_TAG="${transfer_save_name}"
      export MAE_CHECKPOINT="../output/nerf_mae/results/${save_name}/model_best.pt"
      export WANDB_MODE="${WANDB_MODE:-offline}"
      bash "${ROOT_DIR}/nerf_rpn/train_fcos_pretrained.sh"
    ) >> "${LOG_FILE}" 2>&1; then
      status="failed_fcos_train"
      log "fcos train failed variant=${variant}"
    fi
  fi

  best_ckpt="$(latest_fcos_checkpoint "${fcos_save_path}" || true)"
  if [[ -z "${best_ckpt}" ]]; then
    append_summary_row "${variant}" "${pretrain_ckpt}" "" "${status}" ""
    refresh_transfer_summary
    return 0
  fi
  log "done fcos train variant=${variant} checkpoint=${best_ckpt} gpu=${transfer_gpu}"

  if [[ "${RUN_FCOS_EVAL}" != "1" ]]; then
    append_summary_row "${variant}" "${pretrain_ckpt}" "${best_ckpt}" "${status}" ""
    refresh_transfer_summary
    return 0
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${eval_save_path}/eval.json" ]]; then
    log "skip fcos eval variant=${variant} existing_eval=${eval_save_path}/eval.json"
    append_summary_row "${variant}" "${pretrain_ckpt}" "${best_ckpt}" "${status}" "${eval_save_path}/eval.json"
    refresh_transfer_summary
    return 0
  fi

  log "start fcos eval variant=${variant} gpu=${transfer_gpu}"
  if ! (
    cd "${ROOT_DIR}/nerf_rpn"
    export PATH="${PROBE_ENV_PREFIX}/bin:${PATH}"
    export PYTHONPATH="${ROOT_DIR}"
    export GPU_IDS="${FCOS_EVAL_GPU_IDS:-${transfer_gpu}}"
    export DATASET_NAME="${FCOS_DATASET_NAME}"
    export SPLIT_NAME="${FCOS_SPLIT_NAME}"
    export DATA_ROOT="../dataset/finetune/${FCOS_DATASET_NAME}_rpn_data"
    export SAVE_NAME="${transfer_save_name}_eval"
    export SAVE_PATH="../output/nerf_rpn/results/${transfer_save_name}_eval"
    export CHECKPOINT="${best_ckpt}"
    bash "${ROOT_DIR}/nerf_rpn/test_fcos_pretrained.sh"
  ) >> "${LOG_FILE}" 2>&1; then
    status="failed_fcos_eval"
    log "fcos eval failed variant=${variant}"
  fi

  if [[ ! -f "${eval_save_path}/eval.json" ]]; then
    append_summary_row "${variant}" "${pretrain_ckpt}" "${best_ckpt}" "${status}" ""
    refresh_transfer_summary
    return 0
  fi
  log "done fcos eval variant=${variant} eval_json=${eval_save_path}/eval.json"
  append_summary_row "${variant}" "${pretrain_ckpt}" "${best_ckpt}" "${status}" "${eval_save_path}/eval.json"
  refresh_transfer_summary
}

mkdir -p "${LOG_ROOT}"
mkdir -p "$(dirname "${SUMMARY_ROWS_TSV}")"
: > "${SUMMARY_ROWS_TSV}"
refresh_transfer_summary

log "start shortcut probe chain variants=${CHAIN_VARIANTS} run_pretrain=${RUN_PRETRAIN} run_fcos=${RUN_FCOS_FINETUNE} blocker_tmux=${BLOCK_ON_TMUX_SESSIONS:-none}"

if [[ "${RUN_PRETRAIN}" == "1" && "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
  probe_wait_for_runway | tee -a "${LOG_FILE}"
fi

if [[ "${RUN_FCOS_FINETUNE}" == "1" ]]; then
  ensure_transfer_preflight
fi

IFS=',' read -r -a variants <<< "${CHAIN_VARIANTS}"
for variant in "${variants[@]}"; do
  variant="${variant//[[:space:]]/}"
  [[ -z "${variant}" ]] && continue
  if [[ "${RUN_PRETRAIN}" == "1" ]]; then
    run_pretrain_variant "${variant}"
  fi
  if [[ "${RUN_FCOS_FINETUNE}" == "1" ]]; then
    run_fcos_variant "${variant}"
  fi
done

log "shortcut probe chain done"
