#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_probe_common.sh"

CHAIN_NAME="${CHAIN_NAME:-nerfmae_shortcut_probe}"
LOG_ROOT="${LOG_ROOT:-${PROBE_LOG_ROOT_DEFAULT}}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${CHAIN_NAME}.chain.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${CHAIN_NAME}.chain.pid}"
TMUX_SESSION="${TMUX_SESSION:-${CHAIN_NAME//[^[:alnum:]_]/_}_chain}"
LAUNCH_MODE="${LAUNCH_MODE:-tmux}"
ENV_FILE="${ENV_FILE:-${LOG_ROOT}/.${TMUX_SESSION}.env}"

export CHAIN_NAME LOG_ROOT LOG_FILE PID_FILE
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export ALLOWED_GPU_IDS="${ALLOWED_GPU_IDS:-${GPU_IDS}}"
export DATASET_NAME="${DATASET_NAME:-nerfmae}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export PERCENT_TRAIN="${PERCENT_TRAIN:-0.1}"
export WAIT_BEFORE_CHAIN="${WAIT_BEFORE_CHAIN:-1}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export CHAIN_VARIANTS="${CHAIN_VARIANTS:-baseline,alpha_only,radiance_only,masked_only_rgb_loss}"
export RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
export RUN_FCOS_FINETUNE="${RUN_FCOS_FINETUNE:-0}"
export RUN_FCOS_EVAL="${RUN_FCOS_EVAL:-0}"
export FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS:-100}"
export FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU:-2}"
export TRANSFER_GPU_COUNT="${TRANSFER_GPU_COUNT:-1}"
export BLOCK_ON_TMUX_SESSIONS="${BLOCK_ON_TMUX_SESSIONS-pcp_worldvis_base_100ep_chain}"
export BLOCK_ON_PID_FILES="${BLOCK_ON_PID_FILES-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geopcp/full_chain/pcp_worldvis_base_100ep.chain.pid}"
export WAIT_MEMORY_USED_MAX_MIB="${WAIT_MEMORY_USED_MAX_MIB:-512}"
export WAIT_UTIL_MAX="${WAIT_UTIL_MAX:-10}"
export WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-30}"
export WAIT_STABLE_SECONDS="${WAIT_STABLE_SECONDS:-180}"
export WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-0}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export RESULT_SUFFIX="${RESULT_SUFFIX:-p${PERCENT_TRAIN}_e${NUM_EPOCHS}}"
export PRETRAIN_RESULT_SUFFIX="${PRETRAIN_RESULT_SUFFIX:-${RESULT_SUFFIX}}"
export FCOS_RESULT_SUFFIX="${FCOS_RESULT_SUFFIX:-fcos${FCOS_NUM_EPOCHS}}"
export SUMMARY_ROWS_TSV="${SUMMARY_ROWS_TSV:-${LOG_ROOT}/${CHAIN_NAME}.rows.tsv}"
export SUMMARY_JSON="${SUMMARY_JSON:-${LOG_ROOT}/${CHAIN_NAME}.json}"
export SUMMARY_CSV="${SUMMARY_CSV:-${LOG_ROOT}/${CHAIN_NAME}.csv}"
export PROBE_ENV_PREFIX PROBE_PYTHON_BIN PROBE_REPO_ROOT PROBE_RUNTIME_ROOT

mkdir -p "${LOG_ROOT}"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "[info] shortcut probe chain already running (pid=${existing_pid})"
    echo "[info] log=${LOG_FILE}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  bash "${SCRIPT_DIR}/_run_shortcut_probe_chain_inner.sh" 2>&1 | tee -a "${LOG_FILE}"
  exit "${PIPESTATUS[0]}"
fi

case "${LAUNCH_MODE}" in
  tmux)
    command -v tmux >/dev/null 2>&1 || probe_die "tmux not found"
    if env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
      echo "[info] shortcut probe chain already running in tmux session=${TMUX_SESSION}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    : > "${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
    for name in \
      CHAIN_NAME LOG_ROOT LOG_FILE PID_FILE GPU_IDS DATASET_NAME NUM_EPOCHS PERCENT_TRAIN WAIT_BEFORE_CHAIN LOG_INTERVAL EVAL_INTERVAL \
      BATCH_SIZE_PER_GPU USE_WANDB WANDB_MODE CHAIN_VARIANTS RUN_PRETRAIN RUN_FCOS_FINETUNE RUN_FCOS_EVAL FCOS_NUM_EPOCHS \
      FCOS_BATCH_SIZE_PER_GPU ALLOWED_GPU_IDS TRANSFER_GPU_COUNT BLOCK_ON_TMUX_SESSIONS BLOCK_ON_PID_FILES WAIT_MEMORY_USED_MAX_MIB WAIT_UTIL_MAX \
      WAIT_POLL_SECONDS WAIT_STABLE_SECONDS WAIT_TIMEOUT_SECONDS SKIP_EXISTING RESULT_SUFFIX PRETRAIN_RESULT_SUFFIX FCOS_RESULT_SUFFIX \
      SUMMARY_ROWS_TSV SUMMARY_JSON SUMMARY_CSV PROBE_ENV_PREFIX PROBE_PYTHON_BIN PROBE_REPO_ROOT PROBE_RUNTIME_ROOT \
      TMPDIR TMP TEMP XDG_CACHE_HOME TORCH_HOME TORCH_EXTENSIONS_DIR WANDB_DIR
    do
      printf 'export %s=%q\n' "${name}" "${!name:-}" >> "${ENV_FILE}"
    done
    env -u LD_LIBRARY_PATH -u PYTHONPATH tmux new-session -d -s "${TMUX_SESSION}" \
      "cd '${ROOT_DIR}' && source '${ENV_FILE}' && bash '${SCRIPT_DIR}/_run_shortcut_probe_chain_inner.sh'"
    sleep 1
    env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null || probe_die "failed to start tmux session=${TMUX_SESSION}"
    echo "[info] started detached shortcut probe chain in tmux"
    echo "[info] session=${TMUX_SESSION}"
    echo "[info] log=${LOG_FILE}"
    ;;
  nohup)
    nohup bash -lc \
      "cd '${ROOT_DIR}' && bash '${SCRIPT_DIR}/_run_shortcut_probe_chain_inner.sh'" \
      >> "${LOG_FILE}" 2>&1 < /dev/null &
    child_pid=$!
    echo "${child_pid}" > "${PID_FILE}"
    sleep 1
    kill -0 "${child_pid}" 2>/dev/null || probe_die "failed to start shortcut probe chain"
    echo "[info] started detached shortcut probe chain"
    echo "[info] pid=${child_pid}"
    echo "[info] log=${LOG_FILE}"
    ;;
  *)
    probe_die "unsupported LAUNCH_MODE=${LAUNCH_MODE}"
    ;;
esac
