#!/usr/bin/env bash
set -euo pipefail

probe_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

PROBE_REPO_ROOT="${PROBE_REPO_ROOT:-$(probe_repo_root)}"
PROBE_ENV_NAME="${PROBE_ENV_NAME:-nerf-mae-shortcut-probe}"
PROBE_ENV_PREFIX="${PROBE_ENV_PREFIX:-/home/minesawa/anaconda3/envs/${PROBE_ENV_NAME}}"
PROBE_PYTHON_BIN="${PROBE_PYTHON_BIN:-${PROBE_ENV_PREFIX}/bin/python}"
PROBE_LOG_ROOT_DEFAULT="${PROBE_LOG_ROOT_DEFAULT:-/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/launcher}"
PROBE_RUNTIME_ROOT="${PROBE_RUNTIME_ROOT:-/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/runtime}"
PROBE_GPU_REPORT=""

export TMPDIR="${TMPDIR:-${PROBE_RUNTIME_ROOT}/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROBE_RUNTIME_ROOT}/cache}"
export TORCH_HOME="${TORCH_HOME:-${PROBE_RUNTIME_ROOT}/torch_home}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${PROBE_RUNTIME_ROOT}/torch_extensions}"
export WANDB_DIR="${WANDB_DIR:-/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/wandb}"

mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${TORCH_EXTENSIONS_DIR}" "${WANDB_DIR}"

probe_die() {
  echo "[error] $*" >&2
  exit 1
}

probe_timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

probe_log() {
  printf "[launcher] %s %s\n" "$(probe_timestamp)" "$*"
}

probe_require_python() {
  [[ -x "${PROBE_PYTHON_BIN}" ]] || probe_die "python not found: ${PROBE_PYTHON_BIN}"
}

probe_count_gpus() {
  local spec="${1:-}"
  local count=0
  local token start end
  IFS=',' read -r -a tokens <<< "${spec}"
  for token in "${tokens[@]}"; do
    token="${token//[[:space:]]/}"
    [[ -z "${token}" ]] && continue
    if [[ "${token}" == *-* ]]; then
      IFS='-' read -r start end <<< "${token}"
      count=$((count + end - start + 1))
    else
      count=$((count + 1))
    fi
  done
  [[ "${count}" -gt 0 ]] || count=1
  printf "%s\n" "${count}"
}

probe_gpu_selected() {
  local index="$1"
  local spec="${2:-}"
  local token start end
  IFS=',' read -r -a tokens <<< "${spec}"
  for token in "${tokens[@]}"; do
    token="${token//[[:space:]]/}"
    [[ -z "${token}" ]] && continue
    if [[ "${token}" == *-* ]]; then
      IFS='-' read -r start end <<< "${token}"
      if (( index >= start && index <= end )); then
        return 0
      fi
    elif (( index == token )); then
      return 0
    fi
  done
  return 1
}

probe_tmux_session_active() {
  local session="$1"
  [[ -n "${session}" ]] || return 1
  command -v tmux >/dev/null 2>&1 || return 1
  env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${session}" 2>/dev/null
}

probe_pid_file_active() {
  local path="$1"
  local pid
  [[ -n "${path}" ]] || return 1
  [[ -f "${path}" ]] || return 1
  pid="$(cat "${path}" 2>/dev/null || true)"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

probe_any_blocker_active() {
  local blocker
  IFS=',' read -r -a blockers <<< "${BLOCK_ON_TMUX_SESSIONS:-}"
  for blocker in "${blockers[@]}"; do
    blocker="${blocker//[[:space:]]/}"
    [[ -z "${blocker}" ]] && continue
    if probe_tmux_session_active "${blocker}"; then
      echo "tmux:${blocker}"
      return 0
    fi
  done

  IFS=',' read -r -a blockers <<< "${BLOCK_ON_PID_FILES:-}"
  for blocker in "${blockers[@]}"; do
    blocker="${blocker//[[:space:]]/}"
    [[ -z "${blocker}" ]] && continue
    if probe_pid_file_active "${blocker}"; then
      echo "pid:${blocker}"
      return 0
    fi
  done
  return 1
}

probe_gpus_idle() {
  local gpu_spec="${GPU_IDS:-0}"
  local max_mem="${WAIT_MEMORY_USED_MAX_MIB:-512}"
  local max_util="${WAIT_UTIL_MAX:-10}"
  local row index mem util
  local report=()

  while IFS=',' read -r index mem util; do
    index="${index//[[:space:]]/}"
    mem="${mem//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    probe_gpu_selected "${index}" "${gpu_spec}" || continue
    report+=("gpu${index}:mem=${mem}MiB util=${util}%")
    if (( mem > max_mem || util > max_util )); then
      printf "%s\n" "${report[*]}"
      return 1
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)

  printf "%s\n" "${report[*]}"
  return 0
}

probe_select_idle_gpus() {
  local gpu_spec="${1:-${ALLOWED_GPU_IDS:-${GPU_IDS:-0}}}"
  local need_count="${2:-1}"
  local max_mem="${WAIT_MEMORY_USED_MAX_MIB:-512}"
  local max_util="${WAIT_UTIL_MAX:-10}"
  local index mem util
  local report=()
  local idle=()
  local selected=()

  while IFS=',' read -r index mem util; do
    index="${index//[[:space:]]/}"
    mem="${mem//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    probe_gpu_selected "${index}" "${gpu_spec}" || continue
    report+=("gpu${index}:mem=${mem}MiB util=${util}%")
    if (( mem <= max_mem && util <= max_util )); then
      idle+=("${index}")
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)

  PROBE_GPU_REPORT="${report[*]}"
  if (( ${#idle[@]} < need_count )); then
    return 1
  fi
  selected=("${idle[@]:0:need_count}")
  printf "%s\n" "$(IFS=,; echo "${selected[*]}")"
  return 0
}

probe_wait_for_idle_gpu_selection() {
  local gpu_spec="${1:-${ALLOWED_GPU_IDS:-${GPU_IDS:-0}}}"
  local need_count="${2:-${TRANSFER_GPU_COUNT:-1}}"
  local poll="${WAIT_POLL_SECONDS:-30}"
  local stable_seconds="${WAIT_STABLE_SECONDS:-180}"
  local timeout="${WAIT_TIMEOUT_SECONDS:-0}"
  local start_ts now stable_start=0
  local blocker candidate last_candidate=""

  start_ts="$(date +%s)"
  while true; do
    if blocker="$(probe_any_blocker_active)"; then
      probe_log "waiting for blocker ${blocker}" >&2
      stable_start=0
      last_candidate=""
      sleep "${poll}"
      continue
    fi

    if candidate="$(probe_select_idle_gpus "${gpu_spec}" "${need_count}")"; then
      now="$(date +%s)"
      if [[ "${candidate}" != "${last_candidate}" ]]; then
        last_candidate="${candidate}"
        stable_start="${now}"
        probe_log "candidate transfer gpu selection=${candidate}; starting stability window (${stable_seconds}s) ${PROBE_GPU_REPORT}" >&2
      elif (( now - stable_start >= stable_seconds )); then
        probe_log "transfer gpu selection ready=${candidate} ${PROBE_GPU_REPORT}" >&2
        printf "%s\n" "${candidate}"
        return 0
      fi
    else
      [[ -n "${PROBE_GPU_REPORT}" ]] && probe_log "waiting for idle transfer gpu(s) ${PROBE_GPU_REPORT}" >&2
      stable_start=0
      last_candidate=""
    fi

    if (( timeout > 0 )); then
      now="$(date +%s)"
      if (( now - start_ts >= timeout )); then
        probe_die "timed out waiting for idle GPU selection after ${timeout}s"
      fi
    fi
    sleep "${poll}"
  done
}

probe_wait_for_runway() {
  local poll="${WAIT_POLL_SECONDS:-30}"
  local stable_seconds="${WAIT_STABLE_SECONDS:-180}"
  local timeout="${WAIT_TIMEOUT_SECONDS:-0}"
  local start_ts now stable_start=0
  local blocker status

  start_ts="$(date +%s)"
  while true; do
    if blocker="$(probe_any_blocker_active)"; then
      probe_log "waiting for blocker ${blocker}"
      stable_start=0
      sleep "${poll}"
      continue
    fi

    if status="$(probe_gpus_idle)"; then
      now="$(date +%s)"
      if (( stable_start == 0 )); then
        stable_start="${now}"
        probe_log "selected GPUs look idle; starting stability window (${stable_seconds}s) ${status}"
      elif (( now - stable_start >= stable_seconds )); then
        probe_log "runway clear on GPUs ${GPU_IDS:-0} ${status}"
        return 0
      fi
    else
      [[ -n "${status}" ]] && probe_log "waiting for GPUs to clear ${status}"
      stable_start=0
    fi

    if (( timeout > 0 )); then
      now="$(date +%s)"
      if (( now - start_ts >= timeout )); then
        probe_die "timed out waiting for GPUs after ${timeout}s"
      fi
    fi
    sleep "${poll}"
  done
}
