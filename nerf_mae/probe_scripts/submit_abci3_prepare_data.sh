#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ABCI_GROUP="${ABCI_GROUP:-gag51404}"
DATA_QUEUE="${DATA_QUEUE:-rt_HC}"
DATA_WALLTIME="${DATA_WALLTIME:-24:00:00}"
PROBE_ENV_PREFIX="${PROBE_ENV_PREFIX:-${ROOT_DIR}/.venv-abci3}"
DATA_WORK_ROOT="${DATA_WORK_ROOT:-${ROOT_DIR}/dataset/_downloads}"
SUBMIT_LOG_DIR="${SUBMIT_LOG_DIR:-${ROOT_DIR}/output/launcher/abci3_prepare_data}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${SUBMIT_LOG_DIR}"

varlist="ROOT_DIR=${ROOT_DIR},PROBE_ENV_PREFIX=${PROBE_ENV_PREFIX},DATA_WORK_ROOT=${DATA_WORK_ROOT}"
cmd=(
  qsub
  -P "${ABCI_GROUP}"
  -q "${DATA_QUEUE}"
  -l select=1
  -l "walltime=${DATA_WALLTIME}"
  -N nerfmae_data
  -j oe
  -o "${SUBMIT_LOG_DIR}/nerfmae_data.pbs.log"
  -v "${varlist}"
  "${SCRIPT_DIR}/abci3_prepare_data.pbs"
)

cd "${ROOT_DIR}"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[dry-run]'; printf ' %q' "${cmd[@]}"; printf '\n'
else
  jobid="$("${cmd[@]}")"
  jobid="${jobid%% *}"
  echo "[submitted] data prep job=${jobid}"
  echo "${jobid}" > "${SUBMIT_LOG_DIR}/jobid.txt"
fi
