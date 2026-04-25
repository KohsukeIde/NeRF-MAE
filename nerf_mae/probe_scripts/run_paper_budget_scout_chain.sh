#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_probe_common.sh"

CHAIN_NAME="${CHAIN_NAME:-nerfmae_paper_budget_scout}"
CHAIN_LOG_ROOT="${CHAIN_LOG_ROOT:-${PROBE_LOG_ROOT_DEFAULT}}"
CHAIN_LOG_FILE="${CHAIN_LOG_FILE:-${CHAIN_LOG_ROOT}/${CHAIN_NAME}.chain.log}"
CHAIN_PID_FILE="${CHAIN_PID_FILE:-${CHAIN_LOG_ROOT}/${CHAIN_NAME}.chain.pid}"
TMUX_SESSION="${TMUX_SESSION:-${CHAIN_NAME//[^[:alnum:]_]/_}_chain}"

WAIT_BEFORE_CHAIN="${WAIT_BEFORE_CHAIN:-1}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
WAIT_STABLE_SECONDS="${WAIT_STABLE_SECONDS:-300}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-0}"
WAIT_MEMORY_USED_MAX_MIB="${WAIT_MEMORY_USED_MAX_MIB:-2048}"
WAIT_UTIL_MAX="${WAIT_UTIL_MAX:-10}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
DETERMINISTIC="${DETERMINISTIC:-1}"

SCOUT_SEED="${SCOUT_SEED:-1}"
SCOUT_CONDITIONS="${SCOUT_CONDITIONS:-baseline,alpha_target_only}"

PRETRAIN_GPU_IDS="${PRETRAIN_GPU_IDS:-0,1,2,3}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1200}"
PRETRAIN_PERCENT_TRAIN="${PRETRAIN_PERCENT_TRAIN:-1.0}"
PRETRAIN_BATCH_SIZE_PER_GPU="${PRETRAIN_BATCH_SIZE_PER_GPU:-4}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-3}"
PRETRAIN_WEIGHT_DECAY="${PRETRAIN_WEIGHT_DECAY:-0.0}"
PRETRAIN_LOG_INTERVAL="${PRETRAIN_LOG_INTERVAL:-30}"
PRETRAIN_EVAL_INTERVAL="${PRETRAIN_EVAL_INTERVAL:-10}"

FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS:-1000}"
FCOS_ALLOWED_GPU_IDS="${FCOS_ALLOWED_GPU_IDS:-0,1,2,3}"
FCOS_MAX_PARALLEL="${FCOS_MAX_PARALLEL:-2}"
FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU:-2}"
FCOS_LR="${FCOS_LR:-1e-4}"
FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY:-1e-3}"
FCOS_LR_SCHEDULER="${FCOS_LR_SCHEDULER:-onecycle_epoch}"

EXPERIMENT_LOG="${EXPERIMENT_LOG:-${ROOT_DIR}/SHORTCUT_PROBE_EXPERIMENT_LOG.md}"

mkdir -p "${CHAIN_LOG_ROOT}"

if [[ "${FOREGROUND:-0}" != "1" ]]; then
  if env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
    echo "[info] chain already running in tmux session=${TMUX_SESSION}"
    echo "[info] log=${CHAIN_LOG_FILE}"
    exit 0
  fi
  env -u LD_LIBRARY_PATH -u PYTHONPATH tmux new-session -d -s "${TMUX_SESSION}" \
    "cd '${ROOT_DIR}' && FOREGROUND=1 CHAIN_LOG_ROOT='${CHAIN_LOG_ROOT}' CHAIN_LOG_FILE='${CHAIN_LOG_FILE}' CHAIN_PID_FILE='${CHAIN_PID_FILE}' WAIT_BEFORE_CHAIN='${WAIT_BEFORE_CHAIN}' WAIT_POLL_SECONDS='${WAIT_POLL_SECONDS}' WAIT_STABLE_SECONDS='${WAIT_STABLE_SECONDS}' WAIT_TIMEOUT_SECONDS='${WAIT_TIMEOUT_SECONDS}' WAIT_MEMORY_USED_MAX_MIB='${WAIT_MEMORY_USED_MAX_MIB}' WAIT_UTIL_MAX='${WAIT_UTIL_MAX}' SKIP_EXISTING='${SKIP_EXISTING}' USE_WANDB='${USE_WANDB}' WANDB_MODE='${WANDB_MODE}' DETERMINISTIC='${DETERMINISTIC}' SCOUT_SEED='${SCOUT_SEED}' SCOUT_CONDITIONS='${SCOUT_CONDITIONS}' PRETRAIN_GPU_IDS='${PRETRAIN_GPU_IDS}' PRETRAIN_EPOCHS='${PRETRAIN_EPOCHS}' PRETRAIN_PERCENT_TRAIN='${PRETRAIN_PERCENT_TRAIN}' PRETRAIN_BATCH_SIZE_PER_GPU='${PRETRAIN_BATCH_SIZE_PER_GPU}' PRETRAIN_LR='${PRETRAIN_LR}' PRETRAIN_WEIGHT_DECAY='${PRETRAIN_WEIGHT_DECAY}' PRETRAIN_LOG_INTERVAL='${PRETRAIN_LOG_INTERVAL}' PRETRAIN_EVAL_INTERVAL='${PRETRAIN_EVAL_INTERVAL}' FCOS_NUM_EPOCHS='${FCOS_NUM_EPOCHS}' FCOS_ALLOWED_GPU_IDS='${FCOS_ALLOWED_GPU_IDS}' FCOS_MAX_PARALLEL='${FCOS_MAX_PARALLEL}' FCOS_BATCH_SIZE_PER_GPU='${FCOS_BATCH_SIZE_PER_GPU}' FCOS_LR='${FCOS_LR}' FCOS_WEIGHT_DECAY='${FCOS_WEIGHT_DECAY}' FCOS_LR_SCHEDULER='${FCOS_LR_SCHEDULER}' BLOCK_ON_TMUX_SESSIONS='${BLOCK_ON_TMUX_SESSIONS:-}' BLOCK_ON_PID_FILES='${BLOCK_ON_PID_FILES:-}' EXPERIMENT_LOG='${EXPERIMENT_LOG}' bash '${SCRIPT_DIR}/run_paper_budget_scout_chain.sh'"
  echo "[info] started detached chain"
  echo "[info] session=${TMUX_SESSION}"
  echo "[info] log=${CHAIN_LOG_FILE}"
  exit 0
fi

touch "${CHAIN_LOG_FILE}"
exec > >(tee -a "${CHAIN_LOG_FILE}") 2>&1
echo "$$" > "${CHAIN_PID_FILE}"

cleanup() {
  local rc=$?
  probe_log "exit_code=${rc}"
  rm -f "${CHAIN_PID_FILE}"
  exit "${rc}"
}
trap cleanup EXIT

log() {
  probe_log "$*"
}

parse_csv_to_array() {
  local csv="$1"
  local -n out_ref=$2
  IFS=',' read -r -a out_ref <<< "${csv}"
  local i
  for i in "${!out_ref[@]}"; do
    out_ref[$i]="${out_ref[$i]//[[:space:]]/}"
  done
}

condition_save_name() {
  local condition="$1"
  case "${condition}" in
    baseline)
      printf "nerfmae_all_p1.0_e%s_seed%s" "${PRETRAIN_EPOCHS}" "${SCOUT_SEED}"
      ;;
    alpha_target_only)
      printf "nerfmae_alpha_target_only_p1.0_e%s_seed%s" "${PRETRAIN_EPOCHS}" "${SCOUT_SEED}"
      ;;
    *)
      probe_die "unknown condition=${condition}"
      ;;
  esac
}

condition_script() {
  local condition="$1"
  case "${condition}" in
    baseline)
      printf "train_mae3d.sh"
      ;;
    alpha_target_only)
      printf "probe_scripts/train_alpha_target_only.sh"
      ;;
    *)
      probe_die "unknown condition=${condition}"
      ;;
  esac
}

pretrain_epoch_exists() {
  local save_name="$1"
  local epoch="$2"
  [[ -f "${ROOT_DIR}/output/nerf_mae/results/${save_name}/epoch_${epoch}.pt" ]]
}

eval_exists() {
  local save_name="$1"
  [[ -f "${ROOT_DIR}/output/nerf_rpn/results/${save_name}_eval/eval.json" ]]
}

run_pretrain_script() {
  local condition="$1"
  local save_name="$2"
  local script_rel="$3"
  local log_file="$4"

  (
    cd "${ROOT_DIR}/nerf_mae"
    PATH="${PROBE_ENV_PREFIX}/bin:${PATH}" \
      PYTHONPATH="${ROOT_DIR}" \
      SAVE_NAME="${save_name}" \
      RUN_TAG="${save_name}" \
      GPU_IDS="${PRETRAIN_GPU_IDS}" \
      NUM_EPOCHS="${PRETRAIN_EPOCHS}" \
      PERCENT_TRAIN="${PRETRAIN_PERCENT_TRAIN}" \
      BATCH_SIZE_PER_GPU="${PRETRAIN_BATCH_SIZE_PER_GPU}" \
      LR="${PRETRAIN_LR}" \
      WEIGHT_DECAY="${PRETRAIN_WEIGHT_DECAY}" \
      LOG_INTERVAL="${PRETRAIN_LOG_INTERVAL}" \
      EVAL_INTERVAL="${PRETRAIN_EVAL_INTERVAL}" \
      USE_WANDB="${USE_WANDB}" \
      WANDB_MODE="${WANDB_MODE}" \
      SEED="${SCOUT_SEED}" \
      DETERMINISTIC="${DETERMINISTIC}" \
      PROBE_ALPHA_TARGET=keep \
      bash "${script_rel}"
  ) >> "${log_file}" 2>&1

  if ! pretrain_epoch_exists "${save_name}" "${PRETRAIN_EPOCHS}"; then
    probe_die "pretrain finished but checkpoint is missing condition=${condition} save=${save_name} checkpoint=epoch_${PRETRAIN_EPOCHS}.pt"
  fi
}

run_pretrains() {
  local conditions=()
  parse_csv_to_array "${SCOUT_CONDITIONS}" conditions
  local condition save_name script_rel

  log "step_paper_budget_pretrain start conditions=${SCOUT_CONDITIONS}"
  for condition in "${conditions[@]}"; do
    save_name="$(condition_save_name "${condition}")"
    script_rel="$(condition_script "${condition}")"
    if [[ "${SKIP_EXISTING}" == "1" ]] && pretrain_epoch_exists "${save_name}" "${PRETRAIN_EPOCHS}"; then
      log "skip pretrain condition=${condition} save=${save_name}"
      continue
    fi
    if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
      GPU_IDS="${PRETRAIN_GPU_IDS}" probe_wait_for_runway
    fi
    log "pretrain start condition=${condition} save=${save_name} gpus=${PRETRAIN_GPU_IDS} epochs=${PRETRAIN_EPOCHS} percent_train=${PRETRAIN_PERCENT_TRAIN}"
    run_pretrain_script \
      "${condition}" \
      "${save_name}" \
      "${script_rel}" \
      "${CHAIN_LOG_ROOT}/${CHAIN_NAME}.${save_name}.pretrain.log"
    log "pretrain done condition=${condition} save=${save_name}"
  done
  log "step_paper_budget_pretrain done"
}

run_job_probe() {
  local variant="$1"
  local gpu_ids="$2"
  local pretrain_save_name="$3"
  local pretrain_checkpoint="$4"
  local save_name="$5"
  local log_file="$6"
  (
    cd "${ROOT_DIR}"
    PRETRAIN_SAVE_NAME="${pretrain_save_name}" \
      VARIANT_NAME="${variant}" \
      PRETRAIN_CHECKPOINT="${pretrain_checkpoint}" \
      GPU_IDS="${gpu_ids}" \
      DATASET_NAME=front3d \
      SPLIT_NAME=3dfront \
      PERCENT_TRAIN=1.0 \
      FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS}" \
      FCOS_LR="${FCOS_LR}" \
      FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY}" \
      FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU}" \
      USE_WANDB="${USE_WANDB}" \
      WANDB_MODE="${WANDB_MODE}" \
      SAVE_NAME="${save_name}" \
      RUN_TAG="${save_name}" \
      SEED="${SCOUT_SEED}" \
      DETERMINISTIC="${DETERMINISTIC}" \
      LR_SCHEDULER="${FCOS_LR_SCHEDULER}" \
      bash nerf_rpn/run_fcos_probe_variant.sh
  ) >> "${log_file}" 2>&1 &
  RUN_VARIANT_PID=$!
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

select_idle_gpu_wave() {
  local max_count="$1"
  local n candidate
  for (( n=max_count; n>=1; n-- )); do
    if candidate="$(probe_select_idle_gpus "${FCOS_ALLOWED_GPU_IDS}" "${n}")"; then
      printf "%s\n" "${candidate}"
      return 0
    fi
  done
  return 1
}

run_fcos_jobs() {
  local conditions=()
  parse_csv_to_array "${SCOUT_CONDITIONS}" conditions
  local pending_jobs=()
  local remaining_jobs=()
  local condition pretrain_save checkpoint save_name job

  for condition in "${conditions[@]}"; do
    pretrain_save="$(condition_save_name "${condition}")"
    checkpoint="../output/nerf_mae/results/${pretrain_save}/epoch_${PRETRAIN_EPOCHS}.pt"
    save_name="${pretrain_save}_epoch${PRETRAIN_EPOCHS}_sched_epoch_seed${SCOUT_SEED}_fcos${FCOS_NUM_EPOCHS}"
    if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${save_name}"; then
      log "skip fcos condition=${condition} save=${save_name}"
      continue
    fi
    pending_jobs+=("${condition}:${pretrain_save}:${checkpoint}:${save_name}")
  done

  if (( ${#pending_jobs[@]} == 0 )); then
    log "skip fcos; all evals exist"
    return 0
  fi

  remaining_jobs=("${pending_jobs[@]}")
  log "step_paper_budget_fcos start jobs=${#remaining_jobs[@]}"
  while (( ${#remaining_jobs[@]} > 0 )); do
    if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
      probe_wait_for_idle_gpu_selection "${FCOS_ALLOWED_GPU_IDS}" 1 >/dev/null
    fi

    local max_wave="${FCOS_MAX_PARALLEL}"
    if (( max_wave > ${#remaining_jobs[@]} )); then
      max_wave="${#remaining_jobs[@]}"
    fi
    local gpu_csv
    gpu_csv="$(select_idle_gpu_wave "${max_wave}")" || probe_die "no idle GPU selected for FCOS"
    local gpus=()
    parse_csv_to_array "${gpu_csv}" gpus
    log "fcos wave start gpus=${gpu_csv} jobs=${#gpus[@]}/${#remaining_jobs[@]}"

    local pids=()
    local names=()
    local next_jobs=()
    local idx=0
    for job in "${remaining_jobs[@]}"; do
      if (( idx < ${#gpus[@]} )); then
        IFS=':' read -r condition pretrain_save checkpoint save_name <<< "${job}"
        run_job_probe \
          "${condition}" \
          "${gpus[$idx]}" \
          "${pretrain_save}" \
          "${checkpoint}" \
          "${save_name}" \
          "${CHAIN_LOG_ROOT}/${CHAIN_NAME}.${save_name}.log"
        pids+=("${RUN_VARIANT_PID}")
        names+=("${save_name}")
        idx=$((idx + 1))
      else
        next_jobs+=("${job}")
      fi
    done

    wait_for_pids pids names
    remaining_jobs=("${next_jobs[@]}")
  done
  log "step_paper_budget_fcos done"
}

append_result_log() {
  ROOT_DIR="${ROOT_DIR}" \
    EXPERIMENT_LOG="${EXPERIMENT_LOG}" \
    SCOUT_CONDITIONS="${SCOUT_CONDITIONS}" \
    SCOUT_SEED="${SCOUT_SEED}" \
    PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" \
    FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS}" \
    python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
log_path = Path(os.environ["EXPERIMENT_LOG"])
conditions = [c.strip() for c in os.environ["SCOUT_CONDITIONS"].split(",") if c.strip()]
seed = os.environ["SCOUT_SEED"]
pretrain_epochs = os.environ["PRETRAIN_EPOCHS"]
fcos_epochs = os.environ["FCOS_NUM_EPOCHS"]
marker = "## Experiment 14: Paper-Budget Scout"

if log_path.exists() and marker in log_path.read_text():
    print(f"[launcher] result log already contains {marker}; skip append")
    raise SystemExit(0)

def pretrain_name(condition):
    if condition == "baseline":
        return f"nerfmae_all_p1.0_e{pretrain_epochs}_seed{seed}"
    if condition == "alpha_target_only":
        return f"nerfmae_alpha_target_only_p1.0_e{pretrain_epochs}_seed{seed}"
    raise ValueError(condition)

def eval_path(condition):
    pre = pretrain_name(condition)
    save = f"{pre}_epoch{pretrain_epochs}_sched_epoch_seed{seed}_fcos{fcos_epochs}_eval"
    return root / "output" / "nerf_rpn" / "results" / save / "eval.json"

def metric(path, key):
    data = json.loads(path.read_text())
    value = data[key]
    if isinstance(value, dict):
        if "ap" in value:
            return float(value["ap"])
        if "recalls" in value:
            recalls = value["recalls"]
            return float(recalls[0] if recalls else 0.0)
        if "ar" in value:
            return float(value["ar"])
    return float(value)

rows = []
missing = []
for condition in conditions:
    path = eval_path(condition)
    if not path.exists():
        missing.append(str(path))
        continue
    rows.append({
        "condition": condition,
        "ap50": metric(path, "ap_50"),
        "ap25": metric(path, "ap_25"),
        "ap75": metric(path, "ap_75"),
        "recall": metric(path, "recall_50_top_300"),
        "path": path,
    })

if missing:
    raise FileNotFoundError("missing eval files:\n" + "\n".join(missing))

section = []
section.append("")
section.append(marker)
section.append("")
section.append("Date:")
section.append("- auto-appended by `run_paper_budget_scout_chain.sh`")
section.append("")
section.append("Goal:")
section.append("- test whether `alpha_target_only` remains competitive with the vanilla RGBA baseline near the paper budget")
section.append("")
section.append("Protocol:")
section.append(f"- pretrain: `percent_train=1.0`, `epochs={pretrain_epochs}`, seed `{seed}`, checkpoint `epoch_{pretrain_epochs}.pt`")
section.append("- pretrain optimizer setting: `LR=1e-3`, `WEIGHT_DECAY=0.0`, global batch 16 on 4 GPUs")
section.append(f"- downstream: Front3D FCOS, `FCOS_NUM_EPOCHS={fcos_epochs}`, `LR_SCHEDULER=onecycle_epoch`, AP50-best checkpoint selection")
section.append("")
section.append("| condition | AP@50 | AP@25 | AP@75 | Recall@50 top300 |")
section.append("|---|---:|---:|---:|---:|")
for r in rows:
    section.append(f"| {r['condition']} | {r['ap50']:.4f} | {r['ap25']:.4f} | {r['ap75']:.4f} | {r['recall']:.4f} |")
section.append("")
section.append("Eval files:")
for r in rows:
    section.append(f"- `{r['path']}`")
section.append("")
section.append("Reading:")
section.append("- This is the 1-seed paper-budget scout. Promote to multi-seed only if `alpha_target_only` is competitive with or better than baseline.")
section.append("")

with log_path.open("a") as f:
    f.write("\n".join(section))
print(f"[launcher] appended results to {log_path}")
PY
}

log "start paper_budget_scout seed=${SCOUT_SEED} conditions=${SCOUT_CONDITIONS}"
run_pretrains
run_fcos_jobs
append_result_log
log "chain done"
