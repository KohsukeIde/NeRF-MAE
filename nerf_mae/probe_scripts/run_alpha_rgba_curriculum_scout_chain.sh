#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_probe_common.sh"

CHAIN_NAME="${CHAIN_NAME:-nerfmae_alpha_rgba_curriculum_scout}"
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
CURRICULUM_CONDITIONS="${CURRICULUM_CONDITIONS:-warmup10,warmup25,cosine_ramp}"
PRETRAIN_EPOCHS_LIST="${PRETRAIN_EPOCHS_LIST:-300}"
PRETRAIN_GPU_IDS="${PRETRAIN_GPU_IDS:-0,1,2,3}"
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
    "cd '${ROOT_DIR}' && FOREGROUND=1 CHAIN_LOG_ROOT='${CHAIN_LOG_ROOT}' CHAIN_LOG_FILE='${CHAIN_LOG_FILE}' CHAIN_PID_FILE='${CHAIN_PID_FILE}' WAIT_BEFORE_CHAIN='${WAIT_BEFORE_CHAIN}' WAIT_POLL_SECONDS='${WAIT_POLL_SECONDS}' WAIT_STABLE_SECONDS='${WAIT_STABLE_SECONDS}' WAIT_TIMEOUT_SECONDS='${WAIT_TIMEOUT_SECONDS}' WAIT_MEMORY_USED_MAX_MIB='${WAIT_MEMORY_USED_MAX_MIB}' WAIT_UTIL_MAX='${WAIT_UTIL_MAX}' SKIP_EXISTING='${SKIP_EXISTING}' USE_WANDB='${USE_WANDB}' WANDB_MODE='${WANDB_MODE}' DETERMINISTIC='${DETERMINISTIC}' SCOUT_SEED='${SCOUT_SEED}' CURRICULUM_CONDITIONS='${CURRICULUM_CONDITIONS}' PRETRAIN_EPOCHS_LIST='${PRETRAIN_EPOCHS_LIST}' PRETRAIN_GPU_IDS='${PRETRAIN_GPU_IDS}' PRETRAIN_PERCENT_TRAIN='${PRETRAIN_PERCENT_TRAIN}' PRETRAIN_BATCH_SIZE_PER_GPU='${PRETRAIN_BATCH_SIZE_PER_GPU}' PRETRAIN_LR='${PRETRAIN_LR}' PRETRAIN_WEIGHT_DECAY='${PRETRAIN_WEIGHT_DECAY}' PRETRAIN_LOG_INTERVAL='${PRETRAIN_LOG_INTERVAL}' PRETRAIN_EVAL_INTERVAL='${PRETRAIN_EVAL_INTERVAL}' FCOS_NUM_EPOCHS='${FCOS_NUM_EPOCHS}' FCOS_ALLOWED_GPU_IDS='${FCOS_ALLOWED_GPU_IDS}' FCOS_MAX_PARALLEL='${FCOS_MAX_PARALLEL}' FCOS_BATCH_SIZE_PER_GPU='${FCOS_BATCH_SIZE_PER_GPU}' FCOS_LR='${FCOS_LR}' FCOS_WEIGHT_DECAY='${FCOS_WEIGHT_DECAY}' FCOS_LR_SCHEDULER='${FCOS_LR_SCHEDULER}' BLOCK_ON_TMUX_SESSIONS='${BLOCK_ON_TMUX_SESSIONS:-}' BLOCK_ON_PID_FILES='${BLOCK_ON_PID_FILES:-}' EXPERIMENT_LOG='${EXPERIMENT_LOG}' bash '${SCRIPT_DIR}/run_alpha_rgba_curriculum_scout_chain.sh'"
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

parse_space_to_array() {
  local words="$1"
  local -n out_ref=$2
  read -r -a out_ref <<< "${words}"
}

condition_save_name() {
  local condition="$1"
  local epochs="$2"
  case "${condition}" in
    warmup10|warmup25|linear_ramp|cosine_ramp)
      printf "nerfmae_alpha_rgba_curr_%s_p1.0_e%s_seed%s" "${condition}" "${epochs}" "${SCOUT_SEED}"
      ;;
    *)
      probe_die "unknown curriculum condition=${condition}"
      ;;
  esac
}

condition_curriculum_mode() {
  local condition="$1"
  case "${condition}" in
    warmup10|warmup25)
      printf "alpha_warmup"
      ;;
    linear_ramp)
      printf "linear_rgb_ramp"
      ;;
    cosine_ramp)
      printf "cosine_rgb_ramp"
      ;;
    *)
      probe_die "unknown curriculum condition=${condition}"
      ;;
  esac
}

condition_curriculum_epochs() {
  local condition="$1"
  local epochs="$2"
  case "${condition}" in
    warmup10)
      printf "%s" "$(( (epochs + 9) / 10 ))"
      ;;
    warmup25)
      printf "%s" "$(( (epochs + 3) / 4 ))"
      ;;
    linear_ramp|cosine_ramp)
      printf "%s" "${epochs}"
      ;;
    *)
      probe_die "unknown curriculum condition=${condition}"
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

run_pretrain() {
  local condition="$1"
  local epochs="$2"
  local save_name="$3"
  local curriculum_mode="$4"
  local curriculum_epochs="$5"
  local log_file="$6"

  (
    cd "${ROOT_DIR}/nerf_mae"
    PATH="${PROBE_ENV_PREFIX}/bin:${PATH}" \
      PYTHONPATH="${ROOT_DIR}" \
      SAVE_NAME="${save_name}" \
      RUN_TAG="${save_name}" \
      GPU_IDS="${PRETRAIN_GPU_IDS}" \
      NUM_EPOCHS="${epochs}" \
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
      PROBE_MODE=custom \
      PROBE_RGB_INPUT=keep \
      PROBE_ALPHA_INPUT=keep \
      PROBE_ALPHA_TARGET=keep \
      PROBE_RGB_LOSS=occupied \
      PROBE_ALPHA_LOSS=removed \
      PROBE_RGB_WEIGHT=1.0 \
      PROBE_ALPHA_WEIGHT=1.0 \
      PROBE_CURRICULUM="${curriculum_mode}" \
      PROBE_CURRICULUM_EPOCHS="${curriculum_epochs}" \
      PROBE_CURRICULUM_RGB_START_WEIGHT=0.0 \
      PROBE_CURRICULUM_RGB_END_WEIGHT=1.0 \
      PROBE_CURRICULUM_ALPHA_WEIGHT=1.0 \
      bash train_mae3d.sh
  ) >> "${log_file}" 2>&1

  if ! pretrain_epoch_exists "${save_name}" "${epochs}"; then
    probe_die "pretrain finished but checkpoint is missing condition=${condition} save=${save_name} checkpoint=epoch_${epochs}.pt"
  fi
}

run_pretrains() {
  local conditions=()
  local epoch_list=()
  parse_csv_to_array "${CURRICULUM_CONDITIONS}" conditions
  parse_space_to_array "${PRETRAIN_EPOCHS_LIST}" epoch_list

  local epochs condition save_name curriculum_mode curriculum_epochs
  log "step_curriculum_pretrain start conditions=${CURRICULUM_CONDITIONS} epochs=${PRETRAIN_EPOCHS_LIST}"
  for epochs in "${epoch_list[@]}"; do
    for condition in "${conditions[@]}"; do
      save_name="$(condition_save_name "${condition}" "${epochs}")"
      if [[ "${SKIP_EXISTING}" == "1" ]] && pretrain_epoch_exists "${save_name}" "${epochs}"; then
        log "skip pretrain condition=${condition} epochs=${epochs} save=${save_name}"
        continue
      fi

      if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
        GPU_IDS="${PRETRAIN_GPU_IDS}" probe_wait_for_runway
      fi

      curriculum_mode="$(condition_curriculum_mode "${condition}")"
      curriculum_epochs="$(condition_curriculum_epochs "${condition}" "${epochs}")"
      log "pretrain start condition=${condition} save=${save_name} gpus=${PRETRAIN_GPU_IDS} epochs=${epochs} curriculum=${curriculum_mode} curriculum_epochs=${curriculum_epochs}"
      run_pretrain \
        "${condition}" \
        "${epochs}" \
        "${save_name}" \
        "${curriculum_mode}" \
        "${curriculum_epochs}" \
        "${CHAIN_LOG_ROOT}/${CHAIN_NAME}.${save_name}.pretrain.log"
      log "pretrain done condition=${condition} epochs=${epochs} save=${save_name}"
    done
  done
  log "step_curriculum_pretrain done"
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
  local epoch_list=()
  parse_csv_to_array "${CURRICULUM_CONDITIONS}" conditions
  parse_space_to_array "${PRETRAIN_EPOCHS_LIST}" epoch_list
  local pending_jobs=()
  local remaining_jobs=()
  local epochs condition pretrain_save checkpoint save_name job

  for epochs in "${epoch_list[@]}"; do
    for condition in "${conditions[@]}"; do
      pretrain_save="$(condition_save_name "${condition}" "${epochs}")"
      checkpoint="../output/nerf_mae/results/${pretrain_save}/epoch_${epochs}.pt"
      save_name="${pretrain_save}_epoch${epochs}_sched_epoch_seed${SCOUT_SEED}_fcos${FCOS_NUM_EPOCHS}"
      if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${save_name}"; then
        log "skip fcos condition=${condition} epochs=${epochs} save=${save_name}"
        continue
      fi
      pending_jobs+=("${condition}:${epochs}:${pretrain_save}:${checkpoint}:${save_name}")
    done
  done

  if (( ${#pending_jobs[@]} == 0 )); then
    log "skip fcos; all evals exist"
    return 0
  fi

  remaining_jobs=("${pending_jobs[@]}")
  log "step_curriculum_fcos start jobs=${#remaining_jobs[@]}"
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
        IFS=':' read -r condition epochs pretrain_save checkpoint save_name <<< "${job}"
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
  log "step_curriculum_fcos done"
}

append_result_log() {
  ROOT_DIR="${ROOT_DIR}" \
    EXPERIMENT_LOG="${EXPERIMENT_LOG}" \
    CURRICULUM_CONDITIONS="${CURRICULUM_CONDITIONS}" \
    PRETRAIN_EPOCHS_LIST="${PRETRAIN_EPOCHS_LIST}" \
    SCOUT_SEED="${SCOUT_SEED}" \
    FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS}" \
    python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
log_path = Path(os.environ["EXPERIMENT_LOG"])
conditions = [c.strip() for c in os.environ["CURRICULUM_CONDITIONS"].split(",") if c.strip()]
epoch_list = [e.strip() for e in os.environ["PRETRAIN_EPOCHS_LIST"].split() if e.strip()]
seed = os.environ["SCOUT_SEED"]
fcos_epochs = os.environ["FCOS_NUM_EPOCHS"]
marker = f"## Experiment 15: Alpha-to-RGBA Curriculum Scout ({','.join(epoch_list)} epochs, seed {seed})"

if log_path.exists() and marker in log_path.read_text():
    print(f"[launcher] result log already contains {marker}; skip append")
    raise SystemExit(0)

def pretrain_name(condition, epochs):
    return f"nerfmae_alpha_rgba_curr_{condition}_p1.0_e{epochs}_seed{seed}"

def eval_path(condition, epochs):
    pre = pretrain_name(condition, epochs)
    save = f"{pre}_epoch{epochs}_sched_epoch_seed{seed}_fcos{fcos_epochs}_eval"
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
for epochs in epoch_list:
    for condition in conditions:
        path = eval_path(condition, epochs)
        if not path.exists():
            missing.append(str(path))
            continue
        rows.append({
            "epochs": epochs,
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
section.append("Goal:")
section.append("- test alpha-target warmup / RGB-ramp curricula after paper-budget scout showed full RGBA wins asymptotically over zero-input alpha-target-only")
section.append("")
section.append("Protocol:")
section.append(f"- pretrain: `percent_train=1.0`, seed `{seed}`, checkpoint `epoch_N.pt`, visible input kept as RGBA")
section.append("- loss: alpha loss on removed patches is always active; RGB loss on occupied voxels starts at weight 0 and returns or ramps to 1")
section.append("- pretrain optimizer setting: `LR=1e-3`, `WEIGHT_DECAY=0.0`, global batch 16 on 4 GPUs")
section.append(f"- downstream: Front3D FCOS, `FCOS_NUM_EPOCHS={fcos_epochs}`, `LR_SCHEDULER=onecycle_epoch`, AP50-best checkpoint selection")
section.append("")
section.append("| pretrain epochs | condition | AP@50 | AP@25 | AP@75 | Recall@50 top300 |")
section.append("|---:|---|---:|---:|---:|---:|")
for r in rows:
    section.append(f"| {r['epochs']} | {r['condition']} | {r['ap50']:.4f} | {r['ap25']:.4f} | {r['ap75']:.4f} | {r['recall']:.4f} |")
section.append("")
section.append("Eval files:")
for r in rows:
    section.append(f"- `{r['path']}`")
section.append("")
section.append("Reading:")
section.append("- This is a staged scout. Promote only the best curriculum to e600/e1200 or multi-seed if it improves sample efficiency against the existing paper-budget baseline reference.")
section.append("")

with log_path.open("a") as f:
    f.write("\n".join(section))
print(f"[launcher] appended results to {log_path}")
PY
}

log "start alpha_rgba_curriculum_scout seed=${SCOUT_SEED} conditions=${CURRICULUM_CONDITIONS} epochs=${PRETRAIN_EPOCHS_LIST}"
run_pretrains
run_fcos_jobs
append_result_log
log "chain done"
