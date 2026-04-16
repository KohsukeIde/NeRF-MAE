#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/_probe_common.sh"

CHAIN_NAME="${CHAIN_NAME:-nerfmae_tiny_rgb_w0p02_e100_gate}"
CHAIN_LOG_ROOT="${CHAIN_LOG_ROOT:-${PROBE_LOG_ROOT_DEFAULT}}"
CHAIN_LOG_FILE="${CHAIN_LOG_FILE:-${CHAIN_LOG_ROOT}/${CHAIN_NAME}.chain.log}"
CHAIN_PID_FILE="${CHAIN_PID_FILE:-${CHAIN_LOG_ROOT}/${CHAIN_NAME}.chain.pid}"
TMUX_SESSION="${TMUX_SESSION:-${CHAIN_NAME//[^[:alnum:]_]/_}_chain}"

WAIT_BEFORE_CHAIN="${WAIT_BEFORE_CHAIN:-1}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-30}"
WAIT_STABLE_SECONDS="${WAIT_STABLE_SECONDS:-240}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-0}"
WAIT_MEMORY_USED_MAX_MIB="${WAIT_MEMORY_USED_MAX_MIB:-512}"
WAIT_UTIL_MAX="${WAIT_UTIL_MAX:-10}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
DETERMINISTIC="${DETERMINISTIC:-1}"

FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS:-100}"
FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU:-2}"
FCOS_LR="${FCOS_LR:-1e-4}"
FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY:-1e-3}"
FCOS_LR_SCHEDULER="${FCOS_LR_SCHEDULER:-onecycle_epoch}"
FCOS_ALLOWED_GPU_IDS="${FCOS_ALLOWED_GPU_IDS:-0,1,2,3}"
FCOS_MAX_PARALLEL="${FCOS_MAX_PARALLEL:-4}"

TINY_RGB_WEIGHT="${TINY_RGB_WEIGHT:-0.02}"
TINY_RGB_TAG="${TINY_RGB_TAG:-w0p02}"
TINY_RGB_SEEDS="${TINY_RGB_SEEDS:-1,2,3}"
TINY_RGB_PRETRAIN_GPU_IDS="${TINY_RGB_PRETRAIN_GPU_IDS:-0,1,2,3}"
TINY_RGB_PRETRAIN_EPOCHS="${TINY_RGB_PRETRAIN_EPOCHS:-100}"
TINY_RGB_PRETRAIN_PERCENT_TRAIN="${TINY_RGB_PRETRAIN_PERCENT_TRAIN:-0.1}"
TINY_RGB_PRETRAIN_BATCH_SIZE_PER_GPU="${TINY_RGB_PRETRAIN_BATCH_SIZE_PER_GPU:-4}"

EXPERIMENT_LOG="${EXPERIMENT_LOG:-${ROOT_DIR}/SHORTCUT_PROBE_EXPERIMENT_LOG.md}"

mkdir -p "${CHAIN_LOG_ROOT}"

if [[ "${FOREGROUND:-0}" != "1" ]]; then
  if env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
    echo "[info] chain already running in tmux session=${TMUX_SESSION}"
    echo "[info] log=${CHAIN_LOG_FILE}"
    exit 0
  fi
  env -u LD_LIBRARY_PATH -u PYTHONPATH tmux new-session -d -s "${TMUX_SESSION}" \
    "cd '${ROOT_DIR}' && FOREGROUND=1 CHAIN_LOG_ROOT='${CHAIN_LOG_ROOT}' CHAIN_LOG_FILE='${CHAIN_LOG_FILE}' CHAIN_PID_FILE='${CHAIN_PID_FILE}' WAIT_BEFORE_CHAIN='${WAIT_BEFORE_CHAIN}' WAIT_POLL_SECONDS='${WAIT_POLL_SECONDS}' WAIT_STABLE_SECONDS='${WAIT_STABLE_SECONDS}' WAIT_TIMEOUT_SECONDS='${WAIT_TIMEOUT_SECONDS}' WAIT_MEMORY_USED_MAX_MIB='${WAIT_MEMORY_USED_MAX_MIB}' WAIT_UTIL_MAX='${WAIT_UTIL_MAX}' SKIP_EXISTING='${SKIP_EXISTING}' USE_WANDB='${USE_WANDB}' WANDB_MODE='${WANDB_MODE}' DETERMINISTIC='${DETERMINISTIC}' FCOS_NUM_EPOCHS='${FCOS_NUM_EPOCHS}' FCOS_BATCH_SIZE_PER_GPU='${FCOS_BATCH_SIZE_PER_GPU}' FCOS_LR='${FCOS_LR}' FCOS_WEIGHT_DECAY='${FCOS_WEIGHT_DECAY}' FCOS_LR_SCHEDULER='${FCOS_LR_SCHEDULER}' FCOS_ALLOWED_GPU_IDS='${FCOS_ALLOWED_GPU_IDS}' FCOS_MAX_PARALLEL='${FCOS_MAX_PARALLEL}' TINY_RGB_WEIGHT='${TINY_RGB_WEIGHT}' TINY_RGB_TAG='${TINY_RGB_TAG}' TINY_RGB_SEEDS='${TINY_RGB_SEEDS}' TINY_RGB_PRETRAIN_GPU_IDS='${TINY_RGB_PRETRAIN_GPU_IDS}' TINY_RGB_PRETRAIN_EPOCHS='${TINY_RGB_PRETRAIN_EPOCHS}' TINY_RGB_PRETRAIN_PERCENT_TRAIN='${TINY_RGB_PRETRAIN_PERCENT_TRAIN}' TINY_RGB_PRETRAIN_BATCH_SIZE_PER_GPU='${TINY_RGB_PRETRAIN_BATCH_SIZE_PER_GPU}' BLOCK_ON_TMUX_SESSIONS='${BLOCK_ON_TMUX_SESSIONS:-}' BLOCK_ON_PID_FILES='${BLOCK_ON_PID_FILES:-}' EXPERIMENT_LOG='${EXPERIMENT_LOG}' bash '${SCRIPT_DIR}/run_tiny_rgb_w0p02_e100_gate_chain.sh'"
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

tiny_rgb_save_name() {
  local seed="$1"
  printf "nerfmae_alpha_target_tiny_rgb_%s_p0.1_e100_seed%s" "${TINY_RGB_TAG}" "${seed}"
}

eval_exists() {
  local save_name="$1"
  [[ -f "${ROOT_DIR}/output/nerf_rpn/results/${save_name}_eval/eval.json" ]]
}

pretrain_epoch_exists() {
  local save_name="$1"
  local epoch="$2"
  [[ -f "${ROOT_DIR}/output/nerf_mae/results/${save_name}/epoch_${epoch}.pt" ]]
}

run_pretrain_script() {
  local save_name="$1"
  local gpu_ids="$2"
  local num_epochs="$3"
  local percent_train="$4"
  local batch_size_per_gpu="$5"
  local seed="$6"
  local log_file="$7"

  (
    cd "${ROOT_DIR}/nerf_mae"
    TINY_RGB_WEIGHT="${TINY_RGB_WEIGHT}" \
      PATH="${PROBE_ENV_PREFIX}/bin:${PATH}" \
      PYTHONPATH="${ROOT_DIR}" \
      SAVE_NAME="${save_name}" \
      RUN_TAG="${save_name}" \
      GPU_IDS="${gpu_ids}" \
      NUM_EPOCHS="${num_epochs}" \
      PERCENT_TRAIN="${percent_train}" \
      BATCH_SIZE_PER_GPU="${batch_size_per_gpu}" \
      USE_WANDB="${USE_WANDB}" \
      WANDB_MODE="${WANDB_MODE}" \
      SEED="${seed}" \
      DETERMINISTIC="${DETERMINISTIC}" \
      bash "probe_scripts/train_alpha_target_tiny_rgb.sh"
  ) >> "${log_file}" 2>&1
}

run_job_probe() {
  local variant="$1"
  local gpu_ids="$2"
  local pretrain_save_name="$3"
  local pretrain_checkpoint="$4"
  local save_name="$5"
  local seed="$6"
  local percent_train="$7"
  local log_file="$8"
  (
    cd "${ROOT_DIR}"
    PRETRAIN_SAVE_NAME="${pretrain_save_name}" \
      VARIANT_NAME="${variant}" \
      PRETRAIN_CHECKPOINT="${pretrain_checkpoint}" \
      GPU_IDS="${gpu_ids}" \
      DATASET_NAME=front3d \
      SPLIT_NAME=3dfront \
      PERCENT_TRAIN="${percent_train}" \
      FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS}" \
      FCOS_LR="${FCOS_LR}" \
      FCOS_WEIGHT_DECAY="${FCOS_WEIGHT_DECAY}" \
      FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU}" \
      USE_WANDB="${USE_WANDB}" \
      WANDB_MODE="${WANDB_MODE}" \
      SAVE_NAME="${save_name}" \
      RUN_TAG="${save_name}" \
      SEED="${seed}" \
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

run_dynamic_probe_jobs() {
  local percent_train="$1"
  local -n jobs_ref=$2
  local pending_jobs=()
  local remaining_jobs=()
  local job variant pretrain_save checkpoint save_name seed

  for job in "${jobs_ref[@]}"; do
    IFS=':' read -r variant pretrain_save checkpoint save_name seed <<< "${job}"
    if [[ "${SKIP_EXISTING}" == "1" ]] && eval_exists "${save_name}"; then
      log "skip fcos variant=${variant} save=${save_name}"
      continue
    fi
    pending_jobs+=("${job}")
  done

  if (( ${#pending_jobs[@]} == 0 )); then
    log "skip fcos wave percent_train=${percent_train}; all evals exist"
    return 0
  fi

  remaining_jobs=("${pending_jobs[@]}")
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
    log "fcos wave start percent_train=${percent_train} gpus=${gpu_csv} jobs=${#gpus[@]}/${#remaining_jobs[@]}"

    local pids=()
    local names=()
    local next_jobs=()
    local idx=0
    for job in "${remaining_jobs[@]}"; do
      if (( idx < ${#gpus[@]} )); then
        IFS=':' read -r variant pretrain_save checkpoint save_name seed <<< "${job}"
        run_job_probe \
          "${variant}" \
          "${gpus[$idx]}" \
          "${pretrain_save}" \
          "${checkpoint}" \
          "${save_name}" \
          "${seed}" \
          "${percent_train}" \
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
}

run_tiny_rgb_e100_pretrains() {
  local seeds=()
  parse_csv_to_array "${TINY_RGB_SEEDS}" seeds
  local seed save_name

  log "step_tiny_rgb_w0p02_e100_pretrain start"
  for seed in "${seeds[@]}"; do
    save_name="$(tiny_rgb_save_name "${seed}")"
    if [[ "${SKIP_EXISTING}" == "1" ]] && pretrain_epoch_exists "${save_name}" "${TINY_RGB_PRETRAIN_EPOCHS}"; then
      log "skip tiny_rgb e100 pretrain save=${save_name}"
      continue
    fi
    if [[ "${WAIT_BEFORE_CHAIN}" == "1" ]]; then
      GPU_IDS="${TINY_RGB_PRETRAIN_GPU_IDS}" probe_wait_for_runway
    fi
    run_pretrain_script \
      "${save_name}" \
      "${TINY_RGB_PRETRAIN_GPU_IDS}" \
      "${TINY_RGB_PRETRAIN_EPOCHS}" \
      "${TINY_RGB_PRETRAIN_PERCENT_TRAIN}" \
      "${TINY_RGB_PRETRAIN_BATCH_SIZE_PER_GPU}" \
      "${seed}" \
      "${CHAIN_LOG_ROOT}/${CHAIN_NAME}.${save_name}.pretrain.log"
    log "tiny_rgb e100 pretrain done save=${save_name}"
  done
  log "step_tiny_rgb_w0p02_e100_pretrain done"
}

run_tiny_rgb_e100_fcos() {
  local seeds=()
  parse_csv_to_array "${TINY_RGB_SEEDS}" seeds
  local full_jobs=()
  local low_jobs=()
  local seed pretrain_save checkpoint

  for seed in "${seeds[@]}"; do
    pretrain_save="$(tiny_rgb_save_name "${seed}")"
    checkpoint="../output/nerf_mae/results/${pretrain_save}/epoch_${TINY_RGB_PRETRAIN_EPOCHS}.pt"
    full_jobs+=("tiny_rgb_${TINY_RGB_TAG}:${pretrain_save}:${checkpoint}:${pretrain_save}_epoch100_sched_epoch_seed${seed}_fcos${FCOS_NUM_EPOCHS}:${seed}")
    low_jobs+=("tiny_rgb_${TINY_RGB_TAG}:${pretrain_save}:${checkpoint}:${pretrain_save}_epoch100_sched_epoch_pt02_seed${seed}_fcos${FCOS_NUM_EPOCHS}:${seed}")
  done

  log "step_tiny_rgb_w0p02_e100_full_fcos start"
  run_dynamic_probe_jobs "1.0" full_jobs
  log "step_tiny_rgb_w0p02_e100_full_fcos done"

  log "step_tiny_rgb_w0p02_e100_pt02_fcos start"
  run_dynamic_probe_jobs "0.2" low_jobs
  log "step_tiny_rgb_w0p02_e100_pt02_fcos done"
}

append_result_log() {
  TINY_RGB_SEEDS="${TINY_RGB_SEEDS}" \
    TINY_RGB_TAG="${TINY_RGB_TAG}" \
    FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS}" \
    TINY_RGB_PRETRAIN_EPOCHS="${TINY_RGB_PRETRAIN_EPOCHS}" \
    ROOT_DIR="${ROOT_DIR}" \
    EXPERIMENT_LOG="${EXPERIMENT_LOG}" \
    python - <<'PY'
import json
import os
import statistics
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
log_path = Path(os.environ["EXPERIMENT_LOG"])
seeds = [s.strip() for s in os.environ["TINY_RGB_SEEDS"].split(",") if s.strip()]
tag = os.environ["TINY_RGB_TAG"]
fcos_epochs = os.environ["FCOS_NUM_EPOCHS"]
pretrain_epochs = os.environ["TINY_RGB_PRETRAIN_EPOCHS"]
marker = "## Experiment 13: Tiny-RGB w0p02 e100 Gate"

if log_path.exists() and marker in log_path.read_text():
    print(f"[launcher] result log already contains {marker}; skip append")
    raise SystemExit(0)

def eval_path(seed, low_label=False):
    pre = f"nerfmae_alpha_target_tiny_rgb_{tag}_p0.1_e100_seed{seed}"
    suffix = f"epoch100_sched_epoch_pt02_seed{seed}_fcos{fcos_epochs}" if low_label else f"epoch100_sched_epoch_seed{seed}_fcos{fcos_epochs}"
    return root / "output" / "nerf_rpn" / "results" / f"{pre}_{suffix}_eval" / "eval.json"

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

def collect(low_label=False):
    rows = []
    missing = []
    for seed in seeds:
        path = eval_path(seed, low_label)
        if not path.exists():
            missing.append(str(path))
            continue
        rows.append({
            "seed": seed,
            "ap50": metric(path, "ap_50"),
            "ap25": metric(path, "ap_25"),
            "ap75": metric(path, "ap_75"),
            "recall": metric(path, "recall_50_top_300"),
            "path": path,
        })
    if missing:
        raise FileNotFoundError("missing eval files:\n" + "\n".join(missing))
    return rows

def mean(rows, key):
    return statistics.mean(r[key] for r in rows)

def stdev(rows, key):
    return statistics.stdev(r[key] for r in rows) if len(rows) > 1 else 0.0

def table(rows):
    lines = [
        "| seed | AP@50 | AP@25 | AP@75 | Recall@50 top300 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['ap50']:.4f} | {r['ap25']:.4f} | {r['ap75']:.4f} | {r['recall']:.4f} |")
    lines.append(f"| mean | {mean(rows, 'ap50'):.4f} | {mean(rows, 'ap25'):.4f} | {mean(rows, 'ap75'):.4f} | {mean(rows, 'recall'):.4f} |")
    lines.append(f"| std | {stdev(rows, 'ap50'):.4f} | {stdev(rows, 'ap25'):.4f} | {stdev(rows, 'ap75'):.4f} | {stdev(rows, 'recall'):.4f} |")
    return "\n".join(lines)

full = collect(False)
low = collect(True)
full_mean = mean(full, "ap50")
low_mean = mean(low, "ap50")

section = []
section.append("")
section.append(marker)
section.append("")
section.append("Date:")
section.append("- auto-appended by `run_tiny_rgb_w0p02_e100_gate_chain.sh`")
section.append("")
section.append("Goal:")
section.append("- promote the best tiny-RGB candidate from the e30 seed-1 sweep to the `p0.1`, `e100`, 3-seed gate")
section.append("")
section.append("Protocol:")
section.append("- pretrain: `probe_rgb_input=zero`, `probe_alpha_input=zero`, `probe_alpha_target=keep`, `probe_rgb_loss=removed_occupied`, `probe_alpha_loss=removed`")
section.append("- weights: `probe_rgb_weight=0.02`, `probe_alpha_weight=1.0`")
section.append("- pretrain budget: `percent_train=0.1`, `epochs=100`, checkpoint `epoch_100.pt`")
section.append("- downstream: Front3D FCOS, `FCOS_NUM_EPOCHS=100`, `LR_SCHEDULER=onecycle_epoch`, seeds `1,2,3`")
section.append("")
section.append("### Full-label Front3D FCOS")
section.append("")
section.append(table(full))
section.append("")
section.append("Eval files:")
for r in full:
    section.append(f"- `{r['path']}`")
section.append("")
section.append("### 20% label Front3D FCOS")
section.append("")
section.append(table(low))
section.append("")
section.append("Eval files:")
for r in low:
    section.append(f"- `{r['path']}`")
section.append("")
section.append("Reading:")
section.append(f"- Full-label mean AP@50 is `{full_mean:.4f}`; compare against the current e100 means: baseline `0.3711`, alpha_target_only `0.4368`.")
section.append(f"- 20% label mean AP@50 is `{low_mean:.4f}`; compare against the current seed-1 references: baseline e100 `0.1828`, alpha_target_only e100 `0.1910`, tiny-RGB e30 `0.2059`.")
section.append("- This is the method-candidate gate; if it is competitive with alpha_target_only and improves low-label stability, it should be promoted to paper-budget scout.")
section.append("")

with log_path.open("a") as f:
    f.write("\n".join(section))
print(f"[launcher] appended results to {log_path}")
PY
}

log "start tiny_rgb_w0p02_e100_gate"
run_tiny_rgb_e100_pretrains
run_tiny_rgb_e100_fcos
append_result_log
log "chain done"
