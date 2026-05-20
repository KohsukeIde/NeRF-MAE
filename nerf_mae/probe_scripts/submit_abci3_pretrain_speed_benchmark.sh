#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ABCI_GROUP="${ABCI_GROUP:-gag51404}"
PRETRAIN_QUEUE="${PRETRAIN_QUEUE:-rt_HF}"
PRETRAIN_WALLTIME="${PRETRAIN_WALLTIME:-02:00:00}"
PROBE_ENV_PREFIX="${PROBE_ENV_PREFIX:-${ROOT_DIR}/.venv-abci3}"
PRETRAIN_DATA_ROOT="${PRETRAIN_DATA_ROOT:-${ROOT_DIR}/dataset/pretrain}"
PRETRAIN_DATA_SRC="${PRETRAIN_DATA_SRC:-${PRETRAIN_DATA_ROOT}}"
ABCI3_CUDA_MODULE="${ABCI3_CUDA_MODULE:-cuda/11.8/11.8.0}"

BENCH_RUN_ID="${BENCH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BENCH_EPOCHS="${BENCH_EPOCHS:-1}"
BENCH_SEED="${BENCH_SEED:-9101}"
BENCH_GLOBAL_BATCHES="${BENCH_GLOBAL_BATCHES:-16 64}"
BENCH_DETERMINISTIC_VALUES="${BENCH_DETERMINISTIC_VALUES:-1 0}"
BENCH_STAGE_PRETRAIN_VALUES="${BENCH_STAGE_PRETRAIN_VALUES:-0 1}"
BENCH_TOPOLOGIES="${BENCH_TOPOLOGIES:-1n4g:1:0-3 1n8g:1:0-7 2n16g:2:0-7}"
BENCH_CONDITION="${BENCH_CONDITION:-baseline}"
BENCH_SLOTS="${BENCH_SLOTS:-1}"
BENCH_DEPENDENCY_TYPE="${BENCH_DEPENDENCY_TYPE:-afterany}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-3}"
PRETRAIN_WEIGHT_DECAY="${PRETRAIN_WEIGHT_DECAY:-0.0}"
PRETRAIN_LOG_INTERVAL="${PRETRAIN_LOG_INTERVAL:-1}"
PRETRAIN_EVAL_INTERVAL="${PRETRAIN_EVAL_INTERVAL:-${BENCH_EPOCHS}}"
PRETRAIN_CHECKPOINT_INTERVAL="${PRETRAIN_CHECKPOINT_INTERVAL:-${BENCH_EPOCHS}}"
PRETRAIN_PROFILE_STEP_TIME="${PRETRAIN_PROFILE_STEP_TIME:-1}"
PRETRAIN_TRAIN_NUM_WORKERS="${PRETRAIN_TRAIN_NUM_WORKERS:-}"
PRETRAIN_EVAL_NUM_WORKERS="${PRETRAIN_EVAL_NUM_WORKERS:-}"
PRETRAIN_PERSISTENT_WORKERS="${PRETRAIN_PERSISTENT_WORKERS:-0}"
STAGE_PRETRAIN_DATA="${STAGE_PRETRAIN_DATA:-0}"
LOCAL_STAGE_ROOT="${LOCAL_STAGE_ROOT:-}"
STAGE_KEEP="${STAGE_KEEP:-0}"
STAGE_MIN_FREE_GB="${STAGE_MIN_FREE_GB:-80}"
PRETRAIN_MASTER_PORT_BASE="${PRETRAIN_MASTER_PORT_BASE:-29540}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
DRY_RUN="${DRY_RUN:-1}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
SUBMIT_LOG_DIR="${SUBMIT_LOG_DIR:-${ROOT_DIR}/output/launcher/abci3_pretrain_speed_bench/${BENCH_RUN_ID}}"

count_gpu_ids() {
  local spec="$1"
  local count=0 token start end
  IFS=',' read -r -a tokens <<< "${spec//:/,}"
  for token in "${tokens[@]}"; do
    token="${token//[[:space:]]/}"
    [[ -n "${token}" ]] || continue
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

condition_kind() {
  case "$1" in
    baseline) printf "baseline" ;;
    cosine_ramp|cosine_ramp_alpha_shuffle) printf "curriculum" ;;
    *) printf "diagnostic" ;;
  esac
}

pretrain_save_name() {
  local condition="$1"
  local epochs="$2"
  local seed="$3"
  local suffix="$4"
  local suffix_part=""
  [[ -z "${suffix}" ]] || suffix_part="_${suffix}"
  case "${condition}" in
    baseline)
      printf "nerfmae_all_p1.0_e%s_seed%s%s\n" "${epochs}" "${seed}" "${suffix_part}"
      ;;
    cosine_ramp|cosine_ramp_alpha_shuffle)
      printf "nerfmae_alpha_rgba_curr_%s_p1.0_e%s_seed%s%s\n" "${condition}" "${epochs}" "${seed}" "${suffix_part}"
      ;;
    *)
      printf "nerfmae_%s_p1.0_e%s_seed%s%s\n" "${condition}" "${epochs}" "${seed}" "${suffix_part}"
      ;;
  esac
}

mkdir -p "${SUBMIT_LOG_DIR}"

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  REQUIRE_PRETRAIN_DATA=1 \
  REQUIRE_FCOS_DATA=0 \
  PROBE_ENV_PREFIX="${PROBE_ENV_PREFIX}" \
  PRETRAIN_DATA_ROOT="${PRETRAIN_DATA_ROOT}" \
  bash "${SCRIPT_DIR}/abci3_e300_gate_preflight.sh"
fi
if (( BENCH_SLOTS < 1 )); then
  echo "[error] BENCH_SLOTS must be >= 1" >&2
  exit 1
fi

manifest="${SUBMIT_LOG_DIR}/manifest.tsv"
printf "run_id\ttopology\tnodes\tlocal_gpus\tworld_size\tglobal_batch\tbatch_size_per_gpu\tdeterministic\tstage_pretrain_data\tepochs\tseed\trun_suffix\tsave_name\tworker_log\tpbs_log\tjob_id\tstatus\tdependency\n" > "${manifest}"

idx=0
declare -a slot_tail
for slot in $(seq 0 $((BENCH_SLOTS - 1))); do
  slot_tail[slot]=""
done

for topo in ${BENCH_TOPOLOGIES}; do
  IFS=':' read -r topo_name nodes gpu_ids <<< "${topo}"
  local_gpus="$(count_gpu_ids "${gpu_ids}")"
  world_size=$((nodes * local_gpus))
  for global_batch in ${BENCH_GLOBAL_BATCHES}; do
    if (( global_batch % world_size != 0 )); then
      echo "[error] global batch ${global_batch} is not divisible by world size ${world_size} for ${topo_name}" >&2
      exit 1
    fi
    batch_per_gpu=$((global_batch / world_size))
    if (( batch_per_gpu < 1 )); then
      echo "[error] batch size per GPU would be < 1 for ${topo_name} gb${global_batch}" >&2
      exit 1
    fi
    for deterministic in ${BENCH_DETERMINISTIC_VALUES}; do
      for stage_data in ${BENCH_STAGE_PRETRAIN_VALUES}; do
        idx=$((idx + 1))
        slot=$(((idx - 1) % BENCH_SLOTS))
        dependency="${slot_tail[slot]}"
        run_suffix="abci3speed_${BENCH_RUN_ID}_${topo_name}_gb${global_batch}_det${deterministic}_stage${stage_data}"
        save_name="$(pretrain_save_name "${BENCH_CONDITION}" "${BENCH_EPOCHS}" "${BENCH_SEED}" "${run_suffix}")"
        worker_log="${ROOT_DIR}/output/nerf_mae/results/${save_name}/log/worker_0.log"
        pbs_log="${SUBMIT_LOG_DIR}/${topo_name}_gb${global_batch}_det${deterministic}_stage${stage_data}.pbs.log"
        job_name="sp${topo_name//[!0-9a-zA-Z]/}b${global_batch}d${deterministic}s${stage_data}"
        job_name="${job_name:0:15}"
        master_port=$((PRETRAIN_MASTER_PORT_BASE + idx))
        qsub_gpu_ids="${gpu_ids//,/:}"
        varlist="ROOT_DIR=${ROOT_DIR},KIND=$(condition_kind "${BENCH_CONDITION}"),CONDITION=${BENCH_CONDITION},EPOCHS=${BENCH_EPOCHS},SEED=${BENCH_SEED},RUN_SUFFIX=${run_suffix},PROBE_ENV_PREFIX=${PROBE_ENV_PREFIX},PRETRAIN_DATA_ROOT=${PRETRAIN_DATA_ROOT},PRETRAIN_DATA_SRC=${PRETRAIN_DATA_SRC},ABCI3_CUDA_MODULE=${ABCI3_CUDA_MODULE},PRETRAIN_NODES=${nodes},PRETRAIN_GPU_IDS=${qsub_gpu_ids},PRETRAIN_BATCH_SIZE=${global_batch},PRETRAIN_BATCH_SIZE_PER_GPU=${batch_per_gpu},PRETRAIN_LR=${PRETRAIN_LR},PRETRAIN_WEIGHT_DECAY=${PRETRAIN_WEIGHT_DECAY},PRETRAIN_EVAL_INTERVAL=${PRETRAIN_EVAL_INTERVAL},PRETRAIN_CHECKPOINT_INTERVAL=${PRETRAIN_CHECKPOINT_INTERVAL},PRETRAIN_LOG_INTERVAL=${PRETRAIN_LOG_INTERVAL},PRETRAIN_PROFILE_STEP_TIME=${PRETRAIN_PROFILE_STEP_TIME},PRETRAIN_TRAIN_NUM_WORKERS=${PRETRAIN_TRAIN_NUM_WORKERS},PRETRAIN_EVAL_NUM_WORKERS=${PRETRAIN_EVAL_NUM_WORKERS},PRETRAIN_PERSISTENT_WORKERS=${PRETRAIN_PERSISTENT_WORKERS},STAGE_PRETRAIN_DATA=${stage_data},LOCAL_STAGE_ROOT=${LOCAL_STAGE_ROOT},STAGE_KEEP=${STAGE_KEEP},STAGE_MIN_FREE_GB=${STAGE_MIN_FREE_GB},USE_WANDB=${USE_WANDB},WANDB_MODE=${WANDB_MODE},DETERMINISTIC=${deterministic},AUTO_RESUME_PRETRAIN=0,SKIP_EXISTING=0,PRETRAIN_MASTER_PORT=${master_port}"
        cmd=(
          qsub
          -P "${ABCI_GROUP}"
          -q "${PRETRAIN_QUEUE}"
          -l "select=${nodes}"
          -l "walltime=${PRETRAIN_WALLTIME}"
          -N "${job_name}"
          -j oe
          -o "${pbs_log}"
          -v "${varlist}"
        )
        if [[ -n "${dependency}" ]]; then
          cmd+=(-W "depend=${BENCH_DEPENDENCY_TYPE}:${dependency}")
        fi
        cmd+=("${SCRIPT_DIR}/abci3_e300_gate_pretrain.pbs")

        job_id="DRYRUN_${idx}"
        status="dry_run"
        if [[ "${DRY_RUN}" == "1" ]]; then
          printf '[dry-run]'; printf ' %q' "${cmd[@]}"; printf '\n'
        else
          job_id="$("${cmd[@]}")"
          job_id="${job_id%% *}"
          status="submitted"
          echo "[submitted] slot=${slot} topology=${topo_name} gb=${global_batch} det=${deterministic} stage=${stage_data} job=${job_id} dep=${dependency:-none}"
        fi
        slot_tail[slot]="${job_id}"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "${BENCH_RUN_ID}" "${topo_name}" "${nodes}" "${local_gpus}" "${world_size}" \
          "${global_batch}" "${batch_per_gpu}" "${deterministic}" "${stage_data}" \
          "${BENCH_EPOCHS}" "${BENCH_SEED}" "${run_suffix}" "${save_name}" "${worker_log}" \
          "${pbs_log}" "${job_id}" "${status}" "${dependency:-}" >> "${manifest}"
      done
    done
  done
done

echo "[info] manifest: ${manifest}"
echo "[info] parse after completion with:"
echo "  python nerf_mae/tools/parse_pretrain_speed_log.py --manifest ${manifest} --warmup-steps 10 --md-out ${SUBMIT_LOG_DIR}/summary.md --csv-out ${SUBMIT_LOG_DIR}/summary.csv --json-out ${SUBMIT_LOG_DIR}/summary.json"
