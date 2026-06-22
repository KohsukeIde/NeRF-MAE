#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ARTIFACT_DIR="${ROOT_DIR}/results/shortcut_probe_artifacts"
mkdir -p "${ARTIFACT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d)}"
CSV_PATH="${ARTIFACT_DIR}/anchor_head_lowlabel_jobs_${STAMP}.csv"
MD_PATH="${ARTIFACT_DIR}/anchor_head_lowlabel_jobs_${STAMP}.md"

PBS_SCRIPT="${ROOT_DIR}/nerf_rpn/tools/abci3_front3d_low_label_anchor.pbs"
FRONT3D_DATA_ROOT="${FRONT3D_DATA_ROOT:-${ROOT_DIR}/dataset/finetune/front3d_rpn_data}"
PERCENT_TRAIN="${PERCENT_TRAIN:-0.1}"
ANCHOR_NUM_EPOCHS="${ANCHOR_NUM_EPOCHS:-200}"
ANCHOR_BATCH_SIZE="${ANCHOR_BATCH_SIZE:-2}"
ANCHOR_LR="${ANCHOR_LR:-3e-4}"
ANCHOR_WEIGHT_DECAY="${ANCHOR_WEIGHT_DECAY:-1e-3}"
ANCHOR_EVAL_INTERVAL="${ANCHOR_EVAL_INTERVAL:-10}"
GPU_IDS="${GPU_IDS:-0}"
DETERMINISTIC="${DETERMINISTIC:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
USE_WANDB="${USE_WANDB:-0}"

JOINT_CKPT="${JOINT_CKPT:-${ROOT_DIR}/output/nerf_mae/results/nerfmae_all_p1.0_e300_seed1_abci3clean/epoch_300.pt}"
COSINE_CKPT="${COSINE_CKPT:-${ROOT_DIR}/output/nerf_mae/results/nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed1_abci3clean/epoch_300.pt}"

[[ -f "${PBS_SCRIPT}" ]] || { echo "[error] missing PBS script: ${PBS_SCRIPT}" >&2; exit 1; }
[[ -f "${JOINT_CKPT}" ]] || { echo "[error] missing joint checkpoint: ${JOINT_CKPT}" >&2; exit 1; }
[[ -f "${COSINE_CKPT}" ]] || { echo "[error] missing cosine checkpoint: ${COSINE_CKPT}" >&2; exit 1; }

if [[ ! -f "${CSV_PATH}" ]]; then
  printf "job_id,job_name,arm,seed,percent_train,save_name,checkpoint\n" > "${CSV_PATH}"
fi

submit_job() {
  local arm="$1"
  local seed="$2"
  local mode="$3"
  local variant_name="$4"
  local save_name="$5"
  local checkpoint="$6"
  local job_name="$7"
  local env_vars
  local job_id

  env_vars="MODE=${mode},VARIANT_NAME=${variant_name},SAVE_NAME=${save_name},PERCENT_TRAIN=${PERCENT_TRAIN},FINETUNE_SEED=${seed},FRONT3D_DATA_ROOT=${FRONT3D_DATA_ROOT},GPU_IDS=${GPU_IDS},ANCHOR_NUM_EPOCHS=${ANCHOR_NUM_EPOCHS},ANCHOR_BATCH_SIZE=${ANCHOR_BATCH_SIZE},ANCHOR_LR=${ANCHOR_LR},ANCHOR_WEIGHT_DECAY=${ANCHOR_WEIGHT_DECAY},ANCHOR_EVAL_INTERVAL=${ANCHOR_EVAL_INTERVAL},DETERMINISTIC=${DETERMINISTIC},SKIP_EXISTING=${SKIP_EXISTING},USE_WANDB=${USE_WANDB}"

  if [[ "${mode}" == "probe" ]]; then
    env_vars="${env_vars},MAE_CHECKPOINT=${checkpoint},PRETRAIN_SAVE_NAME=${variant_name}"
  fi

  job_id="$(qsub -N "${job_name}" -v "${env_vars}" "${PBS_SCRIPT}")"
  printf "%s,%s,%s,%s,%s,%s,%s\n" \
    "${job_id}" "${job_name}" "${arm}" "${seed}" "${PERCENT_TRAIN}" "${save_name}" "${checkpoint}" \
    >> "${CSV_PATH}"
  printf "[submitted] %s %s arm=%s seed=%s save=%s\n" \
    "${job_id}" "${job_name}" "${arm}" "${seed}" "${save_name}"
}

for seed in 1 2 3; do
  submit_job \
    "scratch" \
    "${seed}" \
    "scratch" \
    "scratch" \
    "front3d_anchor_scratch_p10_seed${seed}_rpn200" \
    "" \
    "anc_s_p10_s${seed}"

  submit_job \
    "joint_e300" \
    "${seed}" \
    "probe" \
    "joint_e300" \
    "front3d_anchor_joint_e300_p10_seed${seed}_rpn200" \
    "${JOINT_CKPT}" \
    "anc_j_p10_s${seed}"

  submit_job \
    "cosine_e300" \
    "${seed}" \
    "probe" \
    "cosine_e300" \
    "front3d_anchor_cosine_e300_p10_seed${seed}_rpn200" \
    "${COSINE_CKPT}" \
    "anc_c_p10_s${seed}"
done

{
  echo "# Anchor-head low-label jobs (${STAMP})"
  echo
  echo "- Purpose: second detector-head breadth check for Front3D 10% labels."
  echo "- Head: anchor-based NeRF-RPN via \`run_rpn.py\`."
  echo "- Arms: scratch, joint e300, structure-first/cosine e300."
  echo "- Seeds: 1, 2, 3."
  echo "- Percent train: ${PERCENT_TRAIN}."
  echo "- Anchor epochs: ${ANCHOR_NUM_EPOCHS}."
  echo "- Deterministic: ${DETERMINISTIC}."
  echo "- CSV: ${CSV_PATH}"
  echo
  echo '```csv'
  cat "${CSV_PATH}"
  echo '```'
} > "${MD_PATH}"

echo "[info] wrote ${CSV_PATH}"
echo "[info] wrote ${MD_PATH}"
