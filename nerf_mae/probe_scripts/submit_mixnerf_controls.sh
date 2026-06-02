#!/usr/bin/env bash
# Submit controls for MixNeRF-MAE-lite. Run from the repo root.
# Controls are important to avoid confusing "mixed plausible filler" with generic noise.
set -euo pipefail
PRETRAIN_SCRIPT="${PRETRAIN_SCRIPT:-nerf_mae/probe_scripts/abci3_e300_gate_pretrain.pbs}"
BASE_RUN="${BASE_RUN:-mixnerf_ctrl}"
SEED="${SEED:-1}"
RUN_SUFFIX="${RUN_SUFFIX:-abci3mixctrl}"
NUM_EPOCHS="${NUM_EPOCHS:-30}"
PERCENT_TRAIN="${PERCENT_TRAIN:-1.0}"
MASK_RATIO="${MASK_RATIO:-0.75}"
QUEUE="${QUEUE:-rt_HF}"
PROJECT="${PROJECT:-gag51404}"
WALLTIME="${WALLTIME:-72:00:00}"
PRETRAIN_GPU_IDS="${PRETRAIN_GPU_IDS:-0:1:2:3:4:5:6:7}"
PRETRAIN_BATCH_SIZE_PER_GPU="${PRETRAIN_BATCH_SIZE_PER_GPU:-2}"

declare -a FILLS=(partner zeros noise)
for FILL in "${FILLS[@]}"; do
  RUN_NAME="${BASE_RUN}_${FILL}_p${PERCENT_TRAIN}_e${NUM_EPOCHS}_seed${SEED}"
  CONDITION="mixnerf_lite"
  if [[ "${FILL}" == "zeros" ]]; then
    CONDITION="mixnerf_lite_zeros"
  elif [[ "${FILL}" == "noise" ]]; then
    CONDITION="mixnerf_lite_noise"
  fi
  echo "Submitting ${RUN_NAME}"
  qsub -P "${PROJECT}" -q "${QUEUE}" -l select=1 -l "walltime=${WALLTIME}" \
    -N "mix_${FILL}_e${NUM_EPOCHS}_pre" -j oe -o "output/launcher/${RUN_NAME}.pbs.log" \
    -v "KIND=mixnerf,CONDITION=${CONDITION},EPOCHS=${NUM_EPOCHS},SEED=${SEED},RUN_SUFFIX=${RUN_SUFFIX},PRETRAIN_GPU_IDS=${PRETRAIN_GPU_IDS},PRETRAIN_BATCH_SIZE_PER_GPU=${PRETRAIN_BATCH_SIZE_PER_GPU},PRETRAIN_EVAL_INTERVAL=${NUM_EPOCHS},PRETRAIN_CHECKPOINT_INTERVAL=50,PRETRAIN_KEEP_CHECKPOINTS=0,DETERMINISTIC=0,USE_WANDB=0,MIXNERF_MODE=mix,MIXNERF_MASK_RATIO=${MASK_RATIO},MIXNERF_FILL_MODE=${FILL},MIXNERF_DISABLE_INTERNAL_MASK=1,MIXNERF_LOG_STATS=1" \
    "$PRETRAIN_SCRIPT"
done
