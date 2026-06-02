#!/usr/bin/env bash
# Submit MixNeRF-MAE-lite smoke/scout jobs. Run from the repo root.
set -euo pipefail
PRETRAIN_SCRIPT="${PRETRAIN_SCRIPT:-nerf_mae/probe_scripts/abci3_e300_gate_pretrain.pbs}"
BASE_RUN="${BASE_RUN:-mixnerf_lite}"
SEED="${SEED:-1}"
RUN_SUFFIX="${RUN_SUFFIX:-abci3mix}"
PERCENT_TRAIN="${PERCENT_TRAIN:-1.0}"
MASK_RATIO="${MASK_RATIO:-0.75}"
QUEUE="${QUEUE:-rt_HF}"
PROJECT="${PROJECT:-gag51404}"
WALLTIME="${WALLTIME:-72:00:00}"
PRETRAIN_GPU_IDS="${PRETRAIN_GPU_IDS:-0:1:2:3:4:5:6:7}"
PRETRAIN_BATCH_SIZE_PER_GPU="${PRETRAIN_BATCH_SIZE_PER_GPU:-2}"

# Conservative staged launch: e10, e30, e100.  Comment out longer runs if desired.
declare -a EPOCHS=(10 30 100)
for E in "${EPOCHS[@]}"; do
  RUN_NAME="${BASE_RUN}_p${PERCENT_TRAIN}_e${E}_seed${SEED}"
  echo "Submitting ${RUN_NAME}"
  qsub -P "${PROJECT}" -q "${QUEUE}" -l select=1 -l "walltime=${WALLTIME}" \
    -N "mix_e${E}_pre" -j oe -o "output/launcher/${RUN_NAME}.pbs.log" \
    -v "KIND=mixnerf,CONDITION=mixnerf_lite,EPOCHS=${E},SEED=${SEED},RUN_SUFFIX=${RUN_SUFFIX},PRETRAIN_GPU_IDS=${PRETRAIN_GPU_IDS},PRETRAIN_BATCH_SIZE_PER_GPU=${PRETRAIN_BATCH_SIZE_PER_GPU},PRETRAIN_EVAL_INTERVAL=${E},PRETRAIN_CHECKPOINT_INTERVAL=50,PRETRAIN_KEEP_CHECKPOINTS=0,DETERMINISTIC=0,USE_WANDB=0,MIXNERF_MODE=mix,MIXNERF_MASK_RATIO=${MASK_RATIO},MIXNERF_FILL_MODE=partner,MIXNERF_DISABLE_INTERNAL_MASK=1,MIXNERF_LOG_STATS=1" \
    "$PRETRAIN_SCRIPT"
done
