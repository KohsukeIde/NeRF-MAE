#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CHAIN_NAME="${CHAIN_NAME:-nerfmae_shortcut_probe_transfer_quick}"
export CHAIN_VARIANTS="${CHAIN_VARIANTS:-baseline,alpha_only,radiance_only}"
export RUN_PRETRAIN="${RUN_PRETRAIN:-0}"
export RUN_FCOS_FINETUNE="${RUN_FCOS_FINETUNE:-1}"
export RUN_FCOS_EVAL="${RUN_FCOS_EVAL:-1}"
export ALLOWED_GPU_IDS="${ALLOWED_GPU_IDS:-0,1,2,3}"
export TRANSFER_GPU_COUNT="${TRANSFER_GPU_COUNT:-1}"
export FCOS_NUM_EPOCHS="${FCOS_NUM_EPOCHS:-100}"
export FCOS_BATCH_SIZE_PER_GPU="${FCOS_BATCH_SIZE_PER_GPU:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export USE_WANDB="${USE_WANDB:-0}"
export PRETRAIN_RESULT_SUFFIX="${PRETRAIN_RESULT_SUFFIX:-p0.1_e10}"
export FCOS_RESULT_SUFFIX="${FCOS_RESULT_SUFFIX:-fcos100}"
export BLOCK_ON_TMUX_SESSIONS="${BLOCK_ON_TMUX_SESSIONS:-pcp_worldvis_base_100ep_chain,pcp_worldvis_base_100ep_resume_chain}"
export BLOCK_ON_PID_FILES="${BLOCK_ON_PID_FILES:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geopcp/full_chain/pcp_worldvis_base_100ep.chain.pid,/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geopcp/full_chain/pcp_worldvis_base_100ep.resume.chain.pid}"
export WAIT_MEMORY_USED_MAX_MIB="${WAIT_MEMORY_USED_MAX_MIB:-512}"
export WAIT_UTIL_MAX="${WAIT_UTIL_MAX:-10}"
export WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-30}"
export WAIT_STABLE_SECONDS="${WAIT_STABLE_SECONDS:-180}"
export SUMMARY_JSON="${SUMMARY_JSON:-/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/launcher/nerfmae_shortcut_probe_transfer_quick.json}"
export SUMMARY_CSV="${SUMMARY_CSV:-/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/launcher/nerfmae_shortcut_probe_transfer_quick.csv}"
export SUMMARY_ROWS_TSV="${SUMMARY_ROWS_TSV:-/mnt/urashima/users/minesawa/nerfmae_shortcut_probe/output/launcher/nerfmae_shortcut_probe_transfer_quick.rows.tsv}"

exec "${SCRIPT_DIR}/run_shortcut_probe_chain.sh"
