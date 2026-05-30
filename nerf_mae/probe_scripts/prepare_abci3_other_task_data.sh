#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/dataset/finetune/front3d_rpn_data}"
DATASET_NAME="${DATASET_NAME:-front3d}"
SPLIT_NAME="${SPLIT_NAME:-3dfront}"

SEMANTIC_SRC="${SEMANTIC_SRC:-}"
SR_SRC="${SR_SRC:-}"
SEMANTIC_LINK_NAME="${SEMANTIC_LINK_NAME:-voxel_${DATASET_NAME}}"
SR_LINK_NAME="${SR_LINK_NAME:-features_384}"

fail() {
  echo "[error] $*" >&2
  exit 1
}

info() {
  echo "[info] $*"
}

safe_symlink() {
  local src="$1"
  local dst="$2"
  local resolved_src resolved_dst
  [[ -d "${src}" ]] || fail "source directory does not exist: ${src}"
  mkdir -p "$(dirname "${dst}")"
  if [[ -L "${dst}" ]]; then
    ln -sfn "${src}" "${dst}"
    info "updated symlink ${dst} -> ${src}"
    return
  fi
  if [[ -e "${dst}" ]]; then
    resolved_src="$(readlink -f "${src}")"
    resolved_dst="$(readlink -f "${dst}")"
    if [[ "${resolved_src}" == "${resolved_dst}" ]]; then
      info "existing path already points to ${src}: ${dst}"
      return
    fi
    fail "${dst} already exists and is not the requested symlink target"
  fi
  ln -s "${src}" "${dst}"
  info "created symlink ${dst} -> ${src}"
}

find_semantic_candidate() {
  local c
  for c in \
    "${DATA_ROOT}/voxel_${DATASET_NAME}" \
    "${DATA_ROOT}/voxel_3dfront" \
    "${ROOT_DIR}/dataset/_downloads/front3d_rpn_extract/front3d_rpn_data/voxel_${DATASET_NAME}" \
    "/groups/gag51402/datasets/NeRF-MAE/front3d_rpn_data/voxel_${DATASET_NAME}" \
    "/groups/gag51402/datasets/front3d_rpn_data/voxel_${DATASET_NAME}" \
    "/groups/gag51402/datasets/hm3d_rpn_data/voxel_${DATASET_NAME}"; do
    if [[ -d "${c}" ]]; then
      printf "%s\n" "${c}"
      return 0
    fi
  done
  return 1
}

find_sr_candidate() {
  local c
  for c in \
    "${DATA_ROOT}/features_384" \
    "${ROOT_DIR}/dataset/_downloads/front3d_rpn_extract/front3d_rpn_data/features_384" \
    "/groups/gag51402/datasets/NeRF-MAE/front3d_rpn_data/features_384" \
    "/groups/gag51402/datasets/front3d_rpn_data/features_384" \
    "/groups/gag51402/datasets/hm3d_rpn_data/features_384"; do
    if [[ -d "${c}" ]]; then
      printf "%s\n" "${c}"
      return 0
    fi
  done
  return 1
}

validate_data_root() {
  [[ -d "${DATA_ROOT}/features" ]] || fail "missing low-res features: ${DATA_ROOT}/features"
  [[ -f "${DATA_ROOT}/${SPLIT_NAME}_split.npz" ]] || fail "missing split: ${DATA_ROOT}/${SPLIT_NAME}_split.npz"
}

validate_optional_target() {
  local kind="$1"
  local path="$2"
  local pattern="$3"
  local count
  if [[ ! -d "${path}" ]]; then
    return 1
  fi
  count="$(find -L "${path}" -maxdepth 1 -type f -name "${pattern}" | wc -l)"
  if [[ "${count}" -eq 0 ]]; then
    fail "${kind} directory exists but contains no ${pattern} files: ${path}"
  fi
  info "${kind} target files: ${count} under ${path}"
}

validate_data_root

if [[ -z "${SEMANTIC_SRC}" ]]; then
  SEMANTIC_SRC="$(find_semantic_candidate || true)"
fi
if [[ -z "${SR_SRC}" ]]; then
  SR_SRC="$(find_sr_candidate || true)"
fi

prepared=0
if [[ -n "${SEMANTIC_SRC}" ]]; then
  validate_optional_target "semantic" "${SEMANTIC_SRC}" "*.npy"
  safe_symlink "${SEMANTIC_SRC}" "${DATA_ROOT}/${SEMANTIC_LINK_NAME}"
  prepared=1
else
  info "semantic voxel target not found. Expected directory shape: <data_root>/voxel_${DATASET_NAME}/*.npy"
fi

if [[ -n "${SR_SRC}" ]]; then
  validate_optional_target "super-resolution" "${SR_SRC}" "*.npz"
  safe_symlink "${SR_SRC}" "${DATA_ROOT}/${SR_LINK_NAME}"
  prepared=1
else
  info "super-resolution target not found. Expected directory shape: <data_root>/features_384/*.npz"
fi

if [[ "${prepared}" -eq 0 ]]; then
  cat >&2 <<EOF
[error] No NeRF-MAE semantic/SR processed targets were found.

Current public NeRF-MAE README lists 3D Semantic Segmentation and
Voxel-Super Resolution finetuning data as "Coming Soon"; the released
Front3D detection archive only contains features/aabb/obb.

If you have private processed data, rerun with:
  SEMANTIC_SRC=/path/to/voxel_front3d \\
  SR_SRC=/path/to/features_384 \\
  bash ${SCRIPT_DIR}/prepare_abci3_other_task_data.sh
EOF
  exit 2
fi

info "other-task data links are ready under ${DATA_ROOT}"
