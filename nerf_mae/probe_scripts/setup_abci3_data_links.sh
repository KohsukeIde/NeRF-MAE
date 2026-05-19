#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-/groups/gag51402/datasets}"
PRETRAIN_DATA_SRC="${PRETRAIN_DATA_SRC:-}"
FCOS_DATA_SRC="${FCOS_DATA_SRC:-}"
PRETRAIN_LINK="${PRETRAIN_LINK:-${ROOT_DIR}/dataset/pretrain}"
FCOS_LINK="${FCOS_LINK:-${ROOT_DIR}/dataset/finetune/front3d_rpn_data}"
REQUIRE_PRETRAIN_DATA="${REQUIRE_PRETRAIN_DATA:-1}"
REQUIRE_FCOS_DATA="${REQUIRE_FCOS_DATA:-1}"

fail() {
  echo "[error] $*" >&2
  exit 1
}

check_pretrain_src() {
  local src="$1"
  [[ -d "${src}/features" && -f "${src}/nerfmae_split.npz" ]]
}

check_fcos_src() {
  local src="$1"
  [[ -d "${src}/features" && -d "${src}/obb" && -f "${src}/3dfront_split.npz" ]]
}

find_pretrain_candidate() {
  local src
  for src in \
    "${DATASET_ROOT}/nerfmae" \
    "${DATASET_ROOT}/NeRF-MAE/pretrain" \
    "${DATASET_ROOT}/front3d_nerfmae/pretrain" \
    "${DATASET_ROOT}/Structure3D/nerfmae"; do
    if check_pretrain_src "${src}"; then
      printf "%s\n" "${src}"
      return 0
    fi
  done
  return 1
}

find_fcos_candidate() {
  local src
  for src in \
    "${DATASET_ROOT}/front3d_rpn_data" \
    "${DATASET_ROOT}/3dfront_rpn_data" \
    "${DATASET_ROOT}/NeRF-MAE/front3d_rpn_data" \
    "${DATASET_ROOT}/Structure3D/front3d_rpn_data"; do
    if check_fcos_src "${src}"; then
      printf "%s\n" "${src}"
      return 0
    fi
  done
  return 1
}

safe_symlink() {
  local src="$1"
  local dst="$2"
  local resolved_src resolved_dst

  [[ -d "${src}" ]] || fail "source directory does not exist: ${src}"
  mkdir -p "$(dirname "${dst}")"

  if [[ -L "${dst}" ]]; then
    ln -sfn "${src}" "${dst}"
    echo "[ok] updated symlink ${dst} -> ${src}"
    return
  fi

  if [[ -e "${dst}" ]]; then
    resolved_src="$(readlink -f "${src}")"
    resolved_dst="$(readlink -f "${dst}")"
    if [[ "${resolved_src}" == "${resolved_dst}" ]]; then
      echo "[ok] existing path already points to ${src}: ${dst}"
      return
    fi
    fail "${dst} already exists and is not the requested symlink target"
  fi

  ln -s "${src}" "${dst}"
  echo "[ok] created symlink ${dst} -> ${src}"
}

if [[ "${REQUIRE_PRETRAIN_DATA}" == "1" && -z "${PRETRAIN_DATA_SRC}" ]]; then
  PRETRAIN_DATA_SRC="$(find_pretrain_candidate || true)"
fi
if [[ "${REQUIRE_FCOS_DATA}" == "1" && -z "${FCOS_DATA_SRC}" ]]; then
  FCOS_DATA_SRC="$(find_fcos_candidate || true)"
fi

if [[ "${REQUIRE_PRETRAIN_DATA}" == "1" && -z "${PRETRAIN_DATA_SRC}" ]] || [[ "${REQUIRE_FCOS_DATA}" == "1" && -z "${FCOS_DATA_SRC}" ]]; then
  {
    cat <<EOF
[error] Could not auto-detect preprocessed NeRF-MAE data under ${DATASET_ROOT}.

EOF
    if [[ "${REQUIRE_PRETRAIN_DATA}" == "1" ]]; then
      cat <<EOF
Required pretrain source:
  <PRETRAIN_DATA_SRC>/features
  <PRETRAIN_DATA_SRC>/nerfmae_split.npz

EOF
    fi
    if [[ "${REQUIRE_FCOS_DATA}" == "1" ]]; then
      cat <<EOF
Required FCOS source:
  <FCOS_DATA_SRC>/features
  <FCOS_DATA_SRC>/obb
  <FCOS_DATA_SRC>/3dfront_split.npz

EOF
    fi
    cat <<EOF
Run again with explicit paths, for example:
EOF
    if [[ "${REQUIRE_PRETRAIN_DATA}" == "1" ]]; then
      cat <<EOF
  PRETRAIN_DATA_SRC=/path/to/pretrain \\
EOF
    fi
    if [[ "${REQUIRE_PRETRAIN_DATA}" == "0" ]]; then
      cat <<EOF
  REQUIRE_PRETRAIN_DATA=0 \\
EOF
    fi
    cat <<EOF
  FCOS_DATA_SRC=/path/to/front3d_rpn_data \\
  bash ${SCRIPT_DIR}/setup_abci3_data_links.sh

Note: ${DATASET_ROOT}/Structure3D looks like raw/converted Structured3D data, not the preprocessed features+boxes format expected by these scripts.
EOF
  } >&2
  exit 1
fi

if [[ "${REQUIRE_PRETRAIN_DATA}" == "1" ]]; then
  check_pretrain_src "${PRETRAIN_DATA_SRC}" || fail "invalid PRETRAIN_DATA_SRC: ${PRETRAIN_DATA_SRC}"
fi
if [[ "${REQUIRE_FCOS_DATA}" == "1" ]]; then
  check_fcos_src "${FCOS_DATA_SRC}" || fail "invalid FCOS_DATA_SRC: ${FCOS_DATA_SRC}"
fi

if [[ "${REQUIRE_PRETRAIN_DATA}" == "1" ]]; then
  safe_symlink "${PRETRAIN_DATA_SRC}" "${PRETRAIN_LINK}"
fi
if [[ "${REQUIRE_FCOS_DATA}" == "1" ]]; then
  safe_symlink "${FCOS_DATA_SRC}" "${FCOS_LINK}"
fi

echo "[ok] data links are ready"
