#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_WORK_ROOT="${DATA_WORK_ROOT:-${ROOT_DIR}/dataset/_downloads}"
PRETRAIN_URL="${PRETRAIN_URL:-https://s3.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/nerfmae/NeRF-MAE_pretrain.tar.gz}"
FRONT3D_RPN_URL="${FRONT3D_RPN_URL:-https://huggingface.co/datasets/lyclyc52/NeRF_RPN/resolve/main/front3d_rpn_data.zip}"
DOWNLOAD_PRETRAIN="${DOWNLOAD_PRETRAIN:-1}"
DOWNLOAD_FCOS="${DOWNLOAD_FCOS:-1}"
PRETRAIN_LINK="${PRETRAIN_LINK:-${ROOT_DIR}/dataset/pretrain}"
FCOS_LINK="${FCOS_LINK:-${ROOT_DIR}/dataset/finetune/front3d_rpn_data}"

PRETRAIN_ARCHIVE="${PRETRAIN_ARCHIVE:-${DATA_WORK_ROOT}/archives/NeRF-MAE_pretrain.tar.gz}"
FRONT3D_RPN_ARCHIVE="${FRONT3D_RPN_ARCHIVE:-${DATA_WORK_ROOT}/archives/front3d_rpn_data.zip}"
PRETRAIN_EXTRACT_DIR="${PRETRAIN_EXTRACT_DIR:-${DATA_WORK_ROOT}/pretrain_extract}"
FCOS_EXTRACT_DIR="${FCOS_EXTRACT_DIR:-${DATA_WORK_ROOT}/front3d_rpn_extract}"

fail() {
  echo "[error] $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 is required"
}

download_file() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "${out}")"
  if [[ -f "${out}" ]]; then
    echo "[info] resume/check existing download: ${out}"
  else
    echo "[info] start download: ${url}"
  fi
  wget --continue --tries=20 --timeout=60 --waitretry=15 -O "${out}" "${url}"
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

find_pretrain_root() {
  local root="$1"
  local split candidate
  while IFS= read -r split; do
    candidate="$(dirname "${split}")"
    if [[ -d "${candidate}/features" ]]; then
      printf "%s\n" "${candidate}"
      return 0
    fi
  done < <(find "${root}" -type f -name 'nerfmae_split.npz' -print)
  return 1
}

find_fcos_root() {
  local root="$1"
  local split candidate
  while IFS= read -r split; do
    candidate="$(dirname "${split}")"
    if [[ -d "${candidate}/features" && -d "${candidate}/obb" ]]; then
      printf "%s\n" "${candidate}"
      return 0
    fi
  done < <(find "${root}" -type f -name '3dfront_split.npz' -print)
  return 1
}

validate_layout() {
  local pretrain_root="$1"
  local fcos_root="$2"
  "${ROOT_DIR}/.venv-abci3/bin/python" - "${pretrain_root}" "${fcos_root}" <<'PY'
import os
import sys
import numpy as np

pretrain, fcos = sys.argv[1:3]

def check_npz(path, required):
    with np.load(path, allow_pickle=True) as f:
        missing = [k for k in required if k not in f]
        if missing:
            raise SystemExit(f"{path} missing keys {missing}")
        return {k: len(f[k]) for k in required}

pretrain_features = [x for x in os.listdir(os.path.join(pretrain, "features")) if x.endswith(".npz")]
fcos_features = [x for x in os.listdir(os.path.join(fcos, "features")) if x.endswith(".npz")]
fcos_boxes = [x for x in os.listdir(os.path.join(fcos, "obb")) if x.endswith(".npy")]
if not pretrain_features:
    raise SystemExit(f"no pretrain feature npz files under {pretrain}/features")
if not fcos_features:
    raise SystemExit(f"no FCOS feature npz files under {fcos}/features")
if not fcos_boxes:
    raise SystemExit(f"no FCOS OBB npy files under {fcos}/obb")
pretrain_split = check_npz(os.path.join(pretrain, "nerfmae_split.npz"), ["train_scenes", "val_scenes", "test_scenes"])
fcos_split = check_npz(os.path.join(fcos, "3dfront_split.npz"), ["train_scenes", "val_scenes", "test_scenes"])
print("[ok] pretrain feature files:", len(pretrain_features), "split:", pretrain_split)
print("[ok] fcos feature files:", len(fcos_features), "obb files:", len(fcos_boxes), "split:", fcos_split)
PY
}

need_cmd wget
need_cmd tar
need_cmd unzip

mkdir -p "${DATA_WORK_ROOT}/archives"

pretrain_root=""
fcos_root=""

if [[ "${DOWNLOAD_PRETRAIN}" == "1" ]]; then
  download_file "${PRETRAIN_URL}" "${PRETRAIN_ARCHIVE}"
  if [[ ! -f "${PRETRAIN_EXTRACT_DIR}/.extract.done" ]]; then
    mkdir -p "${PRETRAIN_EXTRACT_DIR}"
    echo "[info] extracting pretrain archive to ${PRETRAIN_EXTRACT_DIR}"
    tar -xzf "${PRETRAIN_ARCHIVE}" -C "${PRETRAIN_EXTRACT_DIR}"
    touch "${PRETRAIN_EXTRACT_DIR}/.extract.done"
  else
    echo "[info] pretrain extract already marked done: ${PRETRAIN_EXTRACT_DIR}"
  fi
  pretrain_root="$(find_pretrain_root "${PRETRAIN_EXTRACT_DIR}")" || fail "could not locate pretrain root after extraction"
  safe_symlink "${pretrain_root}" "${PRETRAIN_LINK}"
fi

if [[ "${DOWNLOAD_FCOS}" == "1" ]]; then
  download_file "${FRONT3D_RPN_URL}" "${FRONT3D_RPN_ARCHIVE}"
  if [[ ! -f "${FCOS_EXTRACT_DIR}/.extract.done" ]]; then
    mkdir -p "${FCOS_EXTRACT_DIR}"
    echo "[info] extracting Front3D RPN archive to ${FCOS_EXTRACT_DIR}"
    unzip -n "${FRONT3D_RPN_ARCHIVE}" -d "${FCOS_EXTRACT_DIR}"
    touch "${FCOS_EXTRACT_DIR}/.extract.done"
  else
    echo "[info] Front3D RPN extract already marked done: ${FCOS_EXTRACT_DIR}"
  fi
  fcos_root="$(find_fcos_root "${FCOS_EXTRACT_DIR}")" || fail "could not locate FCOS root after extraction"
  safe_symlink "${fcos_root}" "${FCOS_LINK}"
fi

if [[ -z "${pretrain_root}" ]]; then
  pretrain_root="$(readlink -f "${PRETRAIN_LINK}")"
fi
if [[ -z "${fcos_root}" ]]; then
  fcos_root="$(readlink -f "${FCOS_LINK}")"
fi

validate_layout "${pretrain_root}" "${fcos_root}"
echo "[ok] ABCI3 data is ready"
