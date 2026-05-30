#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_WORK_ROOT="${DATA_WORK_ROOT:-${ROOT_DIR}/dataset/_downloads}"
SCANNET_RPN_URL="${SCANNET_RPN_URL:-https://huggingface.co/datasets/lyclyc52/NeRF_RPN/resolve/main/scannet_rpn_data.zip}"
SCANNET_ARCHIVE="${SCANNET_ARCHIVE:-${DATA_WORK_ROOT}/archives/scannet_rpn_data.zip}"
SCANNET_EXTRACT_DIR="${SCANNET_EXTRACT_DIR:-${DATA_WORK_ROOT}/scannet_rpn_extract}"
SCANNET_LINK="${SCANNET_LINK:-${ROOT_DIR}/dataset/finetune/scannet_rpn_data}"

fail() {
  echo "[error] $*" >&2
  exit 1
}

download_file() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "${out}")"
  wget --continue --tries=20 --timeout=60 --waitretry=15 -O "${out}" "${url}"
}

find_scannet_root() {
  local root="$1"
  local split candidate
  while IFS= read -r split; do
    candidate="$(dirname "${split}")"
    if [[ -d "${candidate}/features" && -d "${candidate}/obb" ]]; then
      printf "%s\n" "${candidate}"
      return 0
    fi
  done < <(find "${root}" -type f -name 'scannet_split.npz' -print)
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

validate_layout() {
  local root="$1"
  "${ROOT_DIR}/.venv-abci3/bin/python" - "${root}" <<'PY'
import os
import sys
import numpy as np

root = sys.argv[1]
required = [
    os.path.join(root, "features"),
    os.path.join(root, "obb"),
    os.path.join(root, "scannet_split.npz"),
]
missing = [p for p in required if not os.path.exists(p)]
if missing:
    raise SystemExit(f"missing required ScanNet RPN paths: {missing}")

features = [x for x in os.listdir(os.path.join(root, "features")) if x.endswith(".npz")]
boxes = [x for x in os.listdir(os.path.join(root, "obb")) if x.endswith(".npy")]
if not features:
    raise SystemExit(f"no feature npz files under {root}/features")
if not boxes:
    raise SystemExit(f"no OBB npy files under {root}/obb")
with np.load(os.path.join(root, "scannet_split.npz"), allow_pickle=True) as split:
    counts = {k: len(split[k]) for k in ["train_scenes", "val_scenes", "test_scenes"]}
print("[ok] ScanNet feature files:", len(features), "obb files:", len(boxes), "split:", counts)
PY
}

mkdir -p "${DATA_WORK_ROOT}/archives" "${SCANNET_EXTRACT_DIR}"

download_file "${SCANNET_RPN_URL}" "${SCANNET_ARCHIVE}"

if [[ ! -f "${SCANNET_EXTRACT_DIR}/.extract.done" ]]; then
  unzip -n "${SCANNET_ARCHIVE}" -d "${SCANNET_EXTRACT_DIR}"
  touch "${SCANNET_EXTRACT_DIR}/.extract.done"
else
  echo "[info] ScanNet extract already marked done: ${SCANNET_EXTRACT_DIR}"
fi

scannet_root="$(find_scannet_root "${SCANNET_EXTRACT_DIR}" || true)"
if [[ -z "${scannet_root}" ]]; then
  nested_zip="${SCANNET_EXTRACT_DIR}/scannet_rpn_data.zip"
  if [[ -f "${nested_zip}" ]]; then
    echo "[info] detected nested ScanNet archive; extracting ${nested_zip}"
    unzip -n "${nested_zip}" -d "${SCANNET_EXTRACT_DIR}"
    scannet_root="$(find_scannet_root "${SCANNET_EXTRACT_DIR}" || true)"
  fi
fi
[[ -n "${scannet_root}" ]] || fail "could not locate scannet_split.npz with features/obb under ${SCANNET_EXTRACT_DIR}"
validate_layout "${scannet_root}"
safe_symlink "${scannet_root}" "${SCANNET_LINK}"
echo "[ok] ScanNet RPN data is ready at ${SCANNET_LINK}"
