#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_SUFFIX="${RUN_SUFFIX:-abci3clean}"
LINK_MODE="${LINK_MODE:-symlink}"

BASELINE_E300_SEED1_CKPT="${BASELINE_E300_SEED1_CKPT:-}"
COSINE_RAMP_E300_SEED1_CKPT="${COSINE_RAMP_E300_SEED1_CKPT:-}"
COSINE_RAMP_ALPHA_SHUFFLE_E300_SEED1_CKPT="${COSINE_RAMP_ALPHA_SHUFFLE_E300_SEED1_CKPT:-}"

suffix_part=""
if [[ -n "${RUN_SUFFIX}" ]]; then
  suffix_part="_${RUN_SUFFIX}"
fi

fail() {
  echo "[error] $*" >&2
  exit 1
}

install_ckpt() {
  local label="$1"
  local src="$2"
  local save_name="$3"
  local dst_dir dst

  [[ -n "${src}" ]] || fail "${label} source env var is empty"
  [[ -f "${src}" ]] || fail "${label} checkpoint not found: ${src}"

  dst_dir="${ROOT_DIR}/output/nerf_mae/results/${save_name}"
  dst="${dst_dir}/epoch_300.pt"
  mkdir -p "${dst_dir}"

  if [[ -e "${dst}" || -L "${dst}" ]]; then
    rm -f "${dst}"
  fi

  case "${LINK_MODE}" in
    symlink)
      ln -s "${src}" "${dst}"
      ;;
    copy)
      cp -p "${src}" "${dst}"
      ;;
    *)
      fail "LINK_MODE must be symlink or copy: ${LINK_MODE}"
      ;;
  esac
  echo "[ok] ${label}: ${dst} -> ${src}"
}

install_ckpt \
  "baseline e300 seed1" \
  "${BASELINE_E300_SEED1_CKPT}" \
  "nerfmae_all_p1.0_e300_seed1${suffix_part}"

install_ckpt \
  "cosine_ramp e300 seed1" \
  "${COSINE_RAMP_E300_SEED1_CKPT}" \
  "nerfmae_alpha_rgba_curr_cosine_ramp_p1.0_e300_seed1${suffix_part}"

install_ckpt \
  "cosine_ramp_alpha_shuffle e300 seed1" \
  "${COSINE_RAMP_ALPHA_SHUFFLE_E300_SEED1_CKPT}" \
  "nerfmae_alpha_rgba_curr_cosine_ramp_alpha_shuffle_p1.0_e300_seed1${suffix_part}"
