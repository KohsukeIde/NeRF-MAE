#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BUNDLE="${BUNDLE:-${ROOT_DIR}/nerfmae_abci_pretrain_checkpoints_20260519.zip}"
RUN_SUFFIX="${RUN_SUFFIX:-abci3clean}"
VERIFY_CHECKSUMS="${VERIFY_CHECKSUMS:-1}"
LINK_RUN_SUFFIX="${LINK_RUN_SUFFIX:-1}"
OVERWRITE_LINKS="${OVERWRITE_LINKS:-0}"

fail() {
  echo "[error] $*" >&2
  exit 1
}

[[ -f "${BUNDLE}" ]] || fail "bundle not found: ${BUNDLE}"
command -v unzip >/dev/null 2>&1 || fail "unzip is required"

echo "[info] extracting bundle without overwriting existing files: ${BUNDLE}"
unzip -n "${BUNDLE}" -d "${ROOT_DIR}"

if [[ "${VERIFY_CHECKSUMS}" == "1" ]]; then
  [[ -f "${ROOT_DIR}/checksums.sha256" ]] || fail "checksums.sha256 was not extracted"
  echo "[info] verifying extracted checkpoint checksums"
  (cd "${ROOT_DIR}" && sha256sum -c checksums.sha256)
fi

if [[ "${LINK_RUN_SUFFIX}" != "1" || -z "${RUN_SUFFIX}" ]]; then
  echo "[info] suffix symlink creation skipped"
  exit 0
fi

echo "[info] linking extracted checkpoints into RUN_SUFFIX=${RUN_SUFFIX} layout"
while read -r _checksum relpath; do
  [[ "${relpath}" == output/nerf_mae/results/*/epoch_*.pt ]] || continue
  src="${ROOT_DIR}/${relpath}"
  [[ -f "${src}" ]] || fail "listed checkpoint missing after extraction: ${src}"

  rel_dir="$(dirname "${relpath}")"
  filename="$(basename "${relpath}")"
  save_name="$(basename "${rel_dir}")"
  dst_dir="${ROOT_DIR}/output/nerf_mae/results/${save_name}_${RUN_SUFFIX}"
  dst="${dst_dir}/${filename}"
  mkdir -p "${dst_dir}"

  if [[ -e "${dst}" || -L "${dst}" ]]; then
    src_real="$(readlink -f "${src}")"
    dst_real="$(readlink -f "${dst}")"
    if [[ "${src_real}" == "${dst_real}" ]]; then
      echo "[ok] existing suffix link ${dst} -> ${src}"
      continue
    fi
    if [[ "${OVERWRITE_LINKS}" != "1" ]]; then
      fail "destination exists and points elsewhere: ${dst}"
    fi
    rm -f "${dst}"
  fi

  ln -s "${src}" "${dst}"
  echo "[ok] ${dst} -> ${src}"
done < "${ROOT_DIR}/checksums.sha256"

echo "[ok] checkpoint bundle is installed"
