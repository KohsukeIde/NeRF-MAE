#!/usr/bin/env bash

set -euo pipefail

ROOT="/mnt/urashima/users/minesawa/nerfmae_shortcut_probe"
ARCHIVE="${ROOT}/downloads/NeRF-MAE_pretrain.tar.gz"
DATASET_ROOT="${ROOT}/dataset"
DONE_FILE="${ROOT}/downloads/pretrain_extract.done"
URL="https://s3.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/nerfmae/NeRF-MAE_pretrain.tar.gz"

mkdir -p "${ROOT}/downloads" "${DATASET_ROOT}"

if [ -f "${DONE_FILE}" ]; then
  echo "pretrain extraction already completed: ${DONE_FILE}"
  exit 0
fi

wget -c --tries=0 --retry-connrefused --waitretry=30 -O "${ARCHIVE}" "${URL}"
tar -xzf "${ARCHIVE}" -C "${DATASET_ROOT}"
touch "${DONE_FILE}"
