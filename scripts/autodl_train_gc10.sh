#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

DATA="${DATA:-${REPO_DIR}/GC10dataset/data.yaml}"
PROJECT="${PROJECT:-/root/autodl-tmp/yolo_gc10_runs}"
NAME="${NAME:-gc10_full_innov}"
EPOCHS="${EPOCHS:-300}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-auto}"
RUNS="${RUNS:-1}"
PRETRAINED="${PRETRAINED:-True}"

python train_gc10_full_innov.py \
  --data "${DATA}" \
  --project "${PROJECT}" \
  --name "${NAME}" \
  --epochs "${EPOCHS}" \
  --imgsz "${IMGSZ}" \
  --batch "${BATCH}" \
  --workers "${WORKERS}" \
  --device "${DEVICE}" \
  --runs "${RUNS}" \
  --pretrained "${PRETRAINED}" \
  "$@"
