#!/usr/bin/env bash
# T17 one-command reproduction entry point (thin wrapper around scripts/run_all.py).
#
# Usage (same options as the Python driver):
#   bash scripts/run_all.sh --list-stages
#   bash scripts/run_all.sh --stage check --dry-run
#   bash scripts/run_all.sh --stage deploy
#   bash scripts/run_all.sh --stage all --dry-run
#   bash scripts/run_all.sh --stage eval-sft --gpu-lease-id <LEASE-ID>
#   bash scripts/run_all.sh --stage eval-sft --cpu-eval      # slow, no GPU lease
#   bash scripts/run_all.sh --stage train-sft --allow-train --gpu-lease-id <LEASE-ID>
#
# Safety: training stages additionally require --allow-train; every GPU stage
# requires --gpu-lease-id; existing artifacts are skipped unless a force flag is
# passed; --dry-run executes nothing (no logs, no results, no model load).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "run_all.sh: python executable '${PYTHON_BIN}' not found (set PYTHON=... or install Python)" >&2
  exit 2
fi

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_all.py" "$@"
