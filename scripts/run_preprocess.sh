#!/bin/bash
# Run VIIRS data preprocessing only (download, unpack, subset).
#
# Usage:
#   bash scripts/run_preprocess.sh
#   bash scripts/run_preprocess.sh --years 2024
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -d ".venv" ]]; then source .venv/bin/activate; else echo "ERROR: No venv"; exit 1; fi
fi

START_TIME=$(date +%s)

echo "Preprocessing VIIRS data..."
python3 -m src.preprocess "$@"

END_TIME=$(date +%s)
echo "Preprocessing complete in $((END_TIME - START_TIME))s"
