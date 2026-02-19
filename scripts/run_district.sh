#!/bin/bash
# Run district-level VIIRS ALAN analysis only.
#
# Usage:
#   bash scripts/run_district.sh
#   bash scripts/run_district.sh --years 2020-2024
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -d ".venv" ]]; then source .venv/bin/activate; else echo "ERROR: No venv"; exit 1; fi
fi

START_TIME=$(date +%s)

echo "[1/3] District analysis..."
python3 -m src.viirs_process "$@"

echo "[2/3] Generating district maps..."
python3 -m src.outputs.generate_maps --type district "$@"

echo "[3/3] Generating district reports..."
python3 -m src.outputs.generate_reports --type district "$@"

END_TIME=$(date +%s)
echo "District pipeline complete in $((END_TIME - START_TIME))s"
