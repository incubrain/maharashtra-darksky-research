#!/bin/bash
# Run city-level VIIRS ALAN analysis only.
#
# Usage:
#   bash scripts/run_city.sh
#   bash scripts/run_city.sh --years 2020-2024
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -d ".venv" ]]; then source .venv/bin/activate; else echo "ERROR: No venv"; exit 1; fi
fi

START_TIME=$(date +%s)

echo "[1/3] City analysis..."
python3 -m src.site.site_analysis --type city "$@"

echo "[2/3] Generating city maps..."
python3 -m src.outputs.generate_maps --type city "$@"

echo "[3/3] Generating city reports..."
python3 -m src.outputs.generate_reports --type city "$@"

END_TIME=$(date +%s)
echo "City pipeline complete in $((END_TIME - START_TIME))s"
