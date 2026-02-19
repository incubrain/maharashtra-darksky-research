#!/bin/bash
# Run dark-sky site VIIRS ALAN analysis only.
#
# Usage:
#   bash scripts/run_site.sh
#   bash scripts/run_site.sh --years 2020-2024
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -d ".venv" ]]; then source .venv/bin/activate; else echo "ERROR: No venv"; exit 1; fi
fi

START_TIME=$(date +%s)

echo "[1/3] Dark-sky site analysis..."
python3 -m src.site.site_analysis --type site "$@"

echo "[2/3] Generating site maps..."
python3 -m src.outputs.generate_maps --type site "$@"

echo "[3/3] Generating site reports..."
python3 -m src.outputs.generate_reports --type site "$@"

END_TIME=$(date +%s)
echo "Site pipeline complete in $((END_TIME - START_TIME))s"
