#!/bin/bash
# Run the complete VIIRS ALAN analysis pipeline end-to-end.
#
# Stages: preprocess → district analysis → city analysis → site analysis
#         → map generation → report generation
#
# Usage:
#   bash scripts/run_all.sh
#   bash scripts/run_all.sh --years 2020-2024 --output-dir ./outputs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Check for Python venv
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -d ".venv" ]]; then
        source .venv/bin/activate
    else
        echo "ERROR: No virtual environment found. Run: python3 -m venv .venv && pip install -r requirements.txt"
        exit 1
    fi
fi

START_TIME=$(date +%s)

echo "============================================================"
echo "VIIRS ALAN Analysis Pipeline — Full Run"
echo "============================================================"

echo ""
echo "[1/6] Preprocessing VIIRS data..."
python3 -m src.preprocess "$@"

echo ""
echo "[2/6] District analysis..."
python3 -m src.viirs_process "$@"

echo ""
echo "[3/6] City analysis..."
python3 -m src.site.site_analysis --type city "$@"

echo ""
echo "[4/6] Dark-sky site analysis..."
python3 -m src.site.site_analysis --type site "$@"

echo ""
echo "[5/6] Generating maps..."
python3 -m src.outputs.generate_maps --type all "$@"

echo ""
echo "[6/6] Generating reports..."
python3 -m src.outputs.generate_reports --type all "$@"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "============================================================"
echo "Pipeline complete in ${ELAPSED}s"
echo "============================================================"
