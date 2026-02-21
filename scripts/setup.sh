#!/bin/bash
# Initialize the repository for a fresh clone or new worktree.
#
# Creates required directories, sets up the Python virtual environment,
# and installs dependencies.
#
# Usage:
#   bash scripts/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================================"
echo "Maharashtra VIIRS ALAN Pipeline — Repository Setup"
echo "============================================================"

# 1. Create required directories
echo ""
echo "[1/3] Creating required directories..."
mkdir -p logs
mkdir -p outputs
echo "  Created: logs/, outputs/"

# 2. Set up Python virtual environment
echo ""
echo "[2/3] Setting up Python virtual environment..."
if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
    echo "  Created .venv"
else
    echo "  .venv already exists"
fi

# 3. Install dependencies
echo ""
echo "[3/3] Installing dependencies..."
source .venv/bin/activate
pip install -q -r requirements.txt
echo "  Dependencies installed"

echo ""
echo "============================================================"
echo "Setup complete! Activate the environment with:"
echo "  source .venv/bin/activate"
echo "============================================================"
