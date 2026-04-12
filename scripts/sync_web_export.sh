#!/usr/bin/env bash
# Sync the generated web-export/ bundle into the astronera Nuxt app.
#
# Run AFTER the pipeline has produced web-export/ (i.e. after running
# `python3 -m src.pipeline_runner --pipeline district --export-web`).
#
# Usage:
#   bash scripts/sync_web_export.sh
#   bash scripts/sync_web_export.sh /path/to/astronera/public/data/maharashtra
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$REPO_ROOT/web-export"
DEFAULT_DEST="$REPO_ROOT/../../astronera/public/data/maharashtra"
DEST="${1:-$DEFAULT_DEST}"

if [ ! -d "$SRC" ]; then
  echo "No web-export/ at $SRC — run the pipeline with --export-web first." >&2
  exit 1
fi

mkdir -p "$DEST"
rsync -av --delete "$SRC/" "$DEST/"
echo "Synced $SRC → $DEST"
