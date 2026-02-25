#!/usr/bin/env python3
"""
Build monthly species observation profiles and classify migratory vs resident.

Processes eBird data in chunks to produce:
1. Monthly observation profiles (species × month aggregates)
2. Species migration classification (migratory vs resident vs passage)
3. Neighboring state context for migratory species entry/exit detection

Output:
    data/ebird/species_monthly_profiles.csv
    data/ebird/species_classification.csv
    data/ebird/neighbor_monthly_profiles.csv

Usage:
    python scripts/build_migration_profiles.py
    python scripts/build_migration_profiles.py --skip-neighbors
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.migration.constants import EBIRD_DATA_DIR
from src.migration.profiling import run_profiling_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Build bird migration profiles from eBird data"
    )
    parser.add_argument(
        "--skip-neighbors", action="store_true",
        help="Skip neighboring state analysis (faster, Maharashtra only)",
    )
    parser.add_argument(
        "--data-dir", default=EBIRD_DATA_DIR,
        help="Directory containing eBird .tsv.gz files",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Bird Migration Profile Builder")
    print("=" * 60)

    run_profiling_pipeline(
        data_dir=args.data_dir,
        skip_neighbors=args.skip_neighbors,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
