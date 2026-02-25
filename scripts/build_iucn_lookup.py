#!/usr/bin/env python3
"""
Build IUCN Red List classification lookup for all bird species in eBird data.

Queries GBIF Species API (no key required) for IUCN status of each unique
speciesKey found in the Maharashtra and optionally neighboring state eBird data.

Idempotent: skips species already present in the output CSV.
Rate-limited: ~5 requests/second with automatic retry and backoff.

Output: data/ebird/iucn_species_lookup.csv

Usage:
    python scripts/build_iucn_lookup.py
    python scripts/build_iucn_lookup.py --include-neighbors
    python scripts/build_iucn_lookup.py --dry-run
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.migration.constants import EBIRD_DATA_DIR, IUCN_LOOKUP_CSV
from src.migration.iucn_lookup import build_iucn_lookup


def main():
    parser = argparse.ArgumentParser(
        description="Build IUCN Red List lookup from GBIF Species API"
    )
    parser.add_argument(
        "--include-neighbors", action="store_true",
        help="Also scan neighboring state eBird files for additional species",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Collect species but don't query GBIF API",
    )
    parser.add_argument(
        "--data-dir", default=EBIRD_DATA_DIR,
        help="Directory containing eBird .tsv.gz files",
    )
    parser.add_argument(
        "--output", default=IUCN_LOOKUP_CSV,
        help="Output CSV path",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("IUCN Red List Species Lookup Builder")
    print("=" * 60)

    build_iucn_lookup(
        include_neighbors=args.include_neighbors,
        data_dir=args.data_dir,
        output_path=args.output,
        dry_run=args.dry_run,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
