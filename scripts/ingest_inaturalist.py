#!/usr/bin/env python3
"""
Ingest iNaturalist Maharashtra bird observations.

Reads a CSV export from iNaturalist, matches species to GBIF speciesKeys,
and produces observation points that can be merged with the eBird pipeline.

Expected CSV export query:
    quality_grade=research&identifications=any&iconic_taxa[]=Aves
    &place_id=6683&verifiable=true&spam=false

Place the CSV at: data/inaturalist/maharashtra_birds.csv

Usage:
    python scripts/ingest_inaturalist.py
    python scripts/ingest_inaturalist.py --input path/to/export.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.migration.constants import INATURALIST_RAW_CSV
from src.migration.inaturalist import run_inaturalist_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Ingest iNaturalist bird observations for Maharashtra"
    )
    parser.add_argument(
        "--input", default=INATURALIST_RAW_CSV,
        help=f"Path to iNaturalist CSV export (default: {INATURALIST_RAW_CSV})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("iNaturalist Bird Observation Ingestion")
    print("=" * 60)

    obs = run_inaturalist_pipeline(raw_csv_path=args.input)

    if obs is not None:
        print(f"\nResult: {len(obs):,} observation points ready for pipeline merge")
        print("These will be automatically included when you re-run:")
        print("  python scripts/build_migration_profiles.py")
    else:
        print(f"\nNo data found. Place the iNaturalist CSV export at:")
        print(f"  {INATURALIST_RAW_CSV}")

    print("\nDone!")


if __name__ == "__main__":
    main()
