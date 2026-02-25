#!/usr/bin/env python3
"""
Generate migration pathway visualizations for Maharashtra bird species.

Reads pre-computed profiles from build_migration_profiles.py and produces
static matplotlib maps consistent with the project's dark-theme aesthetic.

Output: outputs/ebird/*.png

Usage:
    python scripts/plot_migration_pathways.py
    python scripts/plot_migration_pathways.py --categories CR,EN,VU
    python scripts/plot_migration_pathways.py --skip-heatmaps
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.migration.constants import (
    MIGRATION_OUTPUT_DIR,
    NEIGHBOR_PROFILES_CSV,
    SPECIES_CLASSIFICATION_CSV,
    SPECIES_PROFILES_CSV,
)
from src.migration.visualization import run_all_visualizations


def main():
    parser = argparse.ArgumentParser(
        description="Generate bird migration pathway maps"
    )
    parser.add_argument(
        "--categories", default=None,
        help="Comma-separated IUCN codes to plot (e.g. CR,EN,VU). Default: all.",
    )
    parser.add_argument(
        "--output-dir", default=MIGRATION_OUTPUT_DIR,
        help="Output directory for PNG maps",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Bird Migration Pathway Visualization")
    print("=" * 60)

    # Load pre-computed data
    print("\nLoading data...")
    if not os.path.isfile(SPECIES_PROFILES_CSV):
        print(f"ERROR: {SPECIES_PROFILES_CSV} not found.")
        print("Run: python scripts/build_migration_profiles.py")
        sys.exit(1)

    if not os.path.isfile(SPECIES_CLASSIFICATION_CSV):
        print(f"ERROR: {SPECIES_CLASSIFICATION_CSV} not found.")
        print("Run: python scripts/build_migration_profiles.py")
        sys.exit(1)

    profiles_df = pd.read_csv(SPECIES_PROFILES_CSV, dtype={"speciesKey": str})
    classification_df = pd.read_csv(SPECIES_CLASSIFICATION_CSV, dtype={"speciesKey": str})

    print(f"  Profiles: {len(profiles_df)} rows, {profiles_df['speciesKey'].nunique()} species")
    print(f"  Classification: {len(classification_df)} species")

    # Migration class summary
    for cls in ["resident", "migratory", "passage"]:
        n = (classification_df["migration_class"] == cls).sum()
        print(f"    {cls}: {n}")

    # Load neighbor data if available
    neighbor_df = None
    if os.path.isfile(NEIGHBOR_PROFILES_CSV):
        neighbor_df = pd.read_csv(NEIGHBOR_PROFILES_CSV, dtype={"speciesKey": str})
        print(f"  Neighbor profiles: {len(neighbor_df)} rows")

    # Parse categories
    categories = None
    if args.categories:
        categories = [c.strip().upper() for c in args.categories.split(",")]
        print(f"\nFiltering to categories: {categories}")

    # Generate all maps
    run_all_visualizations(
        profiles_df=profiles_df,
        classification_df=classification_df,
        neighbor_profiles_df=neighbor_df,
        output_dir=args.output_dir,
        categories=categories,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
