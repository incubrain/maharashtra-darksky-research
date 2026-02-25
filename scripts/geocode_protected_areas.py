#!/usr/bin/env python3
"""
Geocode protected areas using OpenStreetMap Nominatim.

Reads CSV files from data/protected_areas/, geocodes each site that lacks
coordinates, and writes updated CSVs in place.

Idempotent: skips sites that already have lat/lon.
Rate-limited: 1 request/second (Nominatim ToS).

Usage:
    python scripts/geocode_protected_areas.py
    python scripts/geocode_protected_areas.py --dry-run
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.geocoding.nominatim import USER_AGENT, RATE_LIMIT_SECONDS
from src.protected_areas.geocoding import geocode_site

DATA_DIR = "data/protected_areas"


def geocode_csv(csv_path: str, dry_run: bool = False) -> pd.DataFrame:
    """Geocode all pending sites in a single CSV file."""
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    df = pd.read_csv(csv_path)
    filename = os.path.basename(csv_path)
    print(f"\n  [{filename}] {len(df)} sites total")

    pending = df[
        (df["geocode_status"] == "pending")
        | (df["geocode_status"] == "failed")
    ]
    already_done = len(df) - len(pending)
    print(f"    Already geocoded: {already_done}, to geocode: {len(pending)}")

    if len(pending) == 0:
        print("    Nothing to geocode!")
        return df

    if dry_run:
        print("    DRY RUN — not actually geocoding")
        for _, row in pending.head(5).iterrows():
            district = row.get("district", "")
            print(f"      Would geocode: '{row['name']}' ({district})")
        if len(pending) > 5:
            print(f"      ... and {len(pending) - 5} more")
        return df

    geocoder = Nominatim(user_agent=USER_AGENT, timeout=10)
    rate_limiter = RateLimiter(geocoder.geocode, min_delay_seconds=RATE_LIMIT_SECONDS)

    ok_count = 0
    fallback_count = 0
    failed_count = 0

    for idx, row in pending.iterrows():
        district = row.get("district", "")
        if pd.isna(district):
            district = ""
        result = geocode_site(rate_limiter, row["name"], district)

        df.at[idx, "lat"] = result["lat"]
        df.at[idx, "lon"] = result["lon"]
        df.at[idx, "geocode_status"] = result["geocode_status"]

        if result["geocode_status"] == "ok":
            ok_count += 1
            print(f"    OK: {row['name']} -> {result['lat']}, {result['lon']}")
        elif result["geocode_status"] == "fallback":
            fallback_count += 1
            print(f"    FALLBACK: {row['name']} -> {result['lat']}, {result['lon']}")
        else:
            failed_count += 1
            print(f"    FAILED: {row['name']}")

    print(f"    Results: {ok_count} ok, {fallback_count} fallback, {failed_count} failed")

    df.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Geocode protected areas using Nominatim"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be geocoded without doing it",
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR,
        help="Directory containing protected area CSVs",
    )
    args = parser.parse_args()

    print("Protected Area Geocoding")
    print("=" * 60)

    csv_files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if not csv_files:
        print(f"ERROR: No CSV files found in {args.data_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files in {args.data_dir}")

    total_sites = 0
    total_geocoded = 0
    total_failed = 0

    for csv_path in csv_files:
        df = geocode_csv(csv_path, dry_run=args.dry_run)
        total_sites += len(df)
        total_geocoded += (df["geocode_status"].isin(["ok", "fallback", "provided"])).sum()
        total_failed += (df["geocode_status"] == "failed").sum()

    print(f"\n{'=' * 60}")
    print(f"Summary: {total_sites} sites, {total_geocoded} geocoded, {total_failed} failed")
    print(f"Success rate: {total_geocoded / total_sites * 100:.1f}%")
    print("\nDone!")


if __name__ == "__main__":
    main()
