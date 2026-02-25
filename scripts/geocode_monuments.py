#!/usr/bin/env python3
"""
Geocode monuments using OpenStreetMap Nominatim.

Reads per-district CSV files from data/monuments/, geocodes each site
that lacks coordinates, and writes updated CSVs in place.

Monuments often share generic names (e.g. "Mahadev Temple") so the
geocoding strategy emphasises the *place* (village/locality) column
over the monument name for disambiguation.

Idempotent: skips sites that already have lat/lon.
Rate-limited: 1 request/second (Nominatim ToS).

Output: updated data/monuments/<district>.csv files.

Usage:
    python scripts/geocode_monuments.py
    python scripts/geocode_monuments.py --dry-run
    python scripts/geocode_monuments.py --file data/monuments/pune.csv
"""

import argparse
import glob
import os
import sys
import time

import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config

MAHARASHTRA_BBOX = config.MAHARASHTRA_BBOX
DATA_DIR = "data/monuments"

# Nominatim configuration
USER_AGENT = "maharashtra-darksky-research/1.0"
RATE_LIMIT_SECONDS = 1.1  # Slightly over 1s to be safe


def geocode_monument(geocoder, rate_limiter, name: str, place: str,
                     taluka: str, district: str, monument_type: str) -> dict:
    """Geocode a single monument with fallback strategies.

    Monuments often share generic names (Mahadev Temple, Shiva Temple)
    so the *place* (village) is the primary disambiguator.

    Returns dict with lat, lon, geocode_status.
    """
    # For forts, the fort name itself is often a well-known landmark
    is_fort = monument_type == "Fort"

    strategies = []
    if is_fort:
        # Forts are often well-known landmarks — try name first
        strategies.extend([
            f"{name}, {district} district, Maharashtra, India",
            f"{name}, {place}, {district}, Maharashtra, India",
            f"{place}, {taluka}, {district}, Maharashtra",
        ])
    else:
        # Temples/other: place (village) is more reliable than generic name
        strategies.extend([
            f"{name}, {place}, {district} district, Maharashtra, India",
            f"{place}, {taluka}, {district} district, Maharashtra, India",
            f"{name}, {district}, Maharashtra, India",
            f"{place}, {district}, Maharashtra, India",
        ])

    for i, query in enumerate(strategies):
        try:
            location = rate_limiter(query)
            if location is not None:
                lat, lon = location.latitude, location.longitude
                # Validate within Maharashtra bounding box (with margin)
                if (
                    MAHARASHTRA_BBOX["south"] - 0.5 <= lat <= MAHARASHTRA_BBOX["north"] + 0.5
                    and MAHARASHTRA_BBOX["west"] - 0.5 <= lon <= MAHARASHTRA_BBOX["east"] + 0.5
                ):
                    status = "ok" if i == 0 else "fallback"
                    return {
                        "lat": round(lat, 6),
                        "lon": round(lon, 6),
                        "geocode_status": status,
                    }
        except Exception as exc:
            print(f"    Geocode error for '{query}': {exc}")

    return {
        "lat": None,
        "lon": None,
        "geocode_status": "failed",
    }


def geocode_csv(csv_path: str, geocoder, rate_limiter, dry_run: bool = False) -> pd.DataFrame:
    """Geocode all pending sites in a single CSV file."""
    df = pd.read_csv(csv_path)
    filename = os.path.basename(csv_path)
    print(f"\n  [{filename}] {len(df)} monuments")

    # Identify pending sites
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
            print(f"      Would geocode: '{row['name']}' at {row['place']}, {row['district']}")
        if len(pending) > 5:
            print(f"      ... and {len(pending) - 5} more")
        return df

    ok_count = 0
    fallback_count = 0
    failed_count = 0

    for idx, row in pending.iterrows():
        result = geocode_monument(
            geocoder, rate_limiter,
            row["name"], row["place"], row["taluka"],
            row["district"], row["monument_type"],
        )

        df.at[idx, "lat"] = result["lat"]
        df.at[idx, "lon"] = result["lon"]
        df.at[idx, "geocode_status"] = result["geocode_status"]

        if result["geocode_status"] == "ok":
            ok_count += 1
        elif result["geocode_status"] == "fallback":
            fallback_count += 1
        else:
            failed_count += 1
            print(f"    FAILED: {row['name']} ({row['place']}, {row['district']})")

    print(f"    Results: {ok_count} ok, {fallback_count} fallback, {failed_count} failed")

    # Save progress
    df.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Geocode monuments using Nominatim"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be geocoded without doing it",
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR,
        help="Directory containing monument CSVs",
    )
    parser.add_argument(
        "--file", default=None,
        help="Geocode a single CSV file instead of all",
    )
    args = parser.parse_args()

    print("Monument Geocoding")
    print("=" * 60)

    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    geocoder = Nominatim(user_agent=USER_AGENT, timeout=30)
    rate_limiter = RateLimiter(geocoder.geocode, min_delay_seconds=RATE_LIMIT_SECONDS)

    if args.file:
        csv_files = [args.file]
    else:
        csv_files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))

    if not csv_files:
        print(f"ERROR: No CSV files found in {args.data_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files")

    total_sites = 0
    total_geocoded = 0
    total_failed = 0

    for csv_path in csv_files:
        df = geocode_csv(csv_path, geocoder, rate_limiter, dry_run=args.dry_run)
        total_sites += len(df)
        total_geocoded += (df["geocode_status"].isin(["ok", "fallback", "provided", "manual"])).sum()
        total_failed += (df["geocode_status"] == "failed").sum()

    print(f"\n{'=' * 60}")
    print(f"Summary: {total_sites} monuments, {total_geocoded} geocoded, {total_failed} failed")
    if total_sites > 0:
        print(f"Success rate: {total_geocoded / total_sites * 100:.1f}%")
    print("\nDone!")


if __name__ == "__main__":
    main()
