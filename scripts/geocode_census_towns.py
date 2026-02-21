#!/usr/bin/env python3
"""
Geocode census towns using OpenStreetMap Nominatim.

Reads town_master.csv (produced by project_census_towns.py), geocodes each
unique (district, town_name) pair, and writes geocoded output.

Idempotent: skips towns that already have lat/lon in the output file.
Rate-limited: 1 request/second (Nominatim ToS).

Output: data/census/census_towns_geocoded.csv
    Columns: district, town_name, display_name, lat, lon, geocode_status,
             geocode_query, first_census, census_count

Usage:
    python scripts/geocode_census_towns.py
    python scripts/geocode_census_towns.py --dry-run
"""

import argparse
import os
import sys
import time

import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config

MAHARASHTRA_BBOX = config.MAHARASHTRA_BBOX
DATA_DIR = "data/census"
MASTER_PATH = os.path.join(DATA_DIR, "projected_towns", "town_master.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "census_towns_geocoded.csv")

# Nominatim configuration
USER_AGENT = "maharashtra-darksky-research/1.0"
RATE_LIMIT_SECONDS = 1.1  # Slightly over 1s to be safe


def load_master(path: str) -> pd.DataFrame:
    """Load the town master table."""
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run project_census_towns.py first.")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} towns from {path}")
    return df


def load_existing_geocoded(path: str) -> pd.DataFrame | None:
    """Load existing geocoded results for incremental updates."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  Found existing geocoded file: {len(df)} rows")
        return df
    return None


def clean_town_name_for_query(display_name: str) -> str:
    """Clean town name for geocoding query.

    Strips municipal classification suffixes but keeps meaningful
    identifiers like cantonment.
    """
    import re
    name = display_name.strip()
    # Remove municipal classification suffixes
    name = re.sub(
        r"\s*\("
        r"(?:M\.?\s*Corp\.?|M\s*Cl\.?|M\.?|Ct|Cb|R|NP|Mc|Og|"
        r"M\s*Corp\.?\s*\+\s*Og|M\s*Cl\s*\+\s*Og)"
        r"\)\s*\.?\s*$",
        "", name, flags=re.IGNORECASE,
    )
    # Remove (Part) suffix
    name = re.sub(r"\s*\(Part\)\s*$", "", name, flags=re.IGNORECASE)
    # Remove (Rural) suffix
    name = re.sub(r"\s*\(Rural\)\s*$", "", name, flags=re.IGNORECASE)
    # Clean up stray parenthetical residue
    name = re.sub(r"\s*\(\s*\)\s*$", "", name)
    return name.strip("* ").strip()


def geocode_town(
    geocoder,
    rate_limiter,
    town_name: str,
    district: str,
) -> dict:
    """Geocode a single town with fallback strategies.

    Returns dict with lat, lon, geocode_status, geocode_query.
    """
    strategies = [
        # Strategy 1: Town, District district, Maharashtra, India
        f"{town_name}, {district} district, Maharashtra, India",
        # Strategy 2: Town, District, Maharashtra
        f"{town_name}, {district}, Maharashtra",
        # Strategy 3: Town, Maharashtra, India (drop district)
        f"{town_name}, Maharashtra, India",
    ]

    for i, query in enumerate(strategies):
        try:
            location = rate_limiter(query)
            if location is not None:
                lat, lon = location.latitude, location.longitude
                # Validate within Maharashtra bounding box (with margin)
                if (MAHARASHTRA_BBOX["south"] - 0.5 <= lat <= MAHARASHTRA_BBOX["north"] + 0.5 and
                    MAHARASHTRA_BBOX["west"] - 0.5 <= lon <= MAHARASHTRA_BBOX["east"] + 0.5):
                    status = "ok" if i == 0 else "fallback"
                    return {
                        "lat": round(lat, 6),
                        "lon": round(lon, 6),
                        "geocode_status": status,
                        "geocode_query": query,
                    }
                # Location found but outside bbox — try next strategy
        except Exception as exc:
            print(f"    Geocode error for '{query}': {exc}")

    return {
        "lat": None,
        "lon": None,
        "geocode_status": "failed",
        "geocode_query": strategies[0],
    }


def geocode_all(master: pd.DataFrame, existing: pd.DataFrame | None, dry_run: bool = False, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    """Geocode all towns, skipping already-geocoded ones."""
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    # Build result DataFrame
    result_rows = []

    # Identify already-geocoded towns
    already_done = set()
    if existing is not None:
        for _, row in existing.iterrows():
            if pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
                already_done.add((row["district"], row["town_name"]))
                result_rows.append(row.to_dict())
            elif row.get("geocode_status") == "failed":
                # Re-try failed ones
                pass

    to_geocode = []
    for _, town in master.iterrows():
        key = (town["district"], town["norm_name"])
        if key not in already_done:
            to_geocode.append(town)

    print(f"\n  Already geocoded: {len(already_done)}")
    print(f"  To geocode: {len(to_geocode)}")

    if dry_run:
        print("\n  DRY RUN — not actually geocoding")
        for town in to_geocode[:10]:
            clean = clean_town_name_for_query(town["display_name"])
            print(f"    Would geocode: '{clean}' in {town['district']}")
        if len(to_geocode) > 10:
            print(f"    ... and {len(to_geocode) - 10} more")
        # Still return existing data
        return pd.DataFrame(result_rows) if result_rows else pd.DataFrame()

    if not to_geocode:
        print("  Nothing to geocode!")
        return pd.DataFrame(result_rows)

    # Initialize geocoder
    geocoder = Nominatim(user_agent=USER_AGENT, timeout=10)
    rate_limiter = RateLimiter(geocoder.geocode, min_delay_seconds=RATE_LIMIT_SECONDS)

    estimated_time = len(to_geocode) * RATE_LIMIT_SECONDS
    print(f"  Estimated time: {estimated_time / 60:.1f} minutes")
    print()

    # Group by district for batch progress reporting
    district_groups = {}
    for town in to_geocode:
        district_groups.setdefault(town["district"], []).append(town)

    total_done = 0
    total_ok = 0
    total_fallback = 0
    total_failed = 0

    for district, towns in sorted(district_groups.items()):
        print(f"  [{district}] ({len(towns)} towns)...")
        for town in towns:
            clean_name = clean_town_name_for_query(town["display_name"])
            result = geocode_town(geocoder, rate_limiter, clean_name, district)

            row = {
                "district": town["district"],
                "town_name": town["norm_name"],
                "display_name": town["display_name"],
                "first_census": town["first_census"],
                "census_count": town["census_count"],
                **result,
            }
            result_rows.append(row)

            total_done += 1
            if result["geocode_status"] == "ok":
                total_ok += 1
            elif result["geocode_status"] == "fallback":
                total_fallback += 1
            else:
                total_failed += 1
                print(f"    FAILED: {clean_name} ({district})")

        # Save progress after each district
        progress_df = pd.DataFrame(result_rows)
        progress_df.to_csv(output_path, index=False)

    print(f"\n  Geocoding complete: {total_ok} ok, {total_fallback} fallback, {total_failed} failed")
    return pd.DataFrame(result_rows)


def validate_results(df: pd.DataFrame):
    """Validate geocoded results."""
    print("\nValidation:")

    total = len(df)
    with_coords = df["lat"].notna().sum()
    failed = (df["geocode_status"] == "failed").sum()
    print(f"  Total: {total}, geocoded: {with_coords}, failed: {failed}")
    print(f"  Success rate: {with_coords / total * 100:.1f}%")

    # Check for out-of-bounds coordinates
    if with_coords > 0:
        valid = df[df["lat"].notna()]
        oob = valid[
            (valid["lat"] < MAHARASHTRA_BBOX["south"]) |
            (valid["lat"] > MAHARASHTRA_BBOX["north"]) |
            (valid["lon"] < MAHARASHTRA_BBOX["west"]) |
            (valid["lon"] > MAHARASHTRA_BBOX["east"])
        ]
        if len(oob) > 0:
            print(f"\n  WARNING: {len(oob)} towns outside Maharashtra bbox:")
            for _, row in oob.iterrows():
                print(f"    {row['display_name']} ({row['district']}): {row['lat']:.4f}N, {row['lon']:.4f}E")

    # Per-district summary
    print("\n  Per-district success:")
    for district in sorted(df["district"].unique()):
        d = df[df["district"] == district]
        ok = d["lat"].notna().sum()
        total_d = len(d)
        pct = ok / total_d * 100 if total_d > 0 else 0
        if pct < 100:
            print(f"    {district}: {ok}/{total_d} ({pct:.0f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Geocode census towns using Nominatim"
    )
    parser.add_argument("--master", default=MASTER_PATH, help="Path to town_master.csv")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Output path for geocoded CSV")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be geocoded without doing it")
    args = parser.parse_args()

    output_path = args.output

    print("Census Town Geocoding")
    print("=" * 60)

    print("\n1. Loading town master...")
    master = load_master(args.master)

    print("\n2. Checking existing geocoded data...")
    existing = load_existing_geocoded(output_path)

    print("\n3. Geocoding...")
    result = geocode_all(master, existing, dry_run=args.dry_run, output_path=output_path)

    if not args.dry_run and len(result) > 0:
        result.to_csv(output_path, index=False)
        print(f"\n  Saved: {output_path}")

        validate_results(result)

    print("\nDone!")


if __name__ == "__main__":
    main()
