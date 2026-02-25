#!/usr/bin/env python3
"""
Download India states GeoJSON and extract Maharashtra + neighbors.

Downloads from datta07/INDIAN-SHAPEFILES (same source as district boundaries).

Output:
    data/shapefiles/india_states.geojson      - All India state boundaries
    data/shapefiles/maharashtra_region.geojson - Maharashtra + 7 neighbors

Usage:
    python scripts/download_neighbor_shapefiles.py
    python scripts/download_neighbor_shapefiles.py --force
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.migration.api_utils import make_session
from src.migration.constants import (
    INDIA_STATES_GEOJSON_URL,
    REGION_SHAPEFILE,
)

INDIA_STATES_PATH = "data/shapefiles/india_states.geojson"

# State names as they appear in the GeoJSON properties.
# Maharashtra + 7 bordering states.
REGION_STATES = {
    "Maharashtra",
    "Gujarat",
    "Madhya Pradesh",
    "Chhattisgarh",
    "Telangana",
    "Karnataka",
    "Goa",
    "Andhra Pradesh",
}


def download_india_states(output_path: str, force: bool = False) -> str:
    """Download the all-India states GeoJSON."""
    if os.path.isfile(output_path) and not force:
        print(f"  Already exists: {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"  Downloading from: {INDIA_STATES_GEOJSON_URL}")
    session = make_session()
    resp = session.get(INDIA_STATES_GEOJSON_URL, timeout=60)
    resp.raise_for_status()

    with open(output_path, "w") as f:
        f.write(resp.text)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
    return output_path


def extract_region(india_path: str, region_path: str, force: bool = False):
    """Filter India GeoJSON to Maharashtra + neighbors and save."""
    if os.path.isfile(region_path) and not force:
        print(f"  Already exists: {region_path}")
        return

    with open(india_path) as f:
        india = json.load(f)

    # Find the state name property key (varies across GeoJSON sources)
    sample = india["features"][0]["properties"]
    name_key = None
    for key in ["stname", "STNAME", "NAME_1", "name", "ST_NM", "state"]:
        if key in sample:
            name_key = key
            break

    if name_key is None:
        # Try to find any key whose values match known state names
        region_upper = {s.upper() for s in REGION_STATES}
        for key, val in sample.items():
            if isinstance(val, str) and val.upper() in region_upper:
                name_key = key
                break

    if name_key is None:
        print(f"  WARNING: Could not identify state name property in GeoJSON.")
        print(f"  Available properties: {list(sample.keys())}")
        print(f"  Saving full India GeoJSON as region file.")
        import shutil
        shutil.copy(india_path, region_path)
        return

    print(f"  State name property: '{name_key}'")

    # Build case-insensitive lookup
    region_upper = {s.upper() for s in REGION_STATES}

    # Filter features
    region_features = []
    found_states = set()
    for feature in india["features"]:
        state_name = feature["properties"].get(name_key, "")
        if state_name.upper() in region_upper:
            region_features.append(feature)
            found_states.add(state_name)

    found_upper = {s.upper() for s in found_states}
    missing = {s for s in REGION_STATES if s.upper() not in found_upper}
    if missing:
        print(f"  WARNING: States not found in GeoJSON: {missing}")

    region_geojson = {
        "type": "FeatureCollection",
        "features": region_features,
    }

    os.makedirs(os.path.dirname(region_path), exist_ok=True)
    with open(region_path, "w") as f:
        json.dump(region_geojson, f)

    print(f"  Region GeoJSON: {len(region_features)} states → {region_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download India states GeoJSON for migration mapping"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if files already exist",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Neighbor State Shapefile Downloader")
    print("=" * 60)

    print("\n1. Downloading India states GeoJSON...")
    download_india_states(INDIA_STATES_PATH, force=args.force)

    print("\n2. Extracting Maharashtra region...")
    extract_region(INDIA_STATES_PATH, REGION_SHAPEFILE, force=args.force)

    print("\nDone!")


if __name__ == "__main__":
    main()
