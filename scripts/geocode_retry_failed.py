#!/usr/bin/env python3
"""
Retry geocoding for towns that failed with Nominatim.

Uses alternative free geocoders (Photon, ArcGIS) and smarter query
cleaning to resolve remaining towns. Reads and updates the existing
census_towns_geocoded.csv in-place.

Geocoder cascade:
  1. Photon (komoot) — OSM data, Elasticsearch engine, no API key
  2. ArcGIS (Esri)  — non-OSM dataset, no API key (unauthenticated)
  3. Nominatim       — OSM data, re-tried with cleaned query variants

All three are included in geopy 2.4.1.

Usage:
    python scripts/geocode_retry_failed.py
    python scripts/geocode_retry_failed.py --dry-run
"""

import argparse
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config

MAHARASHTRA_BBOX = config.MAHARASHTRA_BBOX
DATA_DIR = "data/census"
GEOCODED_PATH = os.path.join(DATA_DIR, "census_towns_geocoded.csv")

USER_AGENT = "maharashtra-darksky-research/1.0"
RATE_LIMIT_SECONDS = 1.1


# ── Query cleaning ────────────────────────────────────────────────────

# Known name corrections: census name → better search term
NAME_CORRECTIONS = {
    "Ichalkarnji": "Ichalkaranji",
    "Pethumri": "Pathri",
    "Yevla": "Yeola",
    "Manjlegaon": "Manjlegaon",
    "Kurundvad": "Kurundwad",
    "Pandharkaoda": "Pandharkaoda",  # Also try "Pandharkawada"
    "Roha Ashtami": "Roha",
    "Dapoli Camp": "Dapoli",
    "Kanhan (Pipri)": "Kanhan",
    "Shivatkar (Nira)": "Nira",
    "Vadgaon Sheri": "Vadgaon Sheri",
    "Koregaon Bhima": "Koregaon Bhima",
    "Khadakvasla": "Khadakwasla",
    "Kirkee": "Khadki",
    "Sironcha Ry.": "Sironcha",
    "Sawangi (Meghe)": "Sawangi Meghe",
    "Sonegaon (Nipani)": "Sonegaon",
    "Vanvadi (Sadashivgad)": "Sadashivgad",
    "Sindi Turf Hindnagar": "Sindhi Meghe",
}

# Alternative name attempts for towns that need multiple tries
ALTERNATIVE_NAMES = {
    "Pandharkaoda": ["Pandharkawada", "Pandharkaoda"],
    "Ichalkarnji": ["Ichalkaranji", "Ichalkarnji"],
    "Pethumri": ["Pathri", "Pethumri"],
    "Yevla": ["Yeola", "Yevla"],
    "Kurundvad": ["Kurundwad", "Kurundvad"],
    "Kirkee": ["Khadki", "Kirkee Cantonment", "Kirkee"],
    "Kamptee": ["Kamptee", "Kamthi"],
    "Nawapur": ["Navapur", "Nawapur"],
    "Talode": ["Taloda", "Talode"],
    "Shirpur-Warwade": ["Shirpur", "Shirpur Warwade"],
    "Dondaicha-Warwade": ["Dondaicha", "Dondaicha Warwade"],
    "Murgad": ["Murgud", "Murgad"],
    "Roha Ashtami": ["Roha"],
    "Dapoli Camp": ["Dapoli"],
    "Alore": ["Alorye", "Alore", "Aare Ware"],
    "Kanhan (Pipri)": ["Kanhan", "Pipri"],
    "Khadakvasla": ["Khadakwasla", "Khadakvasla"],
    "Dehu Road": ["Dehu Road", "Dehuroad"],
    "Vadgaon Sheri": ["Vadgaon Sheri", "Vadgaonsheri"],
    "Sironcha Ry.": ["Sironcha"],
    "Sawangi (Meghe)": ["Sawangi Meghe", "Sawangi"],
    "Sonegaon (Nipani)": ["Sonegaon Nipani", "Sonegaon"],
    "Vashind": ["Vasind", "Vashind"],
    "Kasara Budruk": ["Kasara", "Kasara Budruk"],
}


def clean_query_name(display_name: str) -> str:
    """Aggressively clean a census town name for geocoding.

    Goes beyond the basic cleaner: strips OG suffixes, cantonment
    markers, 'Turf/Tarf' constructs, Roman numeral prefixes, etc.
    """
    name = display_name.strip()

    # Remove all parenthetical suffixes (M Corp., M Cl, Ct, Og, Part, Rural, C.T., etc.)
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
    # Handle nested: "Greater Mumbai (M Corp.) (Part)"
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)

    # Remove "Ii)" or "Iii)" Roman numeral prefixes
    name = re.sub(r"^[IiVvXx]+\)", "", name).strip()

    # Remove "Cantt." suffix
    name = re.sub(r"\s*Cantt\.?\s*$", "", name, flags=re.IGNORECASE)

    # Remove "Ry." (Railway) suffix
    name = re.sub(r"\s*Ry\.?\s*$", "", name, flags=re.IGNORECASE)

    # Apply known corrections
    if name in NAME_CORRECTIONS:
        name = NAME_CORRECTIONS[name]

    # Clean trailing dots, asterisks, whitespace
    name = name.strip("* .\t")

    return name


def get_query_variants(clean_name: str, district: str) -> list[str]:
    """Generate multiple query variants for a town.

    Returns a list of query strings to try, from most specific to least.
    """
    # Check for known alternative names
    alternatives = ALTERNATIVE_NAMES.get(clean_name, [clean_name])

    queries = []
    for alt in alternatives:
        queries.append(f"{alt}, {district} district, Maharashtra, India")
        queries.append(f"{alt}, {district}, Maharashtra, India")
        queries.append(f"{alt}, Maharashtra, India")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    return unique


def is_in_maharashtra(lat: float, lon: float) -> bool:
    """Check if coordinates fall within Maharashtra bounding box (with margin)."""
    return (
        MAHARASHTRA_BBOX["south"] - 0.5 <= lat <= MAHARASHTRA_BBOX["north"] + 0.5
        and MAHARASHTRA_BBOX["west"] - 0.5 <= lon <= MAHARASHTRA_BBOX["east"] + 0.5
    )


def try_geocoder(geocoder_fn, queries: list[str], geocoder_name: str) -> dict | None:
    """Try a geocoder with multiple query variants.

    Returns result dict on success, None on failure.
    """
    for query in queries:
        try:
            location = geocoder_fn(query)
            if location is not None:
                lat, lon = location.latitude, location.longitude
                if is_in_maharashtra(lat, lon):
                    return {
                        "lat": round(lat, 6),
                        "lon": round(lon, 6),
                        "geocode_status": f"ok_{geocoder_name}",
                        "geocode_query": query,
                    }
        except Exception as exc:
            # Silently continue to next query variant
            pass
    return None


def retry_failed(df: pd.DataFrame, dry_run: bool = False) -> pd.DataFrame:
    """Retry geocoding on all failed rows using Photon + ArcGIS cascade."""
    from geopy.geocoders import Photon, ArcGIS, Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    failed_mask = df["geocode_status"] == "failed"
    failed = df[failed_mask].copy()
    total_failed = len(failed)

    print(f"\nFailed towns to retry: {total_failed}")

    if total_failed == 0:
        print("Nothing to retry!")
        return df

    if dry_run:
        print("\nDRY RUN — showing cleaned queries:")
        for _, row in failed.head(20).iterrows():
            clean = clean_query_name(row["display_name"])
            variants = get_query_variants(clean, row["district"])
            print(f"  {row['display_name']:40s} -> {clean:30s} ({len(variants)} variants)")
        if total_failed > 20:
            print(f"  ... and {total_failed - 20} more")
        return df

    # Initialize geocoders
    photon = Photon(user_agent=USER_AGENT, timeout=10)
    arcgis = ArcGIS(timeout=10)
    nominatim = Nominatim(user_agent=USER_AGENT, timeout=10)

    photon_rl = RateLimiter(photon.geocode, min_delay_seconds=RATE_LIMIT_SECONDS)
    arcgis_rl = RateLimiter(arcgis.geocode, min_delay_seconds=RATE_LIMIT_SECONDS)
    nominatim_rl = RateLimiter(nominatim.geocode, min_delay_seconds=RATE_LIMIT_SECONDS)

    geocoders = [
        (photon_rl, "photon"),
        (arcgis_rl, "arcgis"),
        (nominatim_rl, "nominatim_retry"),
    ]

    resolved = 0
    still_failed = 0

    for idx, row in failed.iterrows():
        clean = clean_query_name(row["display_name"])
        queries = get_query_variants(clean, row["district"])

        result = None
        for gc_fn, gc_name in geocoders:
            result = try_geocoder(gc_fn, queries, gc_name)
            if result is not None:
                break

        if result is not None:
            df.at[idx, "lat"] = result["lat"]
            df.at[idx, "lon"] = result["lon"]
            df.at[idx, "geocode_status"] = result["geocode_status"]
            df.at[idx, "geocode_query"] = result["geocode_query"]
            resolved += 1
            print(f"  RESOLVED: {row['display_name']:40s} via {result['geocode_status']:20s} ({result['lat']:.4f}, {result['lon']:.4f})")
        else:
            still_failed += 1
            print(f"  STILL FAILED: {row['display_name']:40s} ({row['district']})")

    print(f"\nRetry results: {resolved} resolved, {still_failed} still failed (was {total_failed})")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Retry geocoding failed census towns using Photon + ArcGIS"
    )
    parser.add_argument("--input", default=GEOCODED_PATH, help="Path to geocoded CSV")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be retried")
    args = parser.parse_args()

    print("Census Town Geocoding — Retry Failed")
    print("=" * 60)
    print(f"Geocoders: Photon (OSM/Elasticsearch) → ArcGIS (Esri) → Nominatim (retry)")

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found. Run geocode_census_towns.py first.")
        sys.exit(1)

    df = pd.read_csv(args.input)
    total = len(df)
    ok = (df.geocode_status == "ok").sum()
    fallback = (df.geocode_status == "fallback").sum()
    failed = (df.geocode_status == "failed").sum()
    print(f"\nCurrent state: {total} towns — {ok} ok, {fallback} fallback, {failed} failed")

    df = retry_failed(df, dry_run=args.dry_run)

    if not args.dry_run:
        df.to_csv(args.input, index=False)
        print(f"\nSaved: {args.input}")

        # Final summary
        ok = df["geocode_status"].str.contains("ok", na=False).sum()
        fallback = (df["geocode_status"] == "fallback").sum()
        failed = (df["geocode_status"] == "failed").sum()
        with_coords = df["lat"].notna().sum()
        print(f"\nFinal state: {total} towns — {with_coords} with coords ({with_coords/total*100:.1f}%), {failed} failed ({failed/total*100:.1f}%)")

        # List remaining failures
        if failed > 0:
            print(f"\nRemaining {failed} failed towns:")
            remaining = df[df["geocode_status"] == "failed"]
            for _, row in remaining.iterrows():
                print(f"  {row['display_name']:45s} ({row['district']})")

    print("\nDone!")


if __name__ == "__main__":
    main()
