"""
iNaturalist data ingestion for Maharashtra bird observations.

Reads iNaturalist CSV exports and normalizes them into the same observation
point format used by the eBird pipeline. Uses GBIF species matching API to
bridge iNaturalist taxon IDs to eBird speciesKeys, enabling the two datasets
to be combined for richer spatial coverage.

iNaturalist export columns expected:
    id, uuid, observed_on, time_observed_at, user_id, user_login, user_name,
    quality_grade, url, description, captive_cultivated, latitude, longitude,
    positional_accuracy, geoprivacy, coordinates_obscured, positioning_device,
    place_state_name, place_country_name, scientific_name, common_name,
    iconic_taxon_name, taxon_id, taxon_family_name, taxon_genus_name
"""

import os

import pandas as pd

from src.migration.api_utils import CSVCheckpointer, RateLimiter, make_session
from src.migration.constants import (
    GBIF_RATE_LIMIT_PER_SEC,
    GBIF_SPECIES_API,
    INATURALIST_DATA_DIR,
    INATURALIST_MATCHED_CSV,
    INATURALIST_RAW_CSV,
)


def load_inaturalist_csv(path: str = INATURALIST_RAW_CSV) -> pd.DataFrame:
    """Load and clean iNaturalist CSV export.

    Filters to research-grade bird observations with valid coordinates.

    Returns DataFrame with columns:
        inat_id, observed_on, lat, lon, scientificName, species,
        inat_taxon_id, taxon_family, taxon_genus, month, year
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"iNaturalist CSV not found: {path}")

    df = pd.read_csv(path, dtype={"taxon_id": str})
    print(f"  Loaded {len(df):,} rows from {path}")

    # Rename to internal format
    df = df.rename(columns={
        "id": "inat_id",
        "latitude": "lat",
        "longitude": "lon",
        "scientific_name": "scientificName",
        "common_name": "species",
        "taxon_id": "inat_taxon_id",
        "taxon_family_name": "taxon_family",
        "taxon_genus_name": "taxon_genus",
    })

    # Drop rows without coordinates or scientific name
    df = df.dropna(subset=["lat", "lon", "scientificName"])

    # Parse date → month/year
    df["observed_on"] = pd.to_datetime(df["observed_on"], errors="coerce")
    df = df.dropna(subset=["observed_on"])
    df["month"] = df["observed_on"].dt.month
    df["year"] = df["observed_on"].dt.year

    # Filter to modern era (1991+) to match eBird modern split
    df = df[df["year"] >= 1991]

    # Keep only relevant columns
    keep = [
        "inat_id", "observed_on", "lat", "lon", "scientificName", "species",
        "inat_taxon_id", "taxon_family", "taxon_genus", "month", "year",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    print(f"  After cleaning: {len(df):,} observations, "
          f"{df['scientificName'].nunique()} unique species")
    return df


def match_species_to_gbif(
    inat_df: pd.DataFrame,
    output_path: str = INATURALIST_MATCHED_CSV,
) -> pd.DataFrame:
    """Match iNaturalist scientific names to GBIF speciesKeys.

    Uses the GBIF Species Match API to find the canonical speciesKey for each
    unique scientific name in the iNaturalist data. This allows merging with
    our eBird-based species classification.

    Saves matches to CSV for idempotent re-runs.

    Returns DataFrame with columns:
        scientificName, speciesKey, gbif_species, gbif_confidence, gbif_status
    """
    unique_names = sorted(inat_df["scientificName"].dropna().unique())
    print(f"  Matching {len(unique_names)} unique species to GBIF...")

    fieldnames = [
        "scientificName", "speciesKey", "gbif_species",
        "gbif_confidence", "gbif_status",
    ]
    cp = CSVCheckpointer(output_path, fieldnames=fieldnames)
    existing = cp.load_existing(key_column="scientificName")

    to_query = [n for n in unique_names if n not in existing]
    print(f"  Already matched: {len(existing)}, to query: {len(to_query)}")

    if len(to_query) == 0:
        return pd.read_csv(output_path, dtype={"speciesKey": str})

    session = make_session()
    limiter = RateLimiter(calls_per_second=GBIF_RATE_LIMIT_PER_SEC)

    with cp:
        for i, name in enumerate(to_query):
            limiter.wait()
            try:
                resp = session.get(
                    f"{GBIF_SPECIES_API}/match",
                    params={"name": name, "kingdom": "Animalia", "strict": False},
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    cp.write_row({
                        "scientificName": name,
                        "speciesKey": str(data.get("speciesKey", "")),
                        "gbif_species": data.get("species", ""),
                        "gbif_confidence": data.get("confidence", 0),
                        "gbif_status": data.get("matchType", "NONE"),
                    })
                else:
                    cp.write_row({
                        "scientificName": name,
                        "speciesKey": "",
                        "gbif_species": "",
                        "gbif_confidence": 0,
                        "gbif_status": f"HTTP_{resp.status_code}",
                    })
            except Exception as e:
                cp.write_row({
                    "scientificName": name,
                    "speciesKey": "",
                    "gbif_species": "",
                    "gbif_confidence": 0,
                    "gbif_status": f"ERROR: {e}",
                })

            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(to_query)} matched...", flush=True)

    print(f"  Done: {len(to_query)} species matched")
    return pd.read_csv(output_path, dtype={"speciesKey": str})


def build_observation_points(
    inat_df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert matched iNaturalist data to observation points format.

    Produces the same schema as eBird observation points:
        speciesKey, month, lat, lon, source

    Only includes species that were successfully matched to a GBIF speciesKey.
    Adds a 'source' column to distinguish from eBird data.
    """
    # Filter matches to successful ones
    valid_matches = matches_df[
        (matches_df["speciesKey"].notna()) &
        (matches_df["speciesKey"] != "") &
        (matches_df["gbif_status"] != "NONE")
    ][["scientificName", "speciesKey"]].drop_duplicates()

    merged = inat_df.merge(valid_matches, on="scientificName", how="inner")

    obs_points = merged[["speciesKey", "month", "lat", "lon"]].copy()
    obs_points["source"] = "inaturalist"

    print(f"  iNaturalist observation points: {len(obs_points):,} rows, "
          f"{obs_points['speciesKey'].nunique()} species")
    return obs_points


def run_inaturalist_pipeline(
    raw_csv_path: str = INATURALIST_RAW_CSV,
) -> pd.DataFrame | None:
    """Run the full iNaturalist ingestion pipeline.

    1. Load and clean CSV
    2. Match species to GBIF speciesKeys
    3. Convert to observation points format

    Returns observation points DataFrame, or None if raw CSV not found.
    """
    if not os.path.isfile(raw_csv_path):
        print(f"  iNaturalist data not found at {raw_csv_path}")
        print("  Place the CSV export there to include iNaturalist observations.")
        return None

    os.makedirs(INATURALIST_DATA_DIR, exist_ok=True)

    print("\n=== iNaturalist: Load & Clean ===")
    inat_df = load_inaturalist_csv(raw_csv_path)

    print("\n=== iNaturalist: GBIF Species Match ===")
    matches_df = match_species_to_gbif(inat_df)

    print("\n=== iNaturalist: Build Observation Points ===")
    obs_points = build_observation_points(inat_df, matches_df)

    return obs_points
