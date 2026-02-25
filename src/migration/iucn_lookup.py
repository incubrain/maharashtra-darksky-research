"""
IUCN Red List status lookup via GBIF Species API.

Queries https://api.gbif.org/v1/species/{speciesKey}/iucnRedListCategory
for each unique bird species found in the eBird data. No API key required.

This module is used by scripts/build_iucn_lookup.py but can also be
imported directly for programmatic use.
"""

import os
import sys

import pandas as pd

from src.migration.api_utils import CSVCheckpointer, RateLimiter, make_session
from src.migration.constants import (
    EBIRD_CHUNK_SIZE,
    EBIRD_DATA_DIR,
    GBIF_RATE_LIMIT_PER_SEC,
    GBIF_SPECIES_API,
    IUCN_LOOKUP_CSV,
    MAHARASHTRA_EBIRD_FILE,
    NEIGHBORING_STATES,
)

LOOKUP_FIELDS = [
    "speciesKey", "species", "scientificName",
    "order", "family", "iucn_code", "iucn_category",
]


def collect_unique_species(
    include_neighbors: bool = False,
    data_dir: str = EBIRD_DATA_DIR,
) -> pd.DataFrame:
    """Extract unique (speciesKey, species, scientificName, order, family) tuples.

    Reads eBird TSV files in chunks to stay memory-efficient.

    Parameters
    ----------
    include_neighbors : bool
        If True, also scan neighboring state files for additional species.
    data_dir : str
        Directory containing eBird .tsv.gz files.

    Returns
    -------
    pd.DataFrame
        Deduplicated species with columns: speciesKey, species,
        scientificName, order, family.
    """
    files_to_scan = [os.path.join(data_dir, MAHARASHTRA_EBIRD_FILE)]

    if include_neighbors:
        for state_info in NEIGHBORING_STATES.values():
            path = os.path.join(data_dir, state_info["file"])
            if os.path.isfile(path):
                files_to_scan.append(path)

    usecols = ["speciesKey", "species", "scientificName", "order", "family"]
    all_species = {}

    for filepath in files_to_scan:
        basename = os.path.basename(filepath)
        print(f"  Scanning {basename}...", end="", flush=True)
        count = 0
        for chunk in pd.read_csv(
            filepath, sep="\t", compression="gzip",
            usecols=usecols, chunksize=EBIRD_CHUNK_SIZE,
            dtype={"speciesKey": str},
        ):
            for _, row in chunk.drop_duplicates("speciesKey").iterrows():
                key = str(row["speciesKey"])
                if key and key not in all_species:
                    all_species[key] = {
                        "speciesKey": key,
                        "species": row.get("species", ""),
                        "scientificName": row.get("scientificName", ""),
                        "order": row.get("order", ""),
                        "family": row.get("family", ""),
                    }
            count += len(chunk)
        print(f" {count:,} rows, {len(all_species)} unique species so far")

    df = pd.DataFrame(list(all_species.values()))
    print(f"  Total unique species: {len(df)}")
    return df


def lookup_iucn_status(species_key: str, session, rate_limiter) -> dict:
    """Query GBIF for IUCN Red List status of a single species.

    Parameters
    ----------
    species_key : str
        GBIF speciesKey (integer as string).
    session : requests.Session
        HTTP session with retry configuration.
    rate_limiter : RateLimiter
        Rate limiter instance.

    Returns
    -------
    dict
        {"iucn_code": "CR"|"EN"|...|"NE", "iucn_category": "..."}
    """
    rate_limiter.wait()
    url = f"{GBIF_SPECIES_API}/{species_key}/iucnRedListCategory"

    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "iucn_code": data.get("code", "NE"),
                "iucn_category": data.get("category", "NOT_EVALUATED"),
            }
        elif resp.status_code == 404:
            return {"iucn_code": "NE", "iucn_category": "NOT_EVALUATED"}
        else:
            print(f"    WARN: HTTP {resp.status_code} for speciesKey={species_key}")
            return {"iucn_code": "NE", "iucn_category": "NOT_EVALUATED"}
    except Exception as exc:
        print(f"    ERROR: {exc} for speciesKey={species_key}")
        return {"iucn_code": "NE", "iucn_category": "NOT_EVALUATED"}


def build_iucn_lookup(
    include_neighbors: bool = False,
    data_dir: str = EBIRD_DATA_DIR,
    output_path: str = IUCN_LOOKUP_CSV,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Build or update the IUCN species lookup CSV.

    Idempotent: skips species already present in the output CSV.

    Parameters
    ----------
    include_neighbors : bool
        Also scan neighboring state eBird files.
    data_dir : str
        Directory containing eBird .tsv.gz files.
    output_path : str
        Path for the output CSV.
    dry_run : bool
        If True, collect species but don't query GBIF.

    Returns
    -------
    pd.DataFrame
        Complete lookup table.
    """
    print("Collecting unique species from eBird data...")
    species_df = collect_unique_species(include_neighbors, data_dir)

    # Load already-looked-up species
    checkpointer = CSVCheckpointer(output_path, fieldnames=LOOKUP_FIELDS)
    existing_keys = checkpointer.load_existing(key_column="speciesKey")
    print(f"Already looked up: {len(existing_keys)} species")

    # Filter to new species only
    new_species = species_df[~species_df["speciesKey"].isin(existing_keys)]
    print(f"New species to look up: {len(new_species)}")

    if dry_run:
        print("DRY RUN — not querying GBIF API")
        print(f"  Would look up {len(new_species)} species")
        for _, row in new_species.head(10).iterrows():
            print(f"    {row['speciesKey']}: {row['species']} ({row['order']})")
        if len(new_species) > 10:
            print(f"    ... and {len(new_species) - 10} more")
        return checkpointer.load_existing_df() or pd.DataFrame(columns=LOOKUP_FIELDS)

    if len(new_species) == 0:
        print("All species already looked up!")
        return pd.read_csv(output_path)

    # Set up API session and rate limiter
    session = make_session(user_agent="maharashtra-darksky-research/1.0")
    limiter = RateLimiter(calls_per_second=GBIF_RATE_LIMIT_PER_SEC)

    est_seconds = len(new_species) / GBIF_RATE_LIMIT_PER_SEC
    print(f"Querying GBIF API for {len(new_species)} species "
          f"(~{est_seconds:.0f}s at {GBIF_RATE_LIMIT_PER_SEC} req/s)...")

    counts = {"CR": 0, "EN": 0, "VU": 0, "NT": 0, "LC": 0, "DD": 0, "NE": 0}

    with checkpointer:
        for i, (_, row) in enumerate(new_species.iterrows()):
            result = lookup_iucn_status(str(row["speciesKey"]), session, limiter)

            out_row = {
                "speciesKey": row["speciesKey"],
                "species": row["species"],
                "scientificName": row["scientificName"],
                "order": row["order"],
                "family": row["family"],
                "iucn_code": result["iucn_code"],
                "iucn_category": result["iucn_category"],
            }
            checkpointer.write_row(out_row)

            code = result["iucn_code"]
            counts[code] = counts.get(code, 0) + 1

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(new_species)} looked up...")

    print(f"\nIUCN lookup complete: {len(new_species)} new species")
    print("Distribution:")
    for code in ["CR", "EN", "VU", "NT", "LC", "DD", "NE"]:
        if counts.get(code, 0) > 0:
            print(f"  {code}: {counts[code]}")

    print(f"Saved to: {output_path}")
    return pd.read_csv(output_path)
