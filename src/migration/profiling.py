"""
Migration profiling: monthly species observation profiles and migratory
vs. resident classification from eBird data.

Processes eBird data in memory-efficient chunks to produce:
1. Monthly observation profiles (species × month aggregates)
2. Species migration classification (migratory vs. resident vs. passage)
3. Neighboring state context for entry/exit corridor detection
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd

from src.migration.constants import (
    EBIRD_CHUNK_SIZE,
    EBIRD_DATA_DIR,
    EBIRD_USECOLS,
    IUCN_LOOKUP_CSV,
    KONKAN_COAST_LON,
    MAHARASHTRA_EBIRD_FILE,
    MAX_OBS_PER_SPECIES_MONTH,
    MIN_MONTHS_PRESENT,
    MIN_OBS_PER_MONTH,
    NEIGHBORING_STATES,
    NEIGHBOR_PROFILES_CSV,
    OBSERVATION_POINTS_CSV,
    RESIDENT_MONTHS_THRESHOLD,
    SPECIES_CLASSIFICATION_CSV,
    SPECIES_PROFILES_CSV,
)


def _build_monthly_profiles(filepath: str, species_filter: set | None = None) -> dict:
    """Read an eBird TSV.gz in chunks and aggregate monthly species profiles.

    Parameters
    ----------
    filepath : str
        Path to a .tsv.gz eBird file.
    species_filter : set or None
        If provided, only aggregate these speciesKeys. If None, aggregate all.

    Returns
    -------
    tuple (profiles, effort)
        profiles: dict of (speciesKey, month) → {obs_count, lat_sum, lon_sum,
                  individual_sum, lat_sq_sum, lon_sq_sum}
        effort: dict of month → total observation count
    """
    profiles = defaultdict(lambda: {
        "obs_count": 0, "lat_sum": 0.0, "lon_sum": 0.0,
        "individual_sum": 0, "lat_sq_sum": 0.0, "lon_sq_sum": 0.0,
    })
    effort = defaultdict(int)

    usecols = ["speciesKey", "decimalLatitude", "decimalLongitude",
               "month", "individualCount"]
    total_rows = 0

    for chunk in pd.read_csv(
        filepath, sep="\t", compression="gzip",
        usecols=usecols, chunksize=EBIRD_CHUNK_SIZE,
        dtype={"speciesKey": str},
    ):
        chunk = chunk.dropna(subset=["decimalLatitude", "decimalLongitude", "month"])
        chunk["month"] = chunk["month"].astype(int)
        chunk["speciesKey"] = chunk["speciesKey"].astype(str)

        if species_filter is not None:
            chunk = chunk[chunk["speciesKey"].isin(species_filter)]

        for _, row in chunk.iterrows():
            sk = row["speciesKey"]
            m = row["month"]
            lat = row["decimalLatitude"]
            lon = row["decimalLongitude"]
            ind = row["individualCount"] if pd.notna(row["individualCount"]) else 1

            key = (sk, m)
            p = profiles[key]
            p["obs_count"] += 1
            p["lat_sum"] += lat
            p["lon_sum"] += lon
            p["individual_sum"] += int(ind)
            p["lat_sq_sum"] += lat * lat
            p["lon_sq_sum"] += lon * lon

            effort[m] += 1

        total_rows += len(chunk)
        if total_rows % 500_000 == 0:
            print(f"    {total_rows:,} rows processed...", flush=True)

    print(f"    Total: {total_rows:,} rows, {len(profiles)} species-month combos")
    return dict(profiles), dict(effort)


def build_maharashtra_profiles(data_dir: str = EBIRD_DATA_DIR) -> pd.DataFrame:
    """Build monthly species profiles from Maharashtra eBird data.

    Returns DataFrame with columns:
        speciesKey, month, obs_count, norm_frequency, mean_lat, mean_lon,
        individual_sum
    """
    filepath = os.path.join(data_dir, MAHARASHTRA_EBIRD_FILE)
    print(f"  Building profiles from {MAHARASHTRA_EBIRD_FILE}...")

    profiles, effort = _build_monthly_profiles(filepath)

    rows = []
    for (sk, m), p in profiles.items():
        n = p["obs_count"]
        if n < MIN_OBS_PER_MONTH:
            continue
        rows.append({
            "speciesKey": sk,
            "month": m,
            "obs_count": n,
            "norm_frequency": n / effort.get(m, 1),
            "mean_lat": p["lat_sum"] / n,
            "mean_lon": p["lon_sum"] / n,
            "individual_sum": p["individual_sum"],
        })

    df = pd.DataFrame(rows)
    print(f"  Profiles: {len(df)} rows for {df['speciesKey'].nunique()} species")
    return df


def classify_species(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Classify species as resident, migratory, or passage from monthly profiles.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        Output of build_maharashtra_profiles().

    Returns
    -------
    pd.DataFrame
        One row per species with classification and timing.
    """
    results = []

    for sk, grp in profiles_df.groupby("speciesKey"):
        months_present = len(grp)
        month_obs = grp.set_index("month")["norm_frequency"]

        # Seasonality: coefficient of variation of monthly normalized frequency
        freq_values = month_obs.values
        seasonality_cv = float(np.std(freq_values) / np.mean(freq_values)) if np.mean(freq_values) > 0 else 0

        # Centroid shift: max distance between any two monthly centroids
        lats = grp["mean_lat"].values
        lons = grp["mean_lon"].values
        lat_range = lats.max() - lats.min() if len(lats) > 1 else 0
        lon_range = lons.max() - lons.min() if len(lons) > 1 else 0
        centroid_shift = float(np.sqrt(lat_range**2 + lon_range**2))

        # Peak month (highest observation count)
        peak_month = int(grp.loc[grp["obs_count"].idxmax(), "month"])

        # Classification
        if months_present >= RESIDENT_MONTHS_THRESHOLD:
            migration_class = "resident"
        elif months_present >= MIN_MONTHS_PRESENT:
            migration_class = "migratory"
        else:
            migration_class = "passage"

        # For migratory/passage: estimate arrival and departure months
        if migration_class in ("migratory", "passage"):
            present_months = sorted(grp["month"].tolist())
            arrival_month = present_months[0]
            departure_month = present_months[-1]

            # Check for wrap-around (e.g., Nov-Mar → arrival=11, departure=3)
            # If gap between consecutive months > 3, assume wrap-around
            gaps = [present_months[i+1] - present_months[i]
                    for i in range(len(present_months)-1)]
            if gaps and max(gaps) > 3:
                # Find the largest gap — arrival is month after gap
                gap_idx = gaps.index(max(gaps))
                arrival_month = present_months[gap_idx + 1]
                departure_month = present_months[gap_idx]
        else:
            arrival_month = None
            departure_month = None

        # Is the species primarily on the Konkan coast?
        mean_lon = grp["mean_lon"].mean()
        is_coastal = mean_lon < KONKAN_COAST_LON

        results.append({
            "speciesKey": sk,
            "months_present": months_present,
            "seasonality_cv": round(seasonality_cv, 4),
            "centroid_shift_deg": round(centroid_shift, 4),
            "peak_month": peak_month,
            "migration_class": migration_class,
            "arrival_month": arrival_month,
            "departure_month": departure_month,
            "is_coastal": is_coastal,
        })

    df = pd.DataFrame(results)
    print(f"  Classification: {len(df)} species")
    for cls in ["resident", "migratory", "passage"]:
        n = (df["migration_class"] == cls).sum()
        print(f"    {cls}: {n}")
    return df


def build_neighbor_profiles(
    migratory_keys: set,
    data_dir: str = EBIRD_DATA_DIR,
) -> pd.DataFrame:
    """Build monthly profiles for migratory species in neighboring states.

    Processes one state at a time to stay memory-efficient.

    Parameters
    ----------
    migratory_keys : set
        Set of speciesKey strings to track (migratory + passage species).
    data_dir : str
        Directory containing eBird .tsv.gz files.

    Returns
    -------
    pd.DataFrame
        Columns: speciesKey, state, month, obs_count, mean_lat, mean_lon,
                 individual_sum
    """
    all_rows = []

    for state_name, state_info in NEIGHBORING_STATES.items():
        filepath = os.path.join(data_dir, state_info["file"])
        if not os.path.isfile(filepath):
            print(f"  Skipping {state_name} (file not found: {state_info['file']})")
            continue

        print(f"  Processing {state_name}...")
        profiles, _ = _build_monthly_profiles(filepath, species_filter=migratory_keys)

        for (sk, m), p in profiles.items():
            n = p["obs_count"]
            if n < MIN_OBS_PER_MONTH:
                continue
            all_rows.append({
                "speciesKey": sk,
                "state": state_name,
                "direction": state_info["direction"],
                "month": m,
                "obs_count": n,
                "mean_lat": p["lat_sum"] / n,
                "mean_lon": p["lon_sum"] / n,
                "individual_sum": p["individual_sum"],
            })

    df = pd.DataFrame(all_rows)
    print(f"  Neighbor profiles: {len(df)} rows across "
          f"{df['state'].nunique() if len(df) else 0} states")
    return df


def enrich_classification(
    classification_df: pd.DataFrame,
    iucn_path: str = IUCN_LOOKUP_CSV,
) -> pd.DataFrame:
    """Join species classification with IUCN lookup and eBird taxonomy.

    Parameters
    ----------
    classification_df : pd.DataFrame
        Output of classify_species().
    iucn_path : str
        Path to IUCN lookup CSV.

    Returns
    -------
    pd.DataFrame
        Classification enriched with species, scientificName, order, family,
        iucn_code, iucn_category columns.
    """
    if not os.path.isfile(iucn_path):
        print(f"  WARNING: IUCN lookup not found at {iucn_path}")
        return classification_df

    iucn = pd.read_csv(iucn_path, dtype={"speciesKey": str})
    merged = classification_df.merge(iucn, on="speciesKey", how="left")

    # Fill missing IUCN codes
    merged["iucn_code"] = merged["iucn_code"].fillna("NE")
    merged["iucn_category"] = merged["iucn_category"].fillna("NOT_EVALUATED")

    return merged


def extract_observation_points(
    data_dir: str = EBIRD_DATA_DIR,
    max_per_species_month: int = MAX_OBS_PER_SPECIES_MONTH,
) -> pd.DataFrame:
    """Extract individual observation lat/lon points for KDE heatmaps.

    Instead of just species centroids, this preserves the spatial distribution
    of observations. Samples up to `max_per_species_month` observations per
    species per month to keep the dataset manageable.

    Returns DataFrame with columns:
        speciesKey, month, lat, lon
    """
    filepath = os.path.join(data_dir, MAHARASHTRA_EBIRD_FILE)
    usecols = ["speciesKey", "decimalLatitude", "decimalLongitude", "month"]

    print(f"  Extracting observation points from {MAHARASHTRA_EBIRD_FILE}...")

    # Collect all observations grouped by (speciesKey, month)
    # Use reservoir sampling to cap at max_per_species_month
    import random
    random.seed(42)

    # Accumulate chunks into a single dataframe of (speciesKey, month, lat, lon)
    chunks = []
    total_rows = 0
    for chunk in pd.read_csv(
        filepath, sep="\t", compression="gzip",
        usecols=usecols, chunksize=EBIRD_CHUNK_SIZE,
        dtype={"speciesKey": str},
    ):
        chunk = chunk.dropna(subset=["decimalLatitude", "decimalLongitude", "month"])
        chunk["month"] = chunk["month"].astype(int)
        chunk = chunk.rename(columns={
            "decimalLatitude": "lat", "decimalLongitude": "lon",
        })
        chunks.append(chunk[["speciesKey", "month", "lat", "lon"]])
        total_rows += len(chunk)
        if total_rows % 2_000_000 == 0:
            print(f"    {total_rows:,} rows read...", flush=True)

    print(f"    Total: {total_rows:,} rows read")

    # Concatenate and sample
    all_obs = pd.concat(chunks, ignore_index=True)
    del chunks  # free memory

    print(f"    Sampling to max {max_per_species_month} per species-month...")
    sampled_parts = []
    for _, grp in all_obs.groupby(["speciesKey", "month"]):
        if len(grp) <= max_per_species_month:
            sampled_parts.append(grp)
        else:
            sampled_parts.append(grp.sample(n=max_per_species_month, random_state=42))
    sampled = pd.concat(sampled_parts, ignore_index=True)
    del all_obs, sampled_parts  # free memory

    print(f"  eBird observation points: {len(sampled):,} rows "
          f"({sampled['speciesKey'].nunique()} species)")

    # Merge iNaturalist observations if available
    try:
        from src.migration.inaturalist import run_inaturalist_pipeline
        inat_obs = run_inaturalist_pipeline()
        if inat_obs is not None and len(inat_obs) > 0:
            # Add source column to eBird data for tracking
            sampled["source"] = "ebird"
            # Keep only columns in common
            common_cols = ["speciesKey", "month", "lat", "lon", "source"]
            inat_obs = inat_obs[[c for c in common_cols if c in inat_obs.columns]]
            sampled = pd.concat([sampled, inat_obs], ignore_index=True)
            print(f"  Combined total: {len(sampled):,} rows "
                  f"({sampled['speciesKey'].nunique()} species)")
    except Exception as e:
        print(f"  iNaturalist merge skipped: {e}")

    return sampled


def run_profiling_pipeline(
    data_dir: str = EBIRD_DATA_DIR,
    skip_neighbors: bool = False,
):
    """Run the full profiling pipeline: profiles → classification → neighbors.

    Saves output CSVs to the paths defined in constants.
    """
    # Step A: Maharashtra monthly profiles
    print("\n=== Step A: Maharashtra Monthly Profiles ===")
    profiles_df = build_maharashtra_profiles(data_dir)
    profiles_df.to_csv(SPECIES_PROFILES_CSV, index=False)
    print(f"  Saved: {SPECIES_PROFILES_CSV}")

    # Step B: Species classification
    print("\n=== Step B: Species Classification ===")
    classification_df = classify_species(profiles_df)

    # Enrich with IUCN data
    classification_df = enrich_classification(classification_df)
    classification_df.to_csv(SPECIES_CLASSIFICATION_CSV, index=False)
    print(f"  Saved: {SPECIES_CLASSIFICATION_CSV}")

    # Step D: Observation-level points for KDE heatmaps
    print("\n=== Step D: Observation Points for Heatmaps ===")
    obs_df = extract_observation_points(data_dir)
    obs_df.to_csv(OBSERVATION_POINTS_CSV, index=False, compression="gzip")
    print(f"  Saved: {OBSERVATION_POINTS_CSV}")

    # Step C: Neighboring state context
    if not skip_neighbors:
        print("\n=== Step C: Neighboring State Profiles ===")
        migratory_keys = set(
            classification_df[
                classification_df["migration_class"].isin(["migratory", "passage"])
            ]["speciesKey"].astype(str)
        )
        print(f"  Tracking {len(migratory_keys)} migratory/passage species across neighbors")

        neighbor_df = build_neighbor_profiles(migratory_keys, data_dir)
        if len(neighbor_df) > 0:
            neighbor_df.to_csv(NEIGHBOR_PROFILES_CSV, index=False)
            print(f"  Saved: {NEIGHBOR_PROFILES_CSV}")
    else:
        print("\n=== Step C: Skipped (--skip-neighbors) ===")

    return profiles_df, classification_df
