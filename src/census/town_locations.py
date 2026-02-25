"""
Load geocoded census town locations for the city pipeline.

Reads the geocoded census towns CSV and returns a locations dict
compatible with config.URBAN_CITIES format.
"""

import os
import re

import pandas as pd

from src import config
from src.census.town_loader import normalise_town_name
from src.census.name_resolver import normalize_name, resolve_names
from src.logging_config import get_pipeline_logger

log = get_pipeline_logger(__name__)

GEOCODED_CSV = os.path.join("data", "census", "census_towns_geocoded.csv")


def _title_case_town(norm_name: str) -> str:
    """Convert normalised lowercase name to title case for display."""
    return " ".join(w.capitalize() for w in norm_name.split())


def load_census_town_locations(
    data_dir: str = "data/census",
    exclude_failed: bool = True,
    vnl_district_names: list[str] | None = None,
) -> dict[str, dict]:
    """Load geocoded census towns as a city locations dict.

    Returns dict compatible with config.URBAN_CITIES format:
    {town_name: {"lat": float, "lon": float, "district": str, "population": int}}
    """
    csv_path = os.path.join(data_dir, "census_towns_geocoded.csv")
    if not os.path.exists(csv_path):
        log.warning("Geocoded CSV not found: %s — falling back to URBAN_CITIES", csv_path)
        return dict(config.URBAN_CITIES)

    df = pd.read_csv(csv_path)
    log.info("Loaded %d geocoded towns from %s", len(df), csv_path)

    if exclude_failed:
        before = len(df)
        df = df[df["geocode_status"] != "failed"].copy()
        dropped = before - len(df)
        if dropped > 0:
            log.info("Excluded %d towns with failed geocoding", dropped)

    df = df.dropna(subset=["lat", "lon"])

    if vnl_district_names is not None:
        dataset_districts = df["district"].unique().tolist()
        mapping, unmatched = resolve_names(vnl_district_names, dataset_districts)
        df["district"] = df["district"].map(lambda x: mapping.get(x, x))
        if unmatched:
            log.warning("Unmatched districts in geocoded data: %s", unmatched)

    name_counts = df.groupby("town_name").size()
    duplicated_names = set(name_counts[name_counts > 1].index)

    locations = {}
    for _, row in df.iterrows():
        norm = row["town_name"]
        display = _title_case_town(norm)

        if norm in duplicated_names:
            same_name = df[df["town_name"] == norm]
            if same_name["district"].nunique() > 1:
                display = f"{display} ({row['district']})"

        if display in locations:
            existing_pop = locations[display].get("population", 0)
            new_pop = row.get("TOT_P", 0) if pd.notna(row.get("TOT_P")) else 0
            if new_pop <= existing_pop:
                continue

        loc_entry = {
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "district": row["district"],
        }
        locations[display] = loc_entry

    log.info("Census town locations: %d towns across %d districts",
             len(locations), df["district"].nunique())

    return locations
