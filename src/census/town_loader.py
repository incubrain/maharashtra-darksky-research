"""
Shared loader for town-level Census PCA CSV files.

Reads ``data/census/census_{year}_towns.csv`` produced by
``scripts/extract_census_csvs.py``. Each row is a single town with the
12 common demographic columns.
"""

import os
import re
import time

import pandas as pd

from src import config
from src.datasets._base import DatasetMeta, DatasetResult
from src.census.loader import (
    compute_derived_ratios,
    resolve_district_names,
)
from src.logging_config import get_pipeline_logger

log = get_pipeline_logger(__name__)


def load_census_towns_csv(
    data_dir: str,
    year: int,
    meta: DatasetMeta,
    vnl_district_names: list[str] | None = None,
    entity_col: str = "district",
) -> tuple[DatasetResult, pd.DataFrame | None]:
    """Load a pre-extracted town-level census CSV.

    Returns a DataFrame with one row per town, columns prefixed with
    the dataset's short_label.
    """
    start = time.perf_counter()

    csv_path = os.path.join(data_dir, f"census_{year}_towns.csv")
    if not os.path.exists(csv_path):
        elapsed = time.perf_counter() - start
        return DatasetResult(
            dataset_name=meta.name,
            status="error",
            error=f"Towns CSV not found: {csv_path}. Run scripts/extract_census_csvs.py first.",
            timing_seconds=elapsed,
        ), None

    try:
        df = pd.read_csv(csv_path)
        log.info("Loaded %d town rows from %s", len(df), csv_path)

        # Keep town identity columns
        df["town_name"] = df["entity_name"]
        df["town_code"] = df["entity_code"].astype(str)

        # Ensure numeric on metric columns
        for col in config.CENSUS_COMMON_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Compute derived ratios per town
        df = compute_derived_ratios(df)

        # Resolve district names to VNL names
        unmatched = []
        if vnl_district_names is not None:
            df, unmatched = resolve_district_names(df, vnl_district_names, entity_col)

        # Prefix metric + town identity columns
        prefix = meta.short_label
        rename_map = {}
        for col in df.columns:
            if col not in (entity_col, "level", "tru", "entity_name", "entity_code"):
                rename_map[col] = f"{prefix}_{col}"
        df = df.rename(columns=rename_map)

        # Drop raw schema columns (keep prefixed versions)
        drop_cols = [c for c in ("level", "tru", "entity_name", "entity_code") if c in df.columns]
        df = df.drop(columns=drop_cols)

        elapsed = time.perf_counter() - start
        return DatasetResult(
            dataset_name=meta.name,
            status="success",
            districts_matched=df[entity_col].nunique(),
            districts_unmatched=unmatched,
            columns_produced=list(df.columns),
            rows=len(df),
            timing_seconds=elapsed,
        ), df

    except Exception as exc:
        elapsed = time.perf_counter() - start
        log.error("Failed loading %s: %s", csv_path, exc, exc_info=True)
        return DatasetResult(
            dataset_name=meta.name,
            status="error",
            error=str(exc),
            timing_seconds=elapsed,
        ), None


def normalise_town_name(name: str) -> str:
    """Normalise a town name for cross-census matching.

    Strips municipal suffixes like (M Corp.), (M), (M Cl), (CT), (CB),
    (M.Corp.), (Cantt.), (R), etc. and lowercases for comparison.
    """
    name = name.strip()
    name = re.sub(
        r"\s*\("
        r"(?:M\.?\s*Corp\.?|M\s*Cl\.?|M\.?|CT|CB|R|NP|Cantt\.?|"
        r"Nagar\s*Panchayat|Census\s*Town|Cantonment\s*Board)"
        r"\)\s*\.?\s*$",
        "", name, flags=re.IGNORECASE,
    )
    name = re.sub(r"\s*\(Part\)\s*$", "", name, flags=re.IGNORECASE)
    name = name.strip("* ").lower()
    return name


def validate_census_towns(
    df: pd.DataFrame,
    short_label: str,
    entity_col: str = "district",
) -> list[str]:
    """Validate a processed town-level census DataFrame."""
    warnings = []
    if df is None:
        return ["DataFrame is None"]
    if entity_col not in df.columns:
        warnings.append(f"Missing entity column '{entity_col}'")
        return warnings

    prefix = f"{short_label}_"
    town_name_col = f"{prefix}town_name"
    if town_name_col not in df.columns:
        warnings.append(f"Missing town name column '{town_name_col}'")

    n_towns = len(df)
    if n_towns < 50:
        warnings.append(f"Only {n_towns} towns (expected >= 50 for Maharashtra)")

    return warnings
