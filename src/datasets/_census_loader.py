"""
Shared loader for normalized Census PCA CSV files.

All three censuses (1991, 2001, 2011) are pre-extracted into
a common schema by scripts/extract_census_csvs.py, producing
data/census/census_{year}_pca.csv with identical column layout.

This module provides a single load_census_csv() that any year's
adapter can call, plus shared helpers for derived ratios and
name resolution.
"""

import os
import time

import pandas as pd

from src import config
from src.datasets._base import DatasetMeta, DatasetResult
from src.datasets._name_resolver import resolve_names
from src.logging_config import get_pipeline_logger

log = get_pipeline_logger(__name__)


def load_census_csv(
    data_dir: str,
    year: int,
    meta: DatasetMeta,
    vnl_district_names: list[str] | None = None,
    entity_col: str = "district",
) -> tuple[DatasetResult, pd.DataFrame | None]:
    """Load a pre-extracted census CSV and produce a prefixed DataFrame.

    Parameters
    ----------
    data_dir : str
        Directory containing ``census_{year}_pca.csv``.
    year : int
        Census year (1991, 2001, or 2011).
    meta : DatasetMeta
        Metadata for the dataset (provides short_label for column prefix).
    vnl_district_names : list[str], optional
        VNL district names for fuzzy name resolution.
    entity_col : str
        Column name for the district entity.

    Returns
    -------
    tuple[DatasetResult, pd.DataFrame | None]
    """
    start = time.perf_counter()

    csv_path = os.path.join(data_dir, f"census_{year}_pca.csv")
    if not os.path.exists(csv_path):
        elapsed = time.perf_counter() - start
        return DatasetResult(
            dataset_name=meta.name,
            status="error",
            error=f"CSV not found: {csv_path}. Run scripts/extract_census_csvs.py first.",
            timing_seconds=elapsed,
        ), None

    try:
        df = pd.read_csv(csv_path)
        log.info("Loaded %d rows from %s", len(df), csv_path)

        # Ensure numeric
        for col in config.CENSUS_COMMON_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Compute derived ratios
        df = compute_derived_ratios(df)

        # Prefix metric columns
        df = prefix_columns(df, meta.short_label, entity_col)

        # Name resolution
        unmatched = []
        if vnl_district_names is not None:
            df, unmatched = resolve_district_names(df, vnl_district_names, entity_col)

        elapsed = time.perf_counter() - start
        return DatasetResult(
            dataset_name=meta.name,
            status="success",
            districts_matched=len(df),
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


def compute_derived_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived ratio columns from the common census columns."""
    for ratio_name, (numerator_expr, denominator) in config.CENSUS_COMMON_DERIVED_RATIOS.items():
        try:
            if "+" in numerator_expr:
                parts = [p.strip() for p in numerator_expr.split("+")]
                missing = [p for p in parts if p not in df.columns]
                if missing or denominator not in df.columns:
                    continue
                num = sum(df[p] for p in parts)
            else:
                if numerator_expr not in df.columns or denominator not in df.columns:
                    continue
                num = df[numerator_expr]
            denom = df[denominator]
            df[ratio_name] = num / denom.replace(0, float("nan"))
        except Exception as exc:
            log.warning("Failed to compute ratio '%s': %s", ratio_name, exc)
    return df


def prefix_columns(df: pd.DataFrame, short_label: str, entity_col: str) -> pd.DataFrame:
    """Prefix all metric columns with the short_label."""
    rename_map = {}
    for col in df.columns:
        if col != entity_col:
            rename_map[col] = f"{short_label}_{col}"
    return df.rename(columns=rename_map)


def resolve_district_names(
    df: pd.DataFrame,
    vnl_names: list[str],
    entity_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Resolve dataset district names to VNL names."""
    dataset_names = df[entity_col].dropna().unique().tolist()
    mapping, unmatched = resolve_names(vnl_names, dataset_names)
    df[entity_col] = df[entity_col].map(lambda x: mapping.get(x, x))
    return df, unmatched


def validate_census(df: pd.DataFrame, short_label: str, entity_col: str = "district") -> list[str]:
    """Validate a processed census DataFrame. Returns list of warning strings."""
    warnings = []
    if df is None:
        return ["DataFrame is None"]
    if entity_col not in df.columns:
        warnings.append(f"Missing entity column '{entity_col}'")
        return warnings

    n_districts = df[entity_col].nunique()
    if n_districts < 20:
        warnings.append(f"Only {n_districts} districts (expected >= 29)")

    prefix = f"{short_label}_"
    metric_cols = [c for c in df.columns if c.startswith(prefix)]
    if len(metric_cols) < 5:
        warnings.append(f"Only {len(metric_cols)} metric columns (expected >= 5)")

    for col in metric_cols:
        nan_pct = df[col].isna().mean() * 100
        if nan_pct > 20:
            warnings.append(f"Column '{col}' has {nan_pct:.1f}% NaN values")

    return warnings
