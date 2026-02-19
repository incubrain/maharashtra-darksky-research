"""
Dataset aggregator — merge enabled datasets with VNL data.

Determines which datasets are enabled, loads each, and merges with
VNL yearly and trends DataFrames. Handles snapshot (broadcast) and
timeseries (year-matched) merge semantics.
"""

import os
import time

import pandas as pd

from src import config
from src.datasets import DATASET_REGISTRY
from src.datasets._base import DatasetResult
from src.logging_config import get_pipeline_logger

log = get_pipeline_logger(__name__)


def get_enabled_datasets(args) -> list[str]:
    """Determine which datasets are enabled from config + CLI overrides.

    Parameters
    ----------
    args : argparse.Namespace
        Must have ``datasets`` attribute (comma-separated names or "all").

    Returns
    -------
    list[str]
        Sorted list of enabled dataset names.
    """
    # CLI override
    if hasattr(args, "datasets") and args.datasets:
        if args.datasets == "all":
            return sorted(DATASET_REGISTRY.keys())
        names = [n.strip() for n in args.datasets.split(",")]
        valid = [n for n in names if n in DATASET_REGISTRY]
        invalid = [n for n in names if n not in DATASET_REGISTRY]
        if invalid:
            log.warning(
                "Unknown datasets ignored: %s. Available: %s",
                invalid,
                list(DATASET_REGISTRY.keys()),
            )
        return sorted(valid)

    # Fall back to config
    enabled = []
    for name, cfg in config.EXTERNAL_DATASETS.items():
        if cfg.get("enabled", False) and name in DATASET_REGISTRY:
            enabled.append(name)
    return sorted(enabled)


def get_data_dir(dataset_name: str, args) -> str:
    """Resolve the data directory for a dataset.

    Parameters
    ----------
    dataset_name : str
    args : argparse.Namespace
        May have ``census_dir`` for census override.

    Returns
    -------
    str
        Absolute path to the dataset's data directory.
    """
    # CLI override for census datasets
    if dataset_name.startswith("census_") and hasattr(args, "census_dir") and args.census_dir:
        return os.path.abspath(args.census_dir)

    # Config-defined path
    cfg = config.EXTERNAL_DATASETS.get(dataset_name, {})
    data_dir = cfg.get("data_dir", f"../{dataset_name}")
    return os.path.abspath(data_dir)


def load_all_datasets(
    enabled: list[str],
    args,
    vnl_district_names: list[str] | None = None,
) -> dict[str, tuple[DatasetResult, pd.DataFrame | None]]:
    """Load each enabled dataset module.

    Parameters
    ----------
    enabled : list[str]
        Dataset names to load.
    args : argparse.Namespace
    vnl_district_names : list[str], optional
        District names from VNL data for name resolution.

    Returns
    -------
    dict[str, tuple[DatasetResult, pd.DataFrame | None]]
        Maps dataset name -> (result, DataFrame).
    """
    results = {}
    for name in enabled:
        module = DATASET_REGISTRY[name]
        data_dir = get_data_dir(name, args)
        log.info("Loading dataset '%s' from %s", name, data_dir)

        result, df = module.load_and_process(
            data_dir=data_dir,
            vnl_district_names=vnl_district_names,
        )

        if result.ok:
            warnings = module.validate(df)
            if warnings:
                for w in warnings:
                    log.warning("[%s] %s", name, w)
            log.info(
                "Dataset '%s': %d rows, %d columns, %d districts matched",
                name, result.rows, len(result.columns_produced), result.districts_matched,
            )
        else:
            log.error("Dataset '%s' failed: %s", name, result.error)

        results[name] = (result, df)

    return results


def merge_yearly_with_datasets(
    yearly_df: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    entity_col: str = "district",
) -> pd.DataFrame:
    """Left-join VNL yearly data with district-level dataset DataFrames.

    For snapshot datasets (census): broadcast — same census values for every VNL year.
    For timeseries datasets (AQI): year-matched join.

    Town-level datasets (entity_type="town") are skipped here because they
    have multiple rows per district and would create a Cartesian product.
    Use merge_town_datasets() for those.
    """
    merged = yearly_df.copy()

    for name, ds_df in datasets.items():
        if ds_df is None:
            continue

        module = DATASET_REGISTRY[name]
        meta = module.get_meta()

        # Skip town-level datasets — they need separate aggregation
        if meta.entity_type == "town":
            log.debug("Skipping town-level dataset '%s' in district merge", name)
            continue

        before = len(merged)

        if meta.temporal_type == "timeseries" and "year" in ds_df.columns:
            merged = merged.merge(ds_df, on=[entity_col, "year"], how="left")
        else:
            merged = merged.merge(ds_df, on=entity_col, how="left")

        after = len(merged)
        if after > before * 2:
            log.warning(
                "Merge '%s' expanded rows %d -> %d (%.1fx) — possible Cartesian product",
                name, before, after, after / before,
            )

        log.debug("Merged '%s': %d cols -> %d cols", name, len(yearly_df.columns), len(merged.columns))

    return merged


def merge_trends_with_datasets(
    trends_df: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    entity_col: str = "district",
) -> pd.DataFrame:
    """Left-join VNL trends (one row per district) with dataset columns.

    Town-level datasets are skipped (same reason as merge_yearly).
    """
    merged = trends_df.copy()

    for name, ds_df in datasets.items():
        if ds_df is None:
            continue

        module = DATASET_REGISTRY[name]
        meta = module.get_meta()

        if meta.entity_type == "town":
            log.debug("Skipping town-level dataset '%s' in trends merge", name)
            continue

        # Trends are one row per district, drop year if present
        ds_cols = ds_df.copy()
        if "year" in ds_cols.columns:
            ds_cols = ds_cols.drop(columns=["year"]).drop_duplicates(subset=[entity_col])

        merged = merged.merge(ds_cols, on=entity_col, how="left")

    return merged


def get_dataset_suffix(enabled: list[str]) -> str:
    """Generate file suffix from active datasets.

    Examples
    --------
    >>> get_dataset_suffix([])
    ''
    >>> get_dataset_suffix(["census_2011_pca"])
    '_x_c2011'
    >>> get_dataset_suffix(["census_2011_pca", "air_quality_index"])
    '_x_aqi_c2011'
    """
    if not enabled:
        return ""

    labels = []
    for name in sorted(enabled):
        module = DATASET_REGISTRY.get(name)
        if module:
            labels.append(module.get_meta().short_label)
        else:
            labels.append(name[:4])

    return "_x_" + "_".join(sorted(labels))
