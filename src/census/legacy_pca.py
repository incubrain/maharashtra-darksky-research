"""
Census 2011 Primary Census Abstract dataset module (legacy).

Loads district-level census data from the Census of India 2011 PCA
Excel files (DDW_PCA27*_2011_MDDS*.xlsx), computes derived ratios,
and normalizes district names for merging with VNL data.
"""

import glob
import os
import time

import pandas as pd

from src import config
from src.datasets._base import DatasetMeta, DatasetResult
from src.census.name_resolver import resolve_names
from src.logging_config import get_pipeline_logger

log = get_pipeline_logger(__name__)


def get_meta() -> DatasetMeta:
    """Return metadata about the Census 2011 PCA dataset."""
    return DatasetMeta(
        name="census_2011_pca",
        short_label="c2011",
        description="Census of India 2011 Primary Census Abstract",
        temporal_type="snapshot",
        reference_years=[2011],
        entity_col="district",
        source_url="https://censusindia.gov.in/",
        citation=(
            "Office of the Registrar General & Census Commissioner, India. "
            "Census of India 2011: Primary Census Abstract."
        ),
    )


def load_and_process(
    data_dir: str,
    entity_col: str = "district",
    vnl_district_names: list[str] | None = None,
) -> tuple[DatasetResult, pd.DataFrame | None]:
    """Load Census 2011 PCA data, aggregate to district level, compute derived metrics."""
    meta = get_meta()
    start = time.perf_counter()

    try:
        patterns = [
            os.path.join(data_dir, "DDW_PCA27*_2011*.xlsx"),
            os.path.join(data_dir, "DDW_PCA27*_2011*.xls"),
            os.path.join(data_dir, "**", "DDW_PCA27*_2011*.xlsx"),
            os.path.join(data_dir, "**", "DDW_PCA27*_2011*.xls"),
        ]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern, recursive=True))
        files = sorted(set(files))

        if not files:
            csv_path = os.path.join(data_dir, "census_2011_pca.csv")
            if os.path.exists(csv_path):
                return _load_from_csv(csv_path, entity_col, vnl_district_names, meta, start)

            elapsed = time.perf_counter() - start
            return DatasetResult(
                dataset_name=meta.name,
                status="error",
                error=f"No Census 2011 PCA files found in {data_dir}",
                timing_seconds=elapsed,
            ), None

        log.info("Found %d Census 2011 PCA files in %s", len(files), data_dir)

        all_districts = []
        for filepath in files:
            df_file = _load_single_file(filepath)
            if df_file is not None:
                all_districts.append(df_file)

        if not all_districts:
            elapsed = time.perf_counter() - start
            return DatasetResult(
                dataset_name=meta.name,
                status="error",
                error="No valid district data extracted from Census files",
                timing_seconds=elapsed,
            ), None

        df = pd.concat(all_districts, ignore_index=True)
        log.info("Loaded %d district records from Census 2011 PCA", len(df))

        df = _compute_derived_ratios(df)
        df = _prefix_columns(df, meta.short_label, entity_col)

        unmatched = []
        if vnl_district_names is not None:
            df, unmatched = _resolve_district_names(df, vnl_district_names, entity_col)

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
        log.error("Census 2011 PCA loading failed: %s", exc, exc_info=True)
        return DatasetResult(
            dataset_name=meta.name,
            status="error",
            error=str(exc),
            timing_seconds=elapsed,
        ), None


def _load_single_file(filepath: str) -> pd.DataFrame | None:
    """Load a single Census PCA Excel file and extract district-level row."""
    try:
        df = pd.read_excel(filepath, engine="openpyxl")

        if "Level" in df.columns and "TRU" in df.columns:
            mask = (df["Level"] == "DISTRICT") & (df["TRU"] == "Total")
            df = df[mask]
        elif "Level" in df.columns:
            df = df[df["Level"] == "DISTRICT"]

        if df.empty:
            log.warning("No district-level data in %s", filepath)
            return None

        name_col = None
        for candidate in ["Name", "name", "DISTRICT_NAME", "District_Name"]:
            if candidate in df.columns:
                name_col = candidate
                break

        if name_col is None:
            log.warning("No name column found in %s", filepath)
            return None

        keep_cols = [name_col]
        for col in config.CENSUS_COMMON_COLUMNS:
            if col in df.columns:
                keep_cols.append(col)

        result = df[keep_cols].copy()
        result = result.rename(columns={name_col: "district"})

        for col in result.columns:
            if col != "district":
                result[col] = pd.to_numeric(result[col], errors="coerce")

        return result

    except Exception as exc:
        log.warning("Failed to read %s: %s", filepath, exc)
        return None


def _load_from_csv(
    csv_path: str,
    entity_col: str,
    vnl_district_names: list[str] | None,
    meta: DatasetMeta,
    start: float,
) -> tuple[DatasetResult, pd.DataFrame | None]:
    """Load census data from a pre-processed CSV file."""
    log.info("Loading Census 2011 PCA from CSV: %s", csv_path)
    df = pd.read_csv(csv_path)

    if entity_col not in df.columns:
        for candidate in ["Name", "name", "DISTRICT_NAME", "District_Name", "District"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: entity_col})
                break

    keep_cols = [entity_col]
    for col in config.CENSUS_COMMON_COLUMNS:
        if col in df.columns:
            keep_cols.append(col)
    df = df[[c for c in keep_cols if c in df.columns]]

    df = _compute_derived_ratios(df)
    df = _prefix_columns(df, meta.short_label, entity_col)

    unmatched = []
    if vnl_district_names is not None:
        df, unmatched = _resolve_district_names(df, vnl_district_names, entity_col)

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


def _compute_derived_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived ratio columns from raw census columns."""
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


def _prefix_columns(df: pd.DataFrame, short_label: str, entity_col: str) -> pd.DataFrame:
    """Prefix all metric columns with the short_label."""
    rename_map = {}
    for col in df.columns:
        if col != entity_col:
            rename_map[col] = f"{short_label}_{col}"
    return df.rename(columns=rename_map)


def _resolve_district_names(
    df: pd.DataFrame,
    vnl_names: list[str],
    entity_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Resolve dataset district names to VNL names."""
    dataset_names = df[entity_col].dropna().unique().tolist()
    mapping, unmatched = resolve_names(vnl_names, dataset_names)
    df[entity_col] = df[entity_col].map(lambda x: mapping.get(x, x))
    return df, unmatched


def validate(df: pd.DataFrame, entity_col: str = "district") -> list[str]:
    """Validate the processed Census 2011 PCA DataFrame."""
    warnings = []
    if df is None:
        return ["DataFrame is None"]
    if entity_col not in df.columns:
        warnings.append(f"Missing entity column '{entity_col}'")
        return warnings

    n_districts = df[entity_col].nunique()
    if n_districts < 30:
        warnings.append(f"Only {n_districts} districts found (expected ~36 for Maharashtra)")

    meta = get_meta()
    expected_prefix = f"{meta.short_label}_"
    metric_cols = [c for c in df.columns if c.startswith(expected_prefix)]
    if len(metric_cols) < 5:
        warnings.append(f"Only {len(metric_cols)} metric columns found (expected >= 5)")

    for col in metric_cols:
        nan_pct = df[col].isna().mean() * 100
        if nan_pct > 20:
            warnings.append(f"Column '{col}' has {nan_pct:.1f}% NaN values")

    return warnings
