"""
Census-based projected growth dataset.

Loads all available census CSVs (1991, 2001, 2011), computes derived
ratios, then projects values for each VIIRS year (2012-2024) using
linear interpolation/extrapolation between the two nearest census
anchor points.

This produces a *timeseries* dataset (one row per district per year)
that can be merged year-matched with VNL data, enabling direct
comparison of census-projected growth against observed ALAN trends.

Design note: the interpolation strategy is deliberately simple (linear)
to start. The module is structured so that switching to a better model
(exponential, logistic) only requires changing ``_interpolate()``.
"""

import os
import time

import numpy as np
import pandas as pd

from src import config
from src.datasets._base import DatasetMeta, DatasetResult
from src.datasets._census_loader import (
    compute_derived_ratios,
    resolve_district_names,
    validate_census,
)
from src.datasets._name_resolver import resolve_names
from src.logging_config import get_pipeline_logger

log = get_pipeline_logger(__name__)

# Census years to load, in chronological order.
# Add new years here as data becomes available.
CENSUS_YEARS = [1991, 2001, 2011]

# Years to project for (VIIRS study period)
PROJECTION_YEARS = list(config.STUDY_YEARS)  # 2012-2024


def get_meta() -> DatasetMeta:
    return DatasetMeta(
        name="census_projected",
        short_label="cproj",
        description="Census-projected urban demographics (linear, 2012-2024)",
        temporal_type="timeseries",
        reference_years=PROJECTION_YEARS,
        entity_col="district",
        source_url="https://censusindia.gov.in/",
        citation=(
            "Projected from Census of India 1991/2001/2011 PCA data. "
            "Linear interpolation/extrapolation between census anchor years."
        ),
    )


def load_and_process(
    data_dir: str,
    entity_col: str = "district",
    vnl_district_names: list[str] | None = None,
) -> tuple[DatasetResult, pd.DataFrame | None]:
    """Load census CSVs, project values for VIIRS years.

    Returns a timeseries DataFrame with columns:
        [district, year, cproj_TOT_P, cproj_literacy_rate, ...]
    """
    meta = get_meta()
    start = time.perf_counter()

    try:
        # ── Load available census CSVs ──────────────────────────────
        anchors = {}  # year -> DataFrame (unprefixed, with derived ratios)
        for year in CENSUS_YEARS:
            csv_path = os.path.join(data_dir, f"census_{year}_pca.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for col in config.CENSUS_COMMON_COLUMNS:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                df = compute_derived_ratios(df)
                anchors[year] = df
                log.info("Loaded census anchor %d: %d districts", year, len(df))

        if len(anchors) < 2:
            elapsed = time.perf_counter() - start
            return DatasetResult(
                dataset_name=meta.name,
                status="error",
                error=f"Need >= 2 census years for projection, found {list(anchors.keys())}",
                timing_seconds=elapsed,
            ), None

        # ── Resolve names before projection ─────────────────────────
        # Use the latest census for name resolution (most districts)
        latest_year = max(anchors.keys())
        if vnl_district_names is not None:
            for year, df in anchors.items():
                dataset_names = df[entity_col].dropna().unique().tolist()
                mapping, _ = resolve_names(vnl_district_names, dataset_names)
                anchors[year][entity_col] = df[entity_col].map(lambda x, m=mapping: m.get(x, x))

        # ── Find common districts and metric columns ────────────────
        all_districts = set()
        for df in anchors.values():
            all_districts.update(df[entity_col].dropna().unique())

        # Districts present in the two most recent census years
        # (for projection we need at least 2 anchor points)
        sorted_years = sorted(anchors.keys())
        metric_cols = [c for c in anchors[sorted_years[-1]].columns if c != entity_col]

        # ── Project each district for each VIIRS year ───────────────
        rows = []
        for district in sorted(all_districts):
            # Build anchor series for this district: {year: {metric: value}}
            district_anchors = {}
            for year in sorted_years:
                df = anchors[year]
                match = df[df[entity_col] == district]
                if not match.empty:
                    district_anchors[year] = match.iloc[0].to_dict()

            if len(district_anchors) < 2:
                # Can't project with only one data point — skip
                continue

            anchor_years = sorted(district_anchors.keys())

            for proj_year in PROJECTION_YEARS:
                row = {entity_col: district, "year": proj_year}
                for metric in metric_cols:
                    row[metric] = _interpolate(
                        proj_year, anchor_years, district_anchors, metric
                    )
                rows.append(row)

        projected = pd.DataFrame(rows)

        # ── Prefix columns ──────────────────────────────────────────
        rename_map = {}
        for col in projected.columns:
            if col not in (entity_col, "year"):
                rename_map[col] = f"{meta.short_label}_{col}"
        projected = projected.rename(columns=rename_map)

        elapsed = time.perf_counter() - start
        return DatasetResult(
            dataset_name=meta.name,
            status="success",
            districts_matched=projected[entity_col].nunique(),
            districts_unmatched=[],
            columns_produced=list(projected.columns),
            rows=len(projected),
            timing_seconds=elapsed,
        ), projected

    except Exception as exc:
        elapsed = time.perf_counter() - start
        log.error("Census projection failed: %s", exc, exc_info=True)
        return DatasetResult(
            dataset_name=meta.name,
            status="error",
            error=str(exc),
            timing_seconds=elapsed,
        ), None


def _interpolate(
    target_year: int,
    anchor_years: list[int],
    district_anchors: dict[int, dict],
    metric: str,
) -> float | None:
    """Linear interpolation/extrapolation from census anchor points.

    Uses the two nearest anchor years that have valid data for this metric.
    For years beyond the last anchor, extrapolates from the last two.

    Future improvement: replace with exponential or logistic growth model
    when more census years (e.g. 1981, 1991) are added.
    """
    # Collect valid (year, value) pairs
    points = []
    for y in anchor_years:
        val = district_anchors[y].get(metric)
        if val is not None and pd.notna(val):
            points.append((y, float(val)))

    if len(points) < 2:
        # Only one anchor — return that constant value
        return points[0][1] if points else None

    # Find the two best anchor points for interpolation:
    # - If target is between two anchors, use those two (interpolation)
    # - If target is beyond all anchors, use the last two (extrapolation)
    if target_year <= points[0][0]:
        y0, v0 = points[0]
        y1, v1 = points[1]
    elif target_year >= points[-1][0]:
        y0, v0 = points[-2]
        y1, v1 = points[-1]
    else:
        # Find bracketing pair
        for i in range(len(points) - 1):
            if points[i][0] <= target_year <= points[i + 1][0]:
                y0, v0 = points[i]
                y1, v1 = points[i + 1]
                break
        else:
            y0, v0 = points[-2]
            y1, v1 = points[-1]

    # Linear interpolation/extrapolation
    if y1 == y0:
        return v0
    slope = (v1 - v0) / (y1 - y0)
    return v0 + slope * (target_year - y0)


def validate(df: pd.DataFrame, entity_col: str = "district") -> list[str]:
    warnings = validate_census(df, get_meta().short_label, entity_col)
    if df is not None and "year" not in df.columns:
        warnings.append("Missing 'year' column for timeseries dataset")
    return warnings
