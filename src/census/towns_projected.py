"""
Census-based projected growth dataset — town level.

Loads town-level data from all available census years (1991, 2001, 2011),
matches towns across censuses within the same district using normalised
names, then projects values for each VIIRS year (2012-2024) using linear
interpolation/extrapolation.
"""

import os
import time

import numpy as np
import pandas as pd
from difflib import SequenceMatcher

from src import config
from src.datasets._base import DatasetMeta, DatasetResult
from src.census.loader import compute_derived_ratios, resolve_district_names
from src.census.town_loader import normalise_town_name
from src.logging_config import get_pipeline_logger

log = get_pipeline_logger(__name__)

CENSUS_YEARS = [1991, 2001, 2011]
PROJECTION_YEARS = list(config.STUDY_YEARS)  # 2012-2024


def get_meta() -> DatasetMeta:
    return DatasetMeta(
        name="census_towns_projected",
        short_label="ctproj",
        description="Census-projected town demographics (linear, 2012-2024)",
        temporal_type="timeseries",
        entity_type="town",
        reference_years=PROJECTION_YEARS,
        entity_col="district",
        source_url="https://censusindia.gov.in/",
        citation=(
            "Projected from Census of India 1991/2001/2011 town-level PCA data. "
            "Linear interpolation/extrapolation between census anchor years."
        ),
    )


def _match_towns_across_years(
    town_dfs: dict[int, pd.DataFrame],
) -> dict[str, dict[int, dict]]:
    """Match towns across census years within the same district."""
    anchors: dict[tuple[str, str], dict[int, dict]] = {}
    original_names: dict[tuple[str, str], str] = {}

    for year, df in sorted(town_dfs.items()):
        for _, row in df.iterrows():
            district = row["district"]
            town_name = row["entity_name"]
            norm = normalise_town_name(town_name)
            key = (district, norm)

            if key not in anchors:
                anchors[key] = {}
                original_names[key] = town_name
            anchors[key][year] = {col: row[col] for col in df.columns
                                  if col not in ("district", "district_code", "level",
                                                 "tru", "entity_name", "entity_code")}

    by_district: dict[str, list[tuple[str, str]]] = {}
    for key in anchors:
        by_district.setdefault(key[0], []).append(key)

    for district, keys in by_district.items():
        single_year_keys = [k for k in keys if len(anchors[k]) == 1]
        multi_year_keys = [k for k in keys if len(anchors[k]) >= 2]

        for sk in single_year_keys:
            sk_year = list(anchors[sk].keys())[0]
            best_match = None
            best_score = 0.0
            for mk in multi_year_keys:
                if sk_year in anchors[mk]:
                    continue
                score = SequenceMatcher(None, sk[1], mk[1]).ratio()
                if score > best_score:
                    best_score = score
                    best_match = mk
            if best_match and best_score >= 0.85:
                anchors[best_match][sk_year] = anchors[sk][sk_year]
                del anchors[sk]

    result: dict[str, dict[int, dict]] = {}
    for (district, norm), year_data in anchors.items():
        town_key = f"{district}|{norm}"
        result[town_key] = year_data

    n_multi = sum(1 for v in result.values() if len(v) >= 2)
    n_single = sum(1 for v in result.values() if len(v) == 1)
    log.info("Town matching: %d with 2+ anchors, %d with 1 anchor (skipped for projection)",
             n_multi, n_single)

    return result


def _interpolate(
    target_year: int,
    points: list[tuple[int, float]],
) -> float | None:
    """Linear interpolation/extrapolation between anchor points."""
    if len(points) < 2:
        return points[0][1] if points else None

    if target_year <= points[0][0]:
        y0, v0 = points[0]
        y1, v1 = points[1]
    elif target_year >= points[-1][0]:
        y0, v0 = points[-2]
        y1, v1 = points[-1]
    else:
        for i in range(len(points) - 1):
            if points[i][0] <= target_year <= points[i + 1][0]:
                y0, v0 = points[i]
                y1, v1 = points[i + 1]
                break
        else:
            y0, v0 = points[-2]
            y1, v1 = points[-1]

    if y1 == y0:
        return v0
    slope = (v1 - v0) / (y1 - y0)
    return v0 + slope * (target_year - y0)


def load_and_process(
    data_dir: str,
    entity_col: str = "district",
    vnl_district_names: list[str] | None = None,
) -> tuple[DatasetResult, pd.DataFrame | None]:
    """Load town-level census data, match across years, project to VIIRS period."""
    meta = get_meta()
    start = time.perf_counter()

    try:
        town_dfs = {}
        for year in CENSUS_YEARS:
            csv_path = os.path.join(data_dir, f"census_{year}_towns.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for col in config.CENSUS_COMMON_COLUMNS:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                df = compute_derived_ratios(df)
                town_dfs[year] = df
                log.info("Loaded %d towns for %d", len(df), year)

        if len(town_dfs) < 2:
            elapsed = time.perf_counter() - start
            return DatasetResult(
                dataset_name=meta.name,
                status="error",
                error=f"Need >= 2 census years, found {list(town_dfs.keys())}",
                timing_seconds=elapsed,
            ), None

        if vnl_district_names is not None:
            for year, df in town_dfs.items():
                df, _ = resolve_district_names(df, vnl_district_names, entity_col)
                town_dfs[year] = df

        matched = _match_towns_across_years(town_dfs)

        latest_year = max(town_dfs.keys())
        metric_cols = [c for c in town_dfs[latest_year].columns
                       if c not in (entity_col, "district_code", "level", "tru",
                                    "entity_name", "entity_code")]

        rows = []
        for town_key, year_data in matched.items():
            if len(year_data) < 2:
                continue

            district = town_key.split("|")[0]
            norm_name = town_key.split("|")[1]
            anchor_years = sorted(year_data.keys())

            for proj_year in PROJECTION_YEARS:
                row = {
                    entity_col: district,
                    "year": proj_year,
                    "town_name": norm_name,
                }
                for metric in metric_cols:
                    pts = []
                    for y in anchor_years:
                        val = year_data[y].get(metric)
                        if val is not None and pd.notna(val):
                            pts.append((y, float(val)))
                    row[metric] = _interpolate(proj_year, pts) if len(pts) >= 2 else None
                rows.append(row)

        projected = pd.DataFrame(rows)

        prefix = meta.short_label
        rename_map = {}
        for col in projected.columns:
            if col not in (entity_col, "year"):
                rename_map[col] = f"{prefix}_{col}"
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
        log.error("Town projection failed: %s", exc, exc_info=True)
        return DatasetResult(
            dataset_name=meta.name,
            status="error",
            error=str(exc),
            timing_seconds=elapsed,
        ), None


def validate(df: pd.DataFrame, entity_col: str = "district") -> list[str]:
    warnings = []
    if df is None:
        return ["DataFrame is None"]
    if "year" not in df.columns:
        warnings.append("Missing 'year' column for timeseries dataset")
    prefix = f"{get_meta().short_label}_"
    town_col = f"{prefix}town_name"
    if town_col not in df.columns:
        warnings.append(f"Missing town name column '{town_col}'")
    return warnings
