#!/usr/bin/env python3
"""
Project census town data year-by-year for VIIRS period (2012-2024).

Reads census town CSVs from data/census/ and produces per-year projected
CSV files at data/census/projected_towns/towns_{year}.csv.

Key features:
- Towns with 2+ census anchor points: linear interpolation/extrapolation
- Towns with only 1 anchor point: held constant (best available estimate)
- Town count grows over time: new towns that appear in later censuses are
  included starting from the VIIRS year that best matches their emergence
- Each output CSV has: district, town_name, TOT_P, No_HH, ..., census_years

Usage:
    python scripts/project_census_towns.py
    python scripts/project_census_towns.py --output-dir data/census/projected_towns
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from src.census.town_loader import normalise_town_name
from src.census.loader import compute_derived_ratios

CENSUS_YEARS = [1991, 2001, 2011]
VIIRS_YEARS = list(config.STUDY_YEARS)  # 2012-2024
DATA_DIR = "data/census"


def load_census_towns(data_dir: str) -> dict[int, pd.DataFrame]:
    """Load all available census town CSVs."""
    dfs = {}
    for year in CENSUS_YEARS:
        path = os.path.join(data_dir, f"census_{year}_towns.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            for col in config.CENSUS_COMMON_COLUMNS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = compute_derived_ratios(df)
            df["norm_name"] = df["entity_name"].apply(normalise_town_name)
            dfs[year] = df
            print(f"  Loaded {len(df)} towns from {year}")
    return dfs


def match_towns(dfs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Match towns across census years by (district, normalised name).

    Returns a master table with one row per unique town, columns:
    - district, norm_name, display_name
    - first_census, census_count
    - For each census year and metric: {metric}_{year}
    """
    records = {}  # (district, norm_name) -> {meta + metrics}

    for year, df in sorted(dfs.items()):
        for _, row in df.iterrows():
            district = row["district"]
            norm = row["norm_name"]
            key = (district, norm)

            if key not in records:
                records[key] = {
                    "district": district,
                    "norm_name": norm,
                    "display_name": row["entity_name"],
                    "census_years": [],
                }

            records[key]["census_years"].append(year)
            # Store metrics keyed by year
            for col in config.CENSUS_COMMON_COLUMNS:
                if col in row.index:
                    records[key][f"{col}_{year}"] = row[col]

    # Convert to DataFrame
    rows = []
    for (district, norm), rec in records.items():
        rec["first_census"] = min(rec["census_years"])
        rec["census_count"] = len(rec["census_years"])
        rec["census_years_str"] = ",".join(str(y) for y in rec["census_years"])
        del rec["census_years"]
        rows.append(rec)

    master = pd.DataFrame(rows)
    n_multi = (master["census_count"] >= 2).sum()
    n_single = (master["census_count"] == 1).sum()
    print(f"\n  Town matching: {n_multi} with 2+ anchors, {n_single} with 1 anchor")
    print(f"  Total unique towns: {len(master)}")
    return master


def interpolate(target_year: int, points: list[tuple[int, float]]) -> float | None:
    """Linear interpolation/extrapolation between anchor points."""
    points = [(y, v) for y, v in points if v is not None and not np.isnan(v)]
    if not points:
        return None
    if len(points) == 1:
        return points[0][1]

    points.sort()

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
    projected = v0 + slope * (target_year - y0)
    # Population can't go negative
    if projected < 0:
        projected = 0.0
    return projected


def estimate_town_emergence_year(first_census: int) -> int:
    """Estimate the year a town first qualifies as urban.

    Towns newly appearing in a census likely crossed the urban threshold
    sometime during the intercensal period. We estimate the midpoint.

    - Appeared in 1991: existed before VIIRS era -> include from 2012
    - Appeared in 2001: emerged ~1996 -> include from 2012
    - Appeared in 2011: emerged ~2006 -> include from 2012

    Since all three census years predate the VIIRS era (2012+), all
    towns are included for the full VIIRS period. The growing count
    is captured by the population projections showing growth from near-zero
    for newly urbanised towns.
    """
    # All census years (1991, 2001, 2011) predate VIIRS (2012+)
    # so all towns are included from 2012 onwards.
    return 2012


def project_towns(master: pd.DataFrame, output_dir: str):
    """Project each town's metrics for each VIIRS year and save per-year CSVs."""
    os.makedirs(output_dir, exist_ok=True)
    metrics = config.CENSUS_COMMON_COLUMNS

    # Also create a combined long-format CSV
    all_rows = []

    for viirs_year in VIIRS_YEARS:
        year_rows = []

        for _, town in master.iterrows():
            # All census years predate VIIRS, so all towns are included
            row = {
                "district": town["district"],
                "town_name": town["norm_name"],
                "display_name": town["display_name"],
                "year": viirs_year,
                "census_count": town["census_count"],
                "first_census": town["first_census"],
            }

            for metric in metrics:
                # Collect anchor points for this metric
                points = []
                for cy in CENSUS_YEARS:
                    col = f"{metric}_{cy}"
                    if col in town.index and pd.notna(town[col]):
                        points.append((cy, float(town[col])))

                row[metric] = interpolate(viirs_year, points)

            year_rows.append(row)

        year_df = pd.DataFrame(year_rows)
        # Compute derived ratios on the projected data
        year_df = compute_derived_ratios(year_df)

        # Save per-year CSV
        out_path = os.path.join(output_dir, f"towns_{viirs_year}.csv")
        year_df.to_csv(out_path, index=False)
        all_rows.extend(year_rows)

        print(f"  {viirs_year}: {len(year_df)} towns projected")

    # Save combined long-format CSV
    combined = pd.DataFrame(all_rows)
    combined = compute_derived_ratios(combined)
    combined_path = os.path.join(output_dir, "towns_projected_all.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\n  Combined: {len(combined)} rows -> {combined_path}")

    return combined


def print_summary(master: pd.DataFrame, combined: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("PROJECTION SUMMARY")
    print("=" * 60)

    print(f"\nUnique towns across all censuses: {len(master)}")
    for cy in CENSUS_YEARS:
        n = (master["census_years_str"].str.contains(str(cy))).sum()
        print(f"  {cy}: {n} towns")

    print(f"\nAnchoring:")
    print(f"  3 census points: {(master['census_count'] == 3).sum()} towns")
    print(f"  2 census points: {(master['census_count'] == 2).sum()} towns")
    print(f"  1 census point:  {(master['census_count'] == 1).sum()} towns (held constant)")

    print(f"\nProjected VIIRS years: {VIIRS_YEARS[0]}-{VIIRS_YEARS[-1]}")
    for year in VIIRS_YEARS:
        year_data = combined[combined["year"] == year]
        total_pop = year_data["TOT_P"].sum()
        print(f"  {year}: {len(year_data)} towns, projected population: {total_pop:,.0f}")

    print(f"\nTowns per district (2011 census):")
    d2011 = master[master["census_years_str"].str.contains("2011")]
    dist_counts = d2011.groupby("district").size().sort_values(ascending=False)
    for district, count in dist_counts.head(10).items():
        print(f"  {district}: {count}")
    if len(dist_counts) > 10:
        print(f"  ... and {len(dist_counts) - 10} more districts")


def main():
    parser = argparse.ArgumentParser(
        description="Project census town data for VIIRS years (2012-2024)"
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR,
        help="Directory containing census_*_towns.csv files"
    )
    parser.add_argument(
        "--output-dir", default=os.path.join(DATA_DIR, "projected_towns"),
        help="Output directory for projected CSVs"
    )
    args = parser.parse_args()

    print("Census Town Projection")
    print("=" * 60)

    print("\n1. Loading census town data...")
    dfs = load_census_towns(args.data_dir)
    if len(dfs) < 1:
        print("ERROR: No census town CSVs found!")
        sys.exit(1)

    print("\n2. Matching towns across census years...")
    master = match_towns(dfs)

    # Save master matching table
    master_path = os.path.join(args.output_dir, "town_master.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    master.to_csv(master_path, index=False)
    print(f"  Saved town master: {master_path}")

    print("\n3. Projecting for VIIRS years...")
    combined = project_towns(master, args.output_dir)

    print_summary(master, combined)
    print("\nDone!")


if __name__ == "__main__":
    main()
