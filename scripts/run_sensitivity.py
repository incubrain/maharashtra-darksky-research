#!/usr/bin/env python3
"""
Standalone sensitivity analysis runner.

Sweeps the cloud-free coverage threshold parameter and generates
comparison plots showing how different thresholds affect district-level
radiance statistics.

Usage (run from project root):
    python3 scripts/run_sensitivity.py --output-dir ./outputs --year 2024 \
        --shapefile-path ./data/shapefiles/maharashtra_district.geojson
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import geopandas as gpd

from src import config
from src.analysis.sensitivity_analysis import (
    run_cf_threshold_sensitivity,
    plot_sensitivity_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run CF threshold sensitivity analysis"
    )
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--shapefile-path",
                        default="./data/shapefiles/maharashtra_district.geojson")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--thresholds", default="1,3,5,7,10",
                        help="Comma-separated CF threshold values to test")
    args = parser.parse_args()

    gdf = gpd.read_file(args.shapefile_path)
    log.info("Loaded shapefile: %d districts", len(gdf))

    subset_dir = os.path.join(args.output_dir, "subsets", str(args.year))
    if not os.path.isdir(subset_dir):
        log.error("Subset directory not found: %s", subset_dir)
        log.error("Run viirs_process.py first to generate subsets.")
        sys.exit(1)

    thresholds = [int(t.strip()) for t in args.thresholds.split(",")]
    csv_dir = os.path.join(args.output_dir, "csv")
    maps_dir = os.path.join(args.output_dir, "maps")

    log.info("Running sensitivity analysis for year %d", args.year)
    log.info("Thresholds: %s", thresholds)

    results = run_cf_threshold_sensitivity(
        subset_dir=subset_dir,
        gdf=gdf,
        year=args.year,
        param_values=thresholds,
        output_csv=os.path.join(csv_dir, f"sensitivity_cf_{args.year}.csv"),
    )

    if not results.empty:
        plot_sensitivity_results(
            results,
            param_name="cf_threshold",
            output_path=os.path.join(maps_dir, f"sensitivity_cf_{args.year}.png"),
        )
        log.info("Sensitivity analysis complete.")
    else:
        log.warning("No sensitivity results generated.")


if __name__ == "__main__":
    main()
