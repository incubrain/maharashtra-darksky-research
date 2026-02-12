#!/usr/bin/env python3
"""
Independent VIIRS data preprocessing pipeline.

Downloads shapefiles, unpacks .gz VIIRS composites, subsets to Maharashtra
extent, and validates subsets. No analysis — just data preparation.

Both viirs_process.py (district analysis) and site_analysis.py (site analysis)
consume pre-existing subsets produced by this module.

Usage (run from project root):
    python3 -m src.preprocess [--viirs-dir ./viirs] [--output-dir ./outputs]
                              [--years 2012-2024] [--download-shapefiles]
"""

import argparse
import logging
import os
import sys

import geopandas as gpd

from src import config
from src.logging_config import StepTimer, get_pipeline_logger
from src.pipeline_types import StepResult

log = get_pipeline_logger(__name__)


def preprocess_years(years, viirs_dir, gdf, output_dir, cf_threshold=None):
    """Unpack and subset VIIRS data for the given years.

    For each year, finds .gz files, unpacks them, clips to Maharashtra,
    and stores subsets in output_dir/subsets/{year}/. Global TIFs are
    deleted after subsetting to save disk space.

    Parameters
    ----------
    years : list[int]
        Years to process.
    viirs_dir : str
        Root directory containing year folders with .gz files.
    gdf : gpd.GeoDataFrame
        Maharashtra district boundaries for clipping.
    output_dir : str
        Base output directory (subsets go to output_dir/subsets/).
    cf_threshold : int, optional
        Cloud-free threshold (for manifest only).

    Returns
    -------
    dict
        Mapping of year -> dict of layer_name -> subset_path.
    """
    from src.viirs_process import (
        find_gz_files,
        identify_layers,
        unpack_subset_cleanup,
        update_manifest,
    )

    if cf_threshold is None:
        cf_threshold = config.CF_COVERAGE_THRESHOLD

    all_subsets = {}
    for year in years:
        year_dir = os.path.join(viirs_dir, str(year))
        if not os.path.isdir(year_dir):
            log.warning("Year directory not found: %s — skipping", year_dir)
            continue

        log.info("=" * 60)
        log.info("Preprocessing year %d", year)
        log.info("=" * 60)

        gz_files = find_gz_files(year_dir)
        if not gz_files:
            log.error("No .gz files found for %d", year)
            continue

        gz_layers = identify_layers(gz_files)
        log.info("Found layers: %s", list(gz_layers.keys()))

        subset_layers = {}
        for layer_name, gz_path in gz_layers.items():
            result = unpack_subset_cleanup(
                gz_path, layer_name, year, gdf, output_dir
            )
            if result:
                subset_layers[layer_name] = result

        if subset_layers:
            all_subsets[year] = subset_layers
            update_manifest(year, output_dir, cf_threshold)
            log.info(
                "Year %d: %d layers subset to Maharashtra",
                year,
                len(subset_layers),
            )
        else:
            log.warning("Year %d: no subsets produced", year)

    return all_subsets


def validate_subsets(output_dir, years):
    """Check that subsets exist for the requested years.

    Parameters
    ----------
    output_dir : str
        Base output directory.
    years : list[int]
        Years to validate.

    Returns
    -------
    dict
        year -> list of existing subset files.
    """
    found = {}
    for year in years:
        subset_dir = os.path.join(output_dir, "subsets", str(year))
        if not os.path.isdir(subset_dir):
            continue
        files = [
            f
            for f in os.listdir(subset_dir)
            if f.endswith(".tif") and f.startswith("maharashtra_")
        ]
        if files:
            found[year] = sorted(files)
    return found


def step_preprocess(
    years, viirs_dir, shapefile_path, output_dir, cf_threshold=None,
    download_shapefiles_flag=False,
):
    """Run the full preprocessing pipeline as a single step.

    Parameters
    ----------
    years : list[int]
    viirs_dir : str
    shapefile_path : str
    output_dir : str
    cf_threshold : int, optional
    download_shapefiles_flag : bool

    Returns
    -------
    tuple[StepResult, dict | None]
        StepResult and dict of year -> subset_layers.
    """
    from src.viirs_process import download_shapefiles

    result_data = None
    error_tb = None

    with StepTimer() as timer:
        try:
            # Download shapefiles if needed
            if download_shapefiles_flag or not os.path.exists(shapefile_path):
                shapefile_path = download_shapefiles()

            gdf = gpd.read_file(shapefile_path)
            log.info("Loaded boundaries: %d districts", len(gdf))

            if len(gdf) != config.EXPECTED_DISTRICT_COUNT:
                log.warning(
                    "Expected %d districts but found %d in %s",
                    config.EXPECTED_DISTRICT_COUNT,
                    len(gdf),
                    shapefile_path,
                )

            result_data = preprocess_years(
                years, viirs_dir, gdf, output_dir, cf_threshold
            )
        except Exception:
            import traceback
            error_tb = traceback.format_exc()

    if error_tb:
        log.error("Preprocessing failed:\n%s", error_tb)
        return StepResult(
            step_name="preprocess",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    years_done = list(result_data.keys()) if result_data else []
    total_subsets = sum(len(v) for v in result_data.values()) if result_data else 0

    return StepResult(
        step_name="preprocess",
        status="success",
        input_summary={
            "years_requested": years,
            "viirs_dir": viirs_dir,
        },
        output_summary={
            "years_processed": years_done,
            "total_subsets": total_subsets,
        },
        timing_seconds=timer.elapsed,
    ), result_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="VIIRS data preprocessing: download, unpack, subset to Maharashtra"
    )
    parser.add_argument(
        "--viirs-dir",
        default=config.DEFAULT_VIIRS_DIR,
        help="Root directory containing year folders with .gz files",
    )
    parser.add_argument(
        "--shapefile-path",
        default=config.DEFAULT_SHAPEFILE_PATH,
        help="Path to Maharashtra district boundaries (GeoJSON)",
    )
    parser.add_argument(
        "--output-dir",
        default=config.DEFAULT_OUTPUT_DIR,
        help="Output directory for subsets",
    )
    parser.add_argument(
        "--cf-threshold",
        type=int,
        default=config.CF_COVERAGE_THRESHOLD,
        help="Cloud-free coverage threshold (for manifest)",
    )
    parser.add_argument(
        "--years",
        default="2012-2024",
        help="Year range (e.g., '2012-2024') or comma-separated years",
    )
    parser.add_argument(
        "--download-shapefiles",
        action="store_true",
        help="Download Maharashtra shapefiles if not present",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("VIIRS Data Preprocessing")
    log.info("Configuration:")
    log.info("  VIIRS dir:    %s", args.viirs_dir)
    log.info("  Shapefile:    %s", args.shapefile_path)
    log.info("  Output dir:   %s", args.output_dir)
    log.info("  Years:        %s", args.years)

    # Parse year range
    if "-" in args.years:
        start, end = args.years.split("-")
        years = list(range(int(start), int(end) + 1))
    else:
        years = [int(y) for y in args.years.split(",")]

    # Resolve output_dir: use latest symlink if it exists
    output_dir = args.output_dir
    latest_dir = os.path.join(output_dir, "latest")
    if os.path.isdir(latest_dir):
        output_dir = latest_dir
        log.info("  Using latest run dir: %s", output_dir)

    result, subsets = step_preprocess(
        years=years,
        viirs_dir=args.viirs_dir,
        shapefile_path=args.shapefile_path,
        output_dir=output_dir,
        cf_threshold=args.cf_threshold,
        download_shapefiles_flag=args.download_shapefiles,
    )

    if result.ok:
        log.info("Preprocessing complete in %.1fs", result.timing_seconds)
        existing = validate_subsets(output_dir, years)
        log.info("Subsets available for %d years: %s", len(existing), list(existing.keys()))
    else:
        log.error("Preprocessing failed: %s", result.error)
        sys.exit(1)


if __name__ == "__main__":
    main()
