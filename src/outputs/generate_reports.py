#!/usr/bin/env python3
"""
Independent report generation from saved CSV and map data.

Reads analysis CSVs from entity directories and generates PDF reports
without re-running any analysis. Supports --type district|city|site|all.

Usage:
    python3 -m src.generate_reports --type district [--output-dir ./outputs]
    python3 -m src.generate_reports --type all
"""

import argparse
import logging
import os
import sys

import geopandas as gpd
import pandas as pd

from src import config
from src.config import get_entity_dirs
from src.logging_config import StepTimer, get_pipeline_logger
from src.pipeline_types import StepResult

log = get_pipeline_logger(__name__)


def generate_district_reports_standalone(base_dir, shapefile_path):
    """Generate district PDF reports from saved CSVs.

    Parameters
    ----------
    base_dir : str
        Run-level output directory containing district/ subdirectory.
    shapefile_path : str
        Path to Maharashtra district GeoJSON.

    Returns
    -------
    StepResult
    """
    dirs = get_entity_dirs(base_dir, "district")
    csv_dir = dirs["csv"]
    reports_dir = dirs["reports"]
    os.makedirs(reports_dir, exist_ok=True)

    error_tb = None
    with StepTimer() as timer:
        try:
            yearly_path = os.path.join(csv_dir, "districts_yearly_radiance.csv")
            trends_path = os.path.join(csv_dir, "districts_trends.csv")

            if not os.path.exists(yearly_path) or not os.path.exists(trends_path):
                return StepResult(
                    step_name="generate_district_reports",
                    status="skipped",
                    output_summary={"reason": "CSVs not found"},
                    timing_seconds=0,
                )

            yearly_df = pd.read_csv(yearly_path)
            trends_df = pd.read_csv(trends_path)
            gdf = gpd.read_file(shapefile_path)

            # Load stability metrics if available
            stability_path = os.path.join(csv_dir, "district_stability_metrics.csv")
            stability_df = None
            if os.path.exists(stability_path):
                stability_df = pd.read_csv(stability_path)

            from src.outputs.district_reports import generate_all_district_reports
            generate_all_district_reports(
                yearly_df=yearly_df,
                trends_df=trends_df,
                stability_df=stability_df,
                gdf=gdf,
                output_dir=reports_dir,
            )

            report_count = len([
                f for f in os.listdir(reports_dir) if f.endswith(".pdf")
            ])
            log.info("Generated %d district reports in %s", report_count, reports_dir)

        except Exception:
            import traceback
            error_tb = traceback.format_exc()

    if error_tb:
        log.error("generate_district_reports failed:\n%s", error_tb)
        return StepResult(
            step_name="generate_district_reports",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        )

    return StepResult(
        step_name="generate_district_reports",
        status="success",
        output_summary={"reports_dir": reports_dir},
        timing_seconds=timer.elapsed,
    )


def generate_city_reports(base_dir):
    """Generate city PDF reports from saved CSVs.

    Parameters
    ----------
    base_dir : str
        Run-level output directory containing city/ subdirectory.

    Returns
    -------
    StepResult
    """
    dirs = get_entity_dirs(base_dir, "city")
    csv_dir = dirs["csv"]
    reports_dir = dirs["reports"]
    os.makedirs(reports_dir, exist_ok=True)

    error_tb = None
    with StepTimer() as timer:
        try:
            yearly_path = os.path.join(csv_dir, "city_yearly_radiance.csv")
            if not os.path.exists(yearly_path):
                return StepResult(
                    step_name="generate_city_reports",
                    status="skipped",
                    output_summary={"reason": "CSVs not found"},
                    timing_seconds=0,
                )

            yearly_df = pd.read_csv(yearly_path)
            latest_year = yearly_df["year"].max()
            latest_metrics = yearly_df[yearly_df["year"] == latest_year]

            from src.outputs.site_reports import generate_all_site_reports
            generate_all_site_reports(
                all_site_data=latest_metrics,
                yearly_df=yearly_df,
                output_dir=reports_dir,
            )

            report_count = len([
                f for f in os.listdir(reports_dir) if f.endswith(".pdf")
            ])
            log.info("Generated %d city reports in %s", report_count, reports_dir)

        except Exception:
            import traceback
            error_tb = traceback.format_exc()

    if error_tb:
        log.error("generate_city_reports failed:\n%s", error_tb)
        return StepResult(
            step_name="generate_city_reports",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        )

    return StepResult(
        step_name="generate_city_reports",
        status="success",
        output_summary={"reports_dir": reports_dir},
        timing_seconds=timer.elapsed,
    )


def generate_site_reports_standalone(base_dir):
    """Generate dark-sky site PDF reports from saved CSVs.

    Parameters
    ----------
    base_dir : str
        Run-level output directory containing site/ subdirectory.

    Returns
    -------
    StepResult
    """
    dirs = get_entity_dirs(base_dir, "site")
    csv_dir = dirs["csv"]
    reports_dir = dirs["reports"]
    os.makedirs(reports_dir, exist_ok=True)

    error_tb = None
    with StepTimer() as timer:
        try:
            yearly_path = os.path.join(csv_dir, "site_yearly_radiance.csv")
            if not os.path.exists(yearly_path):
                return StepResult(
                    step_name="generate_site_reports",
                    status="skipped",
                    output_summary={"reason": "CSVs not found"},
                    timing_seconds=0,
                )

            yearly_df = pd.read_csv(yearly_path)
            latest_year = yearly_df["year"].max()
            latest_metrics = yearly_df[yearly_df["year"] == latest_year]

            from src.outputs.site_reports import generate_all_site_reports
            generate_all_site_reports(
                all_site_data=latest_metrics,
                yearly_df=yearly_df,
                output_dir=reports_dir,
            )

            report_count = len([
                f for f in os.listdir(reports_dir) if f.endswith(".pdf")
            ])
            log.info("Generated %d site reports in %s", report_count, reports_dir)

        except Exception:
            import traceback
            error_tb = traceback.format_exc()

    if error_tb:
        log.error("generate_site_reports failed:\n%s", error_tb)
        return StepResult(
            step_name="generate_site_reports",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        )

    return StepResult(
        step_name="generate_site_reports",
        status="success",
        output_summary={"reports_dir": reports_dir},
        timing_seconds=timer.elapsed,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PDF reports from saved analysis CSVs (no re-computation)"
    )
    parser.add_argument(
        "--type",
        choices=["district", "city", "site", "all"],
        default="all",
        help="Entity type to generate reports for",
    )
    parser.add_argument(
        "--output-dir",
        default=config.DEFAULT_OUTPUT_DIR,
        help="Base output directory",
    )
    parser.add_argument(
        "--shapefile-path",
        default=config.DEFAULT_SHAPEFILE_PATH,
        help="Path to Maharashtra district boundaries (GeoJSON)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve to latest run dir
    base_dir = args.output_dir
    latest_dir = os.path.join(base_dir, "latest")
    if os.path.isdir(latest_dir):
        base_dir = latest_dir
        log.info("Using latest run directory: %s", base_dir)

    entity_types = (
        ["district", "city", "site"] if args.type == "all" else [args.type]
    )

    for entity_type in entity_types:
        log.info("Generating %s reports...", entity_type)
        if entity_type == "district":
            result = generate_district_reports_standalone(base_dir, args.shapefile_path)
        elif entity_type == "city":
            result = generate_city_reports(base_dir)
        elif entity_type == "site":
            result = generate_site_reports_standalone(base_dir)

        if result.status == "skipped":
            log.info("Skipped %s reports: %s", entity_type, result.output_summary)
        elif not result.ok:
            log.error("Failed %s reports: %s", entity_type, result.error)
        else:
            log.info(
                "Generated %s reports in %.1fs",
                entity_type,
                result.timing_seconds,
            )

    log.info("Report generation complete.")


if __name__ == "__main__":
    main()
