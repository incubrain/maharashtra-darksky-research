#!/usr/bin/env python3
"""
Independent report generation from saved CSV and map data.

Reads analysis CSVs from entity directories and generates PDF reports
without re-running any analysis. Supports --type district|city|site|all.

Usage:
    python3 -m src.outputs.generate_reports --type district [--output-dir ./outputs]
    python3 -m src.outputs.generate_reports --type all
"""

import argparse
import os

import geopandas as gpd
import pandas as pd

from src import config
from src.config import get_entity_dirs
from src.logging_config import get_pipeline_logger

log = get_pipeline_logger(__name__)


def generate_district_reports_standalone(base_dir, shapefile_path):
    """Generate district PDF reports from saved CSVs."""
    from src.pipeline_steps import step_district_reports

    dirs = get_entity_dirs(base_dir, "district")
    csv_dir = dirs["csv"]
    reports_dir = dirs["reports"]

    yearly_path = os.path.join(csv_dir, "districts_yearly_radiance.csv")
    trends_path = os.path.join(csv_dir, "districts_trends.csv")

    if not os.path.exists(yearly_path) or not os.path.exists(trends_path):
        log.info("Skipped district reports: CSVs not found")
        return

    yearly_df = pd.read_csv(yearly_path)
    trends_df = pd.read_csv(trends_path)
    gdf = gpd.read_file(shapefile_path)

    stability_path = os.path.join(csv_dir, "district_stability_metrics.csv")
    stability_df = pd.read_csv(stability_path) if os.path.exists(stability_path) else None

    result, _ = step_district_reports(
        yearly_df=yearly_df,
        trends_df=trends_df,
        stability_df=stability_df,
        gdf=gdf,
        output_dir=base_dir,
        reports_dir=reports_dir,
    )

    if result.ok:
        report_count = len([f for f in os.listdir(reports_dir) if f.endswith(".pdf")])
        log.info("Generated %d district reports in %s", report_count, reports_dir)
    else:
        log.error("District reports failed: %s", result.error)


def generate_city_reports(base_dir):
    """Generate city PDF reports from saved CSVs."""
    from src.site.site_pipeline_steps import step_site_reports

    dirs = get_entity_dirs(base_dir, "city")
    csv_dir = dirs["csv"]
    reports_dir = dirs["reports"]

    yearly_path = os.path.join(csv_dir, "city_yearly_radiance.csv")
    if not os.path.exists(yearly_path):
        log.info("Skipped city reports: CSVs not found")
        return

    yearly_df = pd.read_csv(yearly_path)
    latest_year = yearly_df["year"].max()
    latest_metrics = yearly_df[yearly_df["year"] == latest_year]

    result, _ = step_site_reports(latest_metrics, yearly_df, reports_dir, entity_type="city")

    if result.ok:
        report_count = len([f for f in os.listdir(reports_dir) if f.endswith(".pdf")])
        log.info("Generated %d city reports in %s", report_count, reports_dir)
    else:
        log.error("City reports failed: %s", result.error)


def generate_site_reports_standalone(base_dir):
    """Generate dark-sky site PDF reports from saved CSVs."""
    from src.site.site_pipeline_steps import step_site_reports

    dirs = get_entity_dirs(base_dir, "site")
    csv_dir = dirs["csv"]
    reports_dir = dirs["reports"]

    yearly_path = os.path.join(csv_dir, "site_yearly_radiance.csv")
    if not os.path.exists(yearly_path):
        log.info("Skipped site reports: CSVs not found")
        return

    yearly_df = pd.read_csv(yearly_path)
    latest_year = yearly_df["year"].max()
    latest_metrics = yearly_df[yearly_df["year"] == latest_year]

    result, _ = step_site_reports(latest_metrics, yearly_df, reports_dir, entity_type="site")

    if result.ok:
        report_count = len([f for f in os.listdir(reports_dir) if f.endswith(".pdf")])
        log.info("Generated %d site reports in %s", report_count, reports_dir)
    else:
        log.error("Site reports failed: %s", result.error)


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
            generate_district_reports_standalone(base_dir, args.shapefile_path)
        elif entity_type == "city":
            generate_city_reports(base_dir)
        elif entity_type == "site":
            generate_site_reports_standalone(base_dir)

    log.info("Report generation complete.")


if __name__ == "__main__":
    main()
