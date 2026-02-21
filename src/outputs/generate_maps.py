#!/usr/bin/env python3
"""
Independent map generation from saved CSV data.

Reads analysis CSVs from entity directories and generates visualizations
without re-running any analysis. Supports --type district|city|site|all.

Usage:
    python3 -m src.outputs.generate_maps --type district [--output-dir ./outputs]
    python3 -m src.outputs.generate_maps --type all
"""

import argparse
import os

import geopandas as gpd
import pandas as pd

from src import config
from src.config import get_entity_dirs
from src.logging_config import get_pipeline_logger

log = get_pipeline_logger(__name__)


def generate_district_maps(base_dir, shapefile_path):
    """Generate all district-level maps from saved CSVs.

    Delegates to the same step functions used by the pipeline,
    ensuring no logic duplication.
    """
    from src.pipeline_steps import (
        step_generate_basic_maps,
        step_statewide_visualizations,
        step_graduated_classification,
        step_animation_frames,
        step_per_district_radiance_maps,
    )

    dirs = get_entity_dirs(base_dir, "district")
    csv_dir = dirs["csv"]
    maps_dir = dirs["maps"]
    os.makedirs(maps_dir, exist_ok=True)

    yearly_path = os.path.join(csv_dir, "districts_yearly_radiance.csv")
    trends_path = os.path.join(csv_dir, "districts_trends.csv")

    if not os.path.exists(yearly_path) or not os.path.exists(trends_path):
        log.info("Skipped district maps: CSVs not found")
        return

    yearly_df = pd.read_csv(yearly_path)
    trends_df = pd.read_csv(trends_path)
    gdf = gpd.read_file(shapefile_path)

    # Quality data (optional)
    quality_path = os.path.join(csv_dir, "quality_all_years.csv")
    quality_df = pd.read_csv(quality_path) if os.path.exists(quality_path) else None

    # Basic maps
    step_generate_basic_maps(gdf, trends_df, yearly_df, os.path.join(base_dir, "district"), maps_dir)

    # Statewide visualizations
    step_statewide_visualizations(yearly_df, trends_df, gdf, quality_df, base_dir, maps_dir)

    # Graduated classification
    grad_path = os.path.join(csv_dir, "graduated_classification.csv")
    if os.path.exists(grad_path):
        step_graduated_classification(yearly_df, csv_dir, maps_dir)

    # Animation frames + per-district maps
    available_years = sorted(yearly_df["year"].unique().astype(int).tolist())
    step_animation_frames(available_years, base_dir, gdf, maps_dir)

    latest_year = int(yearly_df["year"].max())
    step_per_district_radiance_maps(base_dir, latest_year, gdf, maps_dir)

    log.info("District maps generated in %s", maps_dir)


def generate_city_maps(base_dir, shapefile_path):
    """Generate city-level maps from saved CSVs."""
    from src.site.site_pipeline_steps import step_site_maps

    dirs = get_entity_dirs(base_dir, "city")
    csv_dir = dirs["csv"]

    yearly_path = os.path.join(csv_dir, "city_yearly_radiance.csv")
    if not os.path.exists(yearly_path):
        log.info("Skipped city maps: CSVs not found")
        return

    yearly_df = pd.read_csv(yearly_path)
    district_gdf = gpd.read_file(shapefile_path)

    from src.site.site_analysis import build_site_geodataframe
    gdf_sites = build_site_geodataframe(entity_type="city")
    latest_year = int(yearly_df["year"].max())

    step_site_maps(gdf_sites, yearly_df, district_gdf, base_dir, latest_year, dirs["maps"])
    log.info("City maps generated in %s", dirs["maps"])


def generate_site_maps_standalone(base_dir, shapefile_path):
    """Generate dark-sky site maps from saved CSVs."""
    from src.site.site_pipeline_steps import step_site_maps

    dirs = get_entity_dirs(base_dir, "site")
    csv_dir = dirs["csv"]

    yearly_path = os.path.join(csv_dir, "site_yearly_radiance.csv")
    if not os.path.exists(yearly_path):
        log.info("Skipped site maps: CSVs not found")
        return

    yearly_df = pd.read_csv(yearly_path)
    district_gdf = gpd.read_file(shapefile_path)

    from src.site.site_analysis import build_site_geodataframe
    gdf_sites = build_site_geodataframe(entity_type="site")
    latest_year = int(yearly_df["year"].max())

    step_site_maps(gdf_sites, yearly_df, district_gdf, base_dir, latest_year, dirs["maps"])

    # Sky brightness distribution
    sky_path = os.path.join(csv_dir, f"sky_brightness_{latest_year}.csv")
    if os.path.exists(sky_path):
        from src.analysis.sky_brightness_model import plot_sky_brightness_distribution
        sky_df = pd.read_csv(sky_path)
        plot_sky_brightness_distribution(
            sky_df,
            output_path=os.path.join(dirs["maps"], "sky_brightness_distribution.png"),
        )

    log.info("Site maps generated in %s", dirs["maps"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate maps from saved analysis CSVs (no re-computation)"
    )
    parser.add_argument(
        "--type",
        choices=["district", "city", "site", "all"],
        default="all",
        help="Entity type to generate maps for",
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

    generators = {
        "district": generate_district_maps,
        "city": generate_city_maps,
        "site": generate_site_maps_standalone,
    }

    for entity_type in entity_types:
        log.info("Generating %s maps...", entity_type)
        generators[entity_type](base_dir, args.shapefile_path)

    log.info("Map generation complete.")


if __name__ == "__main__":
    main()
