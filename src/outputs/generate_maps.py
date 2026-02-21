#!/usr/bin/env python3
"""
Independent map generation from saved CSV data.

Reads analysis CSVs from entity directories and generates visualizations
without re-running any analysis. Supports --type district|city|site|all.

Usage:
    python3 -m src.generate_maps --type district [--output-dir ./outputs]
    python3 -m src.generate_maps --type all
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


def generate_district_maps(base_dir, shapefile_path):
    """Generate all district-level maps from saved CSVs.

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
    maps_dir = dirs["maps"]
    os.makedirs(maps_dir, exist_ok=True)

    error_tb = None
    with StepTimer() as timer:
        try:
            yearly_path = os.path.join(csv_dir, "districts_yearly_radiance.csv")
            trends_path = os.path.join(csv_dir, "districts_trends.csv")

            if not os.path.exists(yearly_path) or not os.path.exists(trends_path):
                return StepResult(
                    step_name="generate_district_maps",
                    status="skipped",
                    output_summary={"reason": "CSVs not found"},
                    timing_seconds=0,
                )

            yearly_df = pd.read_csv(yearly_path)
            trends_df = pd.read_csv(trends_path)
            gdf = gpd.read_file(shapefile_path)

            # Basic maps (choropleth, time series)
            from src.viirs_process import generate_maps
            generate_maps(gdf, trends_df, yearly_df, os.path.join(base_dir, "district"))

            # Statewide visualizations
            from src.outputs.visualization_suite import (
                create_multi_year_comparison_grid,
                create_growth_classification_map,
                create_enhanced_radiance_heatmap,
            )

            create_multi_year_comparison_grid(
                yearly_df, gdf,
                output_path=os.path.join(maps_dir, "multi_year_comparison.png"),
            )
            create_growth_classification_map(
                trends_df, gdf,
                output_path=os.path.join(maps_dir, "growth_classification.png"),
            )
            create_enhanced_radiance_heatmap(
                yearly_df,
                output_path=os.path.join(maps_dir, "radiance_heatmap_log.png"),
            )

            # Quality map if data exists
            quality_path = os.path.join(csv_dir, "quality_all_years.csv")
            if os.path.exists(quality_path):
                from src.outputs.visualization_suite import create_data_quality_map
                quality_df = pd.read_csv(quality_path)
                create_data_quality_map(
                    quality_df, gdf,
                    output_path=os.path.join(maps_dir, "data_quality_map.png"),
                )

            # Graduated classification maps
            grad_path = os.path.join(csv_dir, "graduated_classification.csv")
            if os.path.exists(grad_path):
                from src.analysis.graduated_classification import (
                    plot_tier_distribution,
                    plot_tier_transition_matrix,
                )
                trajectory = pd.read_csv(grad_path)
                if not trajectory.empty:
                    plot_tier_distribution(
                        trajectory,
                        output_path=os.path.join(maps_dir, "tier_distribution.png"),
                    )

            # Animation frames (sprawl, differential, darkness, trend map)
            from src.outputs.visualizations import (
                generate_sprawl_frames,
                generate_differential_frames,
                generate_darkness_frames,
                generate_trend_map,
                generate_per_district_radiance_maps,
            )

            available_years = sorted(yearly_df["year"].unique().astype(int).tolist())

            generate_sprawl_frames(
                available_years, base_dir, gdf, maps_output_dir=maps_dir,
            )
            generate_differential_frames(
                available_years, base_dir, gdf, maps_output_dir=maps_dir,
            )
            generate_darkness_frames(
                available_years, base_dir, gdf, maps_output_dir=maps_dir,
            )
            generate_trend_map(
                available_years, base_dir, gdf, maps_output_dir=maps_dir,
            )

            # Per-district radiance maps
            latest_year = int(yearly_df["year"].max())
            generate_per_district_radiance_maps(
                base_dir, latest_year, gdf, maps_output_dir=maps_dir,
            )

            log.info("District maps generated in %s", maps_dir)

        except Exception:
            import traceback
            error_tb = traceback.format_exc()

    if error_tb:
        log.error("generate_district_maps failed:\n%s", error_tb)
        return StepResult(
            step_name="generate_district_maps",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        )

    return StepResult(
        step_name="generate_district_maps",
        status="success",
        output_summary={"maps_dir": maps_dir},
        timing_seconds=timer.elapsed,
    )


def generate_city_maps(base_dir, shapefile_path):
    """Generate city-level maps from saved CSVs.

    Parameters
    ----------
    base_dir : str
        Run-level output directory containing city/ subdirectory.
    shapefile_path : str
        Path to Maharashtra district GeoJSON.

    Returns
    -------
    StepResult
    """
    dirs = get_entity_dirs(base_dir, "city")
    csv_dir = dirs["csv"]
    maps_dir = dirs["maps"]
    os.makedirs(maps_dir, exist_ok=True)

    error_tb = None
    with StepTimer() as timer:
        try:
            yearly_path = os.path.join(csv_dir, "city_yearly_radiance.csv")
            if not os.path.exists(yearly_path):
                return StepResult(
                    step_name="generate_city_maps",
                    status="skipped",
                    output_summary={"reason": "CSVs not found"},
                    timing_seconds=0,
                )

            yearly_df = pd.read_csv(yearly_path)
            district_gdf = gpd.read_file(shapefile_path)

            from src.site.site_analysis import (
                build_site_geodataframe,
                generate_site_maps,
                generate_site_timeseries,
            )

            gdf_sites = build_site_geodataframe(entity_type="city")
            latest_year = yearly_df["year"].max()
            latest_metrics = yearly_df[yearly_df["year"] == latest_year]

            latest_subset_dir = os.path.join(base_dir, "subsets", str(latest_year))
            city_dir = os.path.join(base_dir, "city")
            generate_site_maps(
                gdf_sites, latest_metrics, district_gdf,
                latest_subset_dir, city_dir, latest_year,
            )
            generate_site_timeseries(yearly_df, city_dir)

            log.info("City maps generated in %s", maps_dir)

        except Exception:
            import traceback
            error_tb = traceback.format_exc()

    if error_tb:
        log.error("generate_city_maps failed:\n%s", error_tb)
        return StepResult(
            step_name="generate_city_maps",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        )

    return StepResult(
        step_name="generate_city_maps",
        status="success",
        output_summary={"maps_dir": maps_dir},
        timing_seconds=timer.elapsed,
    )


def generate_site_maps_standalone(base_dir, shapefile_path):
    """Generate dark-sky site maps from saved CSVs.

    Parameters
    ----------
    base_dir : str
        Run-level output directory containing site/ subdirectory.
    shapefile_path : str
        Path to Maharashtra district GeoJSON.

    Returns
    -------
    StepResult
    """
    dirs = get_entity_dirs(base_dir, "site")
    csv_dir = dirs["csv"]
    maps_dir = dirs["maps"]
    os.makedirs(maps_dir, exist_ok=True)

    error_tb = None
    with StepTimer() as timer:
        try:
            yearly_path = os.path.join(csv_dir, "site_yearly_radiance.csv")
            if not os.path.exists(yearly_path):
                return StepResult(
                    step_name="generate_site_maps",
                    status="skipped",
                    output_summary={"reason": "CSVs not found"},
                    timing_seconds=0,
                )

            yearly_df = pd.read_csv(yearly_path)
            district_gdf = gpd.read_file(shapefile_path)

            from src.site.site_analysis import (
                build_site_geodataframe,
                generate_site_maps,
                generate_site_timeseries,
            )

            gdf_sites = build_site_geodataframe(entity_type="site")
            latest_year = yearly_df["year"].max()
            latest_metrics = yearly_df[yearly_df["year"] == latest_year]

            latest_subset_dir = os.path.join(base_dir, "subsets", str(latest_year))
            site_dir = os.path.join(base_dir, "site")
            generate_site_maps(
                gdf_sites, latest_metrics, district_gdf,
                latest_subset_dir, site_dir, latest_year,
            )
            generate_site_timeseries(yearly_df, site_dir)

            # Sky brightness distribution
            sky_path = os.path.join(csv_dir, "sky_brightness_{}.csv".format(latest_year))
            if os.path.exists(sky_path):
                from src.analysis.sky_brightness_model import plot_sky_brightness_distribution
                sky_df = pd.read_csv(sky_path)
                plot_sky_brightness_distribution(
                    sky_df,
                    output_path=os.path.join(maps_dir, "sky_brightness_distribution.png"),
                )

            log.info("Site maps generated in %s", maps_dir)

        except Exception:
            import traceback
            error_tb = traceback.format_exc()

    if error_tb:
        log.error("generate_site_maps failed:\n%s", error_tb)
        return StepResult(
            step_name="generate_site_maps",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        )

    return StepResult(
        step_name="generate_site_maps",
        status="success",
        output_summary={"maps_dir": maps_dir},
        timing_seconds=timer.elapsed,
    )


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

    # Resolve to latest run dir
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
        result = generators[entity_type](base_dir, args.shapefile_path)
        if result.status == "skipped":
            log.info("Skipped %s maps: %s", entity_type, result.output_summary)
        elif not result.ok:
            log.error("Failed %s maps: %s", entity_type, result.error)
        else:
            log.info("Generated %s maps in %.1fs", entity_type, result.timing_seconds)

    log.info("Map generation complete.")


if __name__ == "__main__":
    main()
