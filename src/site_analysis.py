#!/usr/bin/env python3
"""
Site-level ALAN analysis for 5 cities and 11 dark-sky candidate sites.

Uses Maharashtra VIIRS subsets (already produced by viirs_process.py)
to compute radiance metrics at sub-district resolution via 10 km buffers
around point coordinates, then fits log-linear trends over the year range.

Usage (run from project root):
    python3 -m src.site_analysis [--output-dir ./outputs] [--buffer-km 10]
                                 [--cf-threshold 5] [--years 2012-2024]
"""

import argparse
from src.logging_config import get_pipeline_logger
import os
import tempfile

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import Point

from src import config
from src.formulas.trend import fit_log_linear_trend as _core_fit_trend
from src.formulas.classification import classify_alan, classify_alan_series

log = get_pipeline_logger(__name__)


def _build_locations():
    """Build LOCATIONS dict from config.URBAN_BENCHMARKS and config.DARKSKY_SITES."""
    locations = {}
    for name, info in config.URBAN_BENCHMARKS.items():
        locations[name] = (info["lat"], info["lon"], "city", info["district"])
    for name, info in config.DARKSKY_SITES.items():
        locations[name] = (info["lat"], info["lon"], "site", info["district"])
    return locations


def _build_locations_filtered(entity_type="all"):
    """Build LOCATIONS dict, optionally filtered by entity type."""
    locations = {}
    if entity_type in ("city", "all"):
        for name, info in config.URBAN_BENCHMARKS.items():
            locations[name] = (info["lat"], info["lon"], "city", info["district"])
    if entity_type in ("site", "all"):
        for name, info in config.DARKSKY_SITES.items():
            locations[name] = (info["lat"], info["lon"], "site", info["district"])
    return locations


LOCATIONS = _build_locations()


def build_site_geodataframe(buffer_km=None, entity_type="all"):
    """Create GeoDataFrame with circular buffers around each site.

    BUFFER ANALYSIS methodology:
    Following Wang et al. (2022), a 10 km radius buffer around point sites
    captures the local ALAN environment while being large enough to contain
    sufficient VIIRS pixels (~440 pixels at 450 m resolution) for robust
    statistics. The buffer is computed in UTM Zone 43N (EPSG:32643) for
    metric accuracy, then reprojected to WGS84 for raster extraction.
    Citation: Wang, J. et al. (2022). Protected area buffer analysis.
    """
    if buffer_km is None:
        buffer_km = config.SITE_BUFFER_RADIUS_KM
    locations = _build_locations_filtered(entity_type)
    rows = []
    for name, (lat, lon, loc_type, district) in locations.items():
        rows.append({
            "name": name,
            "type": loc_type,
            "district": district,
            "lat": lat,
            "lon": lon,
            "geometry": Point(lon, lat),
        })

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    # Project to UTM 43N (covers Maharashtra) for metric buffer
    gdf_utm = gdf.to_crs(epsg=32643)
    gdf_utm["geometry"] = gdf_utm.geometry.buffer(buffer_km * 1000)
    gdf_buffered = gdf_utm.to_crs(epsg=4326)

    log.info("Created %d site buffers (%d km radius)", len(gdf_buffered), buffer_km)
    for _, row in gdf_buffered.iterrows():
        area_km2 = gdf_utm[gdf_utm["name"] == row["name"]].geometry.area.values[0] / 1e6
        log.info("  %-30s %s  %.0f km²  (%.4f°N, %.4f°E)",
                 row["name"], row["type"], area_km2, row["lat"], row["lon"])

    return gdf_buffered


def compute_site_metrics(gdf, subset_dir, year=2024, cf_threshold=None):
    """Compute filtered radiance stats for each site buffer."""
    if cf_threshold is None:
        cf_threshold = config.CF_COVERAGE_THRESHOLD
    median_path = os.path.join(subset_dir, f"maharashtra_median_{year}.tif")
    lit_mask_path = os.path.join(subset_dir, f"maharashtra_lit_mask_{year}.tif")
    cf_cvg_path = os.path.join(subset_dir, f"maharashtra_cf_cvg_{year}.tif")

    for p in [median_path, lit_mask_path, cf_cvg_path]:
        if not os.path.exists(p):
            log.error("Missing raster: %s", p)
            return None

    # Build filtered raster: apply quality masks, write temp file
    with rasterio.open(median_path) as src:
        median_data = src.read(1).astype("float32")
        transform = src.transform
        crs = src.crs

    with rasterio.open(lit_mask_path) as src:
        lit_data = src.read(1)

    with rasterio.open(cf_cvg_path) as src:
        cf_data = src.read(1)

    valid = np.isfinite(median_data) & (lit_data > 0) & (cf_data >= cf_threshold)
    total_valid = valid.sum()
    filtered = np.where(valid, median_data, np.nan)
    log.info("Quality filter: %d pixels pass (lit_mask & cf_cvg >= %d)",
             total_valid, cf_threshold)

    # Use in-memory raster to avoid temp file side effects
    from rasterio.io import MemoryFile

    meta = {
        "driver": "GTiff", "height": filtered.shape[0], "width": filtered.shape[1],
        "count": 1, "dtype": "float32", "crs": crs, "transform": transform,
        "nodata": np.nan,
    }

    # Create filtered raster in memory
    with MemoryFile() as memfile_filt:
        with memfile_filt.open(**meta) as dst:
            dst.write(filtered, 1)

        # Create unfiltered raster in memory for pixel count comparison
        with MemoryFile() as memfile_unfilt:
            unfilt = np.where(np.isfinite(median_data), median_data, np.nan)
            with memfile_unfilt.open(**meta) as dst:
                dst.write(unfilt, 1)

            # Compute zonal stats on both
            with memfile_filt.open() as src_filt:
                results_filt = zonal_stats(
                    gdf, src_filt.name,
                    stats=["mean", "median", "count", "min", "max", "std"],
                    nodata=np.nan, all_touched=True,
                )

            with memfile_unfilt.open() as src_unfilt:
                results_unfilt = zonal_stats(
                    gdf, src_unfilt.name,
                    stats=["count"],
                    nodata=np.nan, all_touched=True,
                )

    df = pd.DataFrame(results_filt)
    df["name"] = gdf["name"].values
    df["type"] = gdf["type"].values
    df["district"] = gdf["district"].values
    df["total_pixels"] = [r["count"] for r in results_unfilt]
    df["quality_pct"] = (df["count"] / df["total_pixels"].replace(0, np.nan) * 100).round(1)
    df["year"] = year

    df = df.rename(columns={
        "mean": "mean_radiance",
        "median": "median_radiance",
        "count": "valid_pixels",
        "min": "min_radiance",
        "max": "max_radiance",
        "std": "std_radiance",
    })

    # Classify ALAN level
    df["alan_class"] = classify_alan_series(df["median_radiance"])

    cols = ["name", "type", "district", "year", "mean_radiance", "median_radiance",
            "min_radiance", "max_radiance", "std_radiance",
            "valid_pixels", "total_pixels", "quality_pct", "alan_class"]
    return df[cols]


def generate_site_maps(gdf_sites, metrics_df, district_gdf, subset_dir, output_dir,
                       year=2024):
    """Generate maps overlaying sites on Maharashtra with radiance data."""
    maps_dir = os.path.join(output_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)

    merged = gdf_sites.merge(metrics_df[["name", "median_radiance", "alan_class"]],
                             on="name")

    # ── Map 1: Site overlay on Maharashtra ────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 11))
    district_gdf.plot(ax=ax, color="whitesmoke", edgecolor="grey", linewidth=0.5)

    cities = merged[merged["type"] == "city"]
    sites = merged[merged["type"] == "site"]

    sites.plot(ax=ax, color="forestgreen", alpha=0.4, edgecolor="darkgreen",
               linewidth=1.2, label="Dark-sky sites")
    cities.plot(ax=ax, color="crimson", alpha=0.4, edgecolor="darkred",
                linewidth=1.2, label="Cities")

    for _, row in merged.iterrows():
        centroid = row.geometry.centroid
        offset_x = 0.15 if row["type"] == "city" else 0.12
        ax.annotate(
            f"{row['name']}\n{row['median_radiance']:.2f} nW",
            xy=(centroid.x, centroid.y),
            xytext=(centroid.x + offset_x, centroid.y + 0.08),
            fontsize=6, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.8),
        )

    city_patch = mpatches.Patch(color="crimson", alpha=0.4, label="Cities (10 km buffer)")
    site_patch = mpatches.Patch(color="forestgreen", alpha=0.4,
                                label="Dark-sky sites (10 km buffer)")
    ax.legend(handles=[city_patch, site_patch], loc="lower left", fontsize=9)
    ax.set_title(f"Maharashtra: ALAN Analysis Sites ({year})", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    path = os.path.join(maps_dir, "site_overlay_map.png")
    fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)

    # ── Map 2: Bar chart comparison ───────────────────────────────────
    df_sorted = metrics_df.sort_values("median_radiance", ascending=True)
    colors = ["crimson" if t == "city" else "forestgreen" for t in df_sorted["type"]]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(df_sorted["name"], df_sorted["median_radiance"], color=colors,
                   edgecolor="grey", linewidth=0.5)
    ax.set_xlabel("Median Radiance (nW/cm²/sr)", fontsize=12)
    ax.set_title(f"ALAN Comparison: Cities vs Dark-Sky Sites ({year})", fontsize=14)
    ax.axvline(x=config.ALAN_LOW_THRESHOLD, color="orange", linestyle="--", linewidth=1,
               label=f"Low-ALAN threshold ({config.ALAN_LOW_THRESHOLD} nW)")

    for bar, val in zip(bars, df_sorted["median_radiance"]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)

    city_patch = mpatches.Patch(color="crimson", label="Cities")
    site_patch = mpatches.Patch(color="forestgreen", label="Dark-sky sites")
    ax.legend(handles=[city_patch, site_patch, ax.lines[0]], loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(maps_dir, "site_comparison_chart.png")
    fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)

    # ── Map 3: Radiance raster with site overlay ──────────────────────
    median_path = os.path.join(subset_dir, f"maharashtra_median_{year}.tif")
    with rasterio.open(median_path) as src:
        raster = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    fig, ax = plt.subplots(figsize=(14, 11))
    # Log-scale radiance for visibility
    raster_log = np.log10(np.clip(raster, 0.01, None))
    raster_log = np.where(np.isfinite(raster_log), raster_log, np.nan)
    im = ax.imshow(raster_log, extent=extent, cmap="magma", vmin=-2, vmax=2,
                   origin="upper", aspect="auto")
    district_gdf.boundary.plot(ax=ax, edgecolor="white", linewidth=0.3, alpha=0.5)
    sites.plot(ax=ax, facecolor="none", edgecolor="lime", linewidth=1.5)
    cities.plot(ax=ax, facecolor="none", edgecolor="cyan", linewidth=1.5)

    for _, row in merged.iterrows():
        c = row.geometry.centroid
        ax.annotate(row["name"], xy=(c.x, c.y), fontsize=5, color="white",
                    ha="center", va="bottom", fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, label="log₁₀(Radiance nW/cm²/sr)")
    ax.set_title(f"Maharashtra: Nighttime Radiance with Analysis Sites ({year})",
                 fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    path = os.path.join(maps_dir, "radiance_with_sites.png")
    fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)


def fit_site_trends(yearly_df):
    """Fit log-linear OLS trends per site, with bootstrap CI.

    Uses src.formulas.trend.fit_log_linear_trend() for core computation.
    Returns DataFrame with annual_pct_change, ci_low, ci_high, r_squared, p_value.
    """
    results = []
    for name in yearly_df["name"].unique():
        sub = yearly_df[yearly_df["name"] == name].sort_values("year")
        loc_type = sub["type"].iloc[0]
        district = sub["district"].iloc[0]

        years = sub["year"].values.astype(float)
        radiance = sub["median_radiance"].values.astype(float)

        core_result = _core_fit_trend(
            years, radiance,
            min_years=config.MIN_YEARS_FOR_SITE_TREND,
        )

        row = {
            "name": name, "type": loc_type, "district": district,
            "annual_pct_change": core_result["annual_pct_change"],
            "ci_low": core_result["ci_low"],
            "ci_high": core_result["ci_high"],
            "r_squared": core_result["r_squared"],
            "p_value": core_result["p_value"],
            "n_years": core_result["n_years"],
        }

        # Add latest radiance if we had enough data
        if not np.isnan(core_result["annual_pct_change"]):
            latest = sub.iloc[-1]
            row["median_radiance_latest"] = latest["median_radiance"]
            row["mean_radiance_latest"] = latest["mean_radiance"]

        results.append(row)

    return pd.DataFrame(results)


def generate_site_timeseries(yearly_df, output_dir):
    """Generate time-series plots for cities and dark-sky sites."""
    maps_dir = os.path.join(output_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)

    # ── Cities time series ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    for name in yearly_df[yearly_df["type"] == "city"]["name"].unique():
        sub = yearly_df[yearly_df["name"] == name].sort_values("year")
        ax.plot(sub["year"], sub["median_radiance"], "o-", label=name, markersize=4)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Median Radiance (nW/cm²/sr)", fontsize=12)
    ax.set_title("ALAN Time Series: Urban Benchmarks (5 Cities)", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(maps_dir, "site_timeseries_cities.png")
    fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)

    # ── Dark-sky sites time series ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 8))
    for name in yearly_df[yearly_df["type"] == "site"]["name"].unique():
        sub = yearly_df[yearly_df["name"] == name].sort_values("year")
        ax.plot(sub["year"], sub["median_radiance"], "o-", label=name, markersize=4)
    ax.axhline(y=config.ALAN_LOW_THRESHOLD, color="orange", linestyle="--", linewidth=1,
               label=f"Low-ALAN threshold ({config.ALAN_LOW_THRESHOLD} nW)")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Median Radiance (nW/cm²/sr)", fontsize=12)
    ax.set_title("ALAN Time Series: Dark-Sky Candidate Sites", fontsize=14)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(maps_dir, "site_timeseries_darksites.png")
    fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)


def main():
    parser = argparse.ArgumentParser(
        description="Site-level ALAN analysis (5 cities + 11 dark-sky sites)"
    )
    parser.add_argument("--output-dir", default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shapefile-path",
                        default=config.DEFAULT_SHAPEFILE_PATH)
    parser.add_argument("--buffer-km", type=float, default=config.SITE_BUFFER_RADIUS_KM)
    parser.add_argument("--cf-threshold", type=int, default=config.CF_COVERAGE_THRESHOLD)
    parser.add_argument("--years", default="2012-2024",
                        help="Year range (e.g., '2012-2024') or single year")
    parser.add_argument("--type", choices=["city", "site", "all"], default="all",
                        help="Entity type to analyze: city, site, or all (default: all)")
    args = parser.parse_args()

    # If using default output dir and 'latest' symlink exists, use that
    # This ensures we pick up the subsets from the most recent run
    if args.output_dir == config.DEFAULT_OUTPUT_DIR:
        latest_dir = os.path.join(args.output_dir, "latest")
        if os.path.isdir(latest_dir):
            args.output_dir = latest_dir
            log.info("Using latest run directory: %s", args.output_dir)

    # Parse year range
    if "-" in args.years:
        start, end = args.years.split("-")
        years = list(range(int(start), int(end) + 1))
    else:
        years = [int(y) for y in args.years.split(",")]

    log.info("Site-level ALAN analysis (%s)", args.years)
    log.info("Buffer radius: %d km | CF threshold: %d | Type: %s",
             args.buffer_km, args.cf_threshold, args.type)

    # Validate district names
    from src.validate_names import validate_or_exit
    validate_or_exit(args.shapefile_path, check_config=True)

    # Determine which entity types to process
    entity_types = [args.type] if args.type != "all" else ["city", "site"]

    for entity_type in entity_types:
        log.info("Running %s analysis...", entity_type)
        _run_entity_pipeline(args, years, entity_type)

    log.info("\nDone! Outputs in %s/", args.output_dir)


def _run_entity_pipeline(args, years, entity_type):
    """Run the site/city analysis pipeline for a specific entity type."""
    # Import step functions
    from src.site_pipeline_steps import (
        step_build_site_buffers,
        step_compute_yearly_metrics,
        step_save_site_yearly,
        step_fit_site_trends,
        step_site_maps,
        step_spatial_analysis,
        step_sky_brightness,
        step_site_stability,
        step_site_breakpoints,
        step_site_benchmark,
        step_site_visualizations,
        step_site_reports,
    )

    steps = []

    # Get entity-specific output directories
    entity_dirs = config.get_entity_dirs(args.output_dir, entity_type)
    csv_dir = entity_dirs["csv"]
    maps_dir = entity_dirs["maps"]
    reports_dir = entity_dirs["reports"]
    diagnostics_dir = entity_dirs["diagnostics"]

    # Create directories
    for d in entity_dirs.values():
        os.makedirs(d, exist_ok=True)

    # Step 1: Build site buffers (filtered by entity_type)
    result, gdf_sites = step_build_site_buffers(args.buffer_km, entity_type)
    steps.append(result)
    if not result.ok:
        log.error("Pipeline aborted: %s", result.error)
        return

    # Step 2: Compute yearly metrics
    result, yearly_df = step_compute_yearly_metrics(years, gdf_sites, args.output_dir, args.cf_threshold)
    steps.append(result)
    if not result.ok:
        log.error("No data processed. Run viirs_process.py first.")
        return

    # Step 3: Save yearly CSV
    result, yearly_path = step_save_site_yearly(yearly_df, csv_dir)
    steps.append(result)
    if not result.ok:
        log.error("Pipeline aborted: %s", result.error)
        return

    # Step 4: Fit trends
    result, trends_df = step_fit_site_trends(yearly_df, csv_dir)
    steps.append(result)
    if not result.ok:
        log.error("Pipeline aborted: %s", result.error)
        return

    # Step 5: Generate maps (using latest year for static maps)
    latest_year = max(years)
    district_gdf = gpd.read_file(args.shapefile_path)
    result, _ = step_site_maps(gdf_sites, yearly_df, district_gdf, args.output_dir, latest_year, maps_dir)
    steps.append(result)
    if not result.ok:
        log.warning("Maps generation failed: %s", result.error)

    # Step 6: Spatial analysis enhancements
    result, spatial_results = step_spatial_analysis(gdf_sites, args.output_dir, latest_year, csv_dir, maps_dir)
    steps.append(result)
    if not result.ok:
        log.warning("Spatial analysis failed: %s", result.error)

    # Step 7: Sky brightness estimation
    latest_metrics = yearly_df[yearly_df["year"] == latest_year]
    result, sky_metrics = step_sky_brightness(latest_metrics, csv_dir, maps_dir, latest_year)
    steps.append(result)
    if not result.ok:
        log.warning("Sky brightness analysis failed: %s", result.error)

    # Step 8: Temporal stability for sites
    result, site_stability = step_site_stability(yearly_df, csv_dir, diagnostics_dir)
    steps.append(result)
    if not result.ok:
        log.warning("Stability analysis failed: %s", result.error)

    # Step 9: Breakpoint detection for sites
    result, site_breakpoints = step_site_breakpoints(yearly_df, csv_dir)
    steps.append(result)
    if not result.ok:
        log.warning("Breakpoint detection failed: %s", result.error)

    # Step 10: Benchmark comparison for sites
    result, _ = step_site_benchmark(trends_df, csv_dir)
    steps.append(result)
    if not result.ok:
        log.warning("Benchmark comparison failed: %s", result.error)

    # Step 11: City vs site visualizations
    result, _ = step_site_visualizations(latest_metrics, maps_dir)
    steps.append(result)
    if not result.ok:
        log.warning("Visualizations failed: %s", result.error)

    # Step 12: Site-level deep-dive reports
    result, _ = step_site_reports(latest_metrics, yearly_df, reports_dir, entity_type)
    steps.append(result)
    if not result.ok:
        log.warning("Report generation failed: %s", result.error)


if __name__ == "__main__":
    main()
