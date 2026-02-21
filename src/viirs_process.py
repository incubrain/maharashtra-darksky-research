#!/usr/bin/env python3
"""
Maharashtra VIIRS Nighttime Lights: ALAN Trend Analysis (2012-2024)

Processes VIIRS DNB annual composites to compute district-level
Artificial Light at Night (ALAN) trends for Maharashtra, India.

Methods follow Section 3.1 of "Preserving India's Rural Night Skies".

Usage (run from project root):
    python3 -m src.viirs_process [--viirs-dir ./viirs] [--shapefile-path ./data/shapefiles/maharashtra_district.geojson]
                                 [--output-dir ./outputs] [--cf-threshold 5] [--years 2012-2024]
                                 [--download-shapefiles]
"""

import argparse
import gzip
from src.logging_config import get_pipeline_logger
import os
import shutil
import sys
import tempfile
import warnings
from glob import glob
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import requests
from rasterstats import zonal_stats
from scipy import stats as scipy_stats
from shapely.geometry import mapping
import statsmodels.api as sm

from src import config
from src.formulas.trend import fit_log_linear_trend as _core_fit_trend
from src.formulas.classification import classify_alan

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
log = get_pipeline_logger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Data download helpers
# ---------------------------------------------------------------------------

def download_shapefiles(out_dir="data/shapefiles"):
    """Download Maharashtra district boundaries and standardise column names.

    Downloads GeoJSON from the configured URL, renames columns using
    config.SHAPEFILE_COLUMN_MAP so downstream code always sees a "district"
    column regardless of the upstream source's naming convention.

    Note
    ----
    Creates ``out_dir`` if it does not exist.
    """
    os.makedirs(out_dir, exist_ok=True)
    geojson_path = os.path.join(out_dir, "maharashtra_district.geojson")
    if os.path.exists(geojson_path):
        log.info("District boundaries already present at %s", geojson_path)
        return geojson_path

    log.info("Downloading Maharashtra district boundaries...")
    r = requests.get(config.SHAPEFILE_URL, allow_redirects=True, timeout=120)
    r.raise_for_status()

    # Load into GeoDataFrame and standardise column names
    import io as _io
    gdf = gpd.read_file(_io.BytesIO(r.content))

    # Apply column renaming from config (e.g. dtname → district)
    rename_map = {k: v for k, v in config.SHAPEFILE_COLUMN_MAP.items()
                  if k in gdf.columns}
    gdf = gdf.rename(columns=rename_map)

    if "district" not in gdf.columns:
        raise ValueError(
            f"After renaming, 'district' column not found. "
            f"Columns: {list(gdf.columns)}. "
            f"Check config.SHAPEFILE_COLUMN_MAP."
        )

    # Title-case district names for consistency
    gdf["district"] = gdf["district"].str.strip().str.title()

    log.info("Downloaded %d districts, saving to %s", len(gdf), geojson_path)
    gdf.to_file(geojson_path, driver="GeoJSON")
    return geojson_path


# ---------------------------------------------------------------------------
# Step 2: Unpack and subset
# ---------------------------------------------------------------------------

def find_gz_files(year_dir):
    """Find all .gz raster files in a year directory."""
    gz_files = glob(os.path.join(year_dir, "*.gz"))
    return sorted(set(gz_files))


def unpack_gz_files(year_dir):
    """Decompress all .gz files in a year directory, return list of .tif paths.

    Handles NOAA EOG naming convention: *.dat.tif.gz → *.dat.tif
    """
    gz_files = find_gz_files(year_dir)
    if not gz_files:
        log.warning("No .gz files found in %s", year_dir)
        return []

    tif_paths = []
    for gz_path in gz_files:
        # Strip exactly the .gz suffix
        out_path = gz_path[:-3] if gz_path.endswith(".gz") else gz_path
        if os.path.exists(out_path):
            log.info("Already unpacked: %s", os.path.basename(out_path))
            tif_paths.append(out_path)
            continue
        log.info("Unpacking %s ...", os.path.basename(gz_path))
        with gzip.open(gz_path, "rb") as f_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        tif_paths.append(out_path)
    return tif_paths


def identify_layers(tif_paths):
    """Classify TIF files into layer types based on filename patterns.

    Handles both NOAA EOG naming:
      VNL_v21_npp_2013_global_vcmcfg_c202205302300.median_masked.dat.tif
    and simplified test data naming:
      median_masked_2023.tif
    """
    layers = {}
    for p in tif_paths:
        basename = os.path.basename(p).lower()
        # Check in order: most specific first to avoid false matches.
        # Use keyword-in-filename matching so both dot-delimited (NOAA)
        # and underscore-delimited (test data) names are detected.
        if "median_masked" in basename:
            layers["median"] = p
        elif "average_masked" in basename or "avg_rade9h" in basename:
            layers["average"] = p
        elif "cf_cvg" in basename:
            layers["cf_cvg"] = p
        elif "lit_mask" in basename:
            layers["lit_mask"] = p
        else:
            log.debug("Unrecognized layer file, skipping: %s", basename)
    return layers


def subset_to_maharashtra(tif_path, gdf, output_path=None):
    """Clip a global raster to the Maharashtra shapefile extent.

    SPATIAL SUBSETTING methodology:
    Global VIIRS rasters (~11 GB) are clipped to the union of all Maharashtra
    district polygons. This preserves the original VIIRS resolution (~15
    arc-seconds, ~450 m at the equator) while reducing file size to ~12 MB.
    all_touched=True ensures pixels overlapping district boundaries are
    included, preventing data loss at polygon edges.
    """
    union_geom = gdf.union_all()
    geom_json = [mapping(union_geom)]

    with rasterio.open(tif_path) as src:
        src_dtype = src.dtypes[0]
        # Use dtype-appropriate nodata: NaN for float, 0 for int
        if np.issubdtype(np.dtype(src_dtype), np.floating):
            nodata_val = np.nan
            out_dtype = "float32"
        else:
            nodata_val = 0
            out_dtype = src_dtype

        out_image, out_transform = rasterio.mask.mask(
            src, geom_json, crop=True, nodata=nodata_val, all_touched=True
        )
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": nodata_val,
            "dtype": out_dtype,
        })

    if output_path is None:
        base = os.path.splitext(os.path.basename(tif_path))[0]
        output_path = tif_path.replace(base, f"maharashtra_{base}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(out_image.astype(out_dtype))

    log.info("Subset saved: %s (%.1f MB)", output_path,
             os.path.getsize(output_path) / 1e6)
    return output_path


# ---------------------------------------------------------------------------
# Step 3: Filtering and aggregation
# ---------------------------------------------------------------------------

def apply_quality_filters(median_path, lit_mask_path=None, cf_cvg_path=None,
                          cf_threshold=None):
    """Apply quality filters and return filtered array + metadata.

    QUALITY FILTERING methodology:
    Following Elvidge et al. (2017, 2021) VIIRS preprocessing guidelines,
    three sequential filters are applied:
      1. Finite-value check: removes NaN/Inf from sensor artefacts.
      2. lit_mask filter: excludes background pixels (water, desert) that may
         contain sensor noise. "Background values are set to zero and excluded
         from composites." (Elvidge et al. 2021, Section 2.2)
      3. Cloud-free coverage filter: excludes pixels with fewer than
         cf_threshold cloud-free observations per year. "Pixels with low
         temporal coverage are susceptible to ephemeral light contamination."
         (Elvidge et al. 2017, Section 3.1, p. 5864)

    Returns
    -------
    tuple[np.ndarray, dict, rasterio.Affine]
        (filtered_array, raster_metadata, transform). The filtered array has
        NaN where quality filters excluded pixels.
    """
    if cf_threshold is None:
        cf_threshold = config.CF_COVERAGE_THRESHOLD
    with rasterio.open(median_path) as src:
        median_data = src.read(1).astype("float32")
        meta = src.meta.copy()
        transform = src.transform

    valid_mask = np.isfinite(median_data)

    if lit_mask_path and os.path.exists(lit_mask_path):
        with rasterio.open(lit_mask_path) as src:
            lit_data = src.read(1)
        valid_mask &= (lit_data > 0)
        log.info("Applied lit_mask: %d pixels pass", valid_mask.sum())

    if cf_cvg_path and os.path.exists(cf_cvg_path):
        with rasterio.open(cf_cvg_path) as src:
            cf_data = src.read(1)
        valid_mask &= (cf_data >= cf_threshold)
        log.info("Applied cf_cvg >= %d: %d pixels pass", cf_threshold,
                 valid_mask.sum())

    filtered = np.where(valid_mask, median_data, np.nan)
    return filtered, meta, transform


def compute_district_stats(filtered_array, transform, gdf):
    """Compute zonal statistics per district from filtered raster array.

    ZONAL STATISTICS methodology:
    We compute mean, median, count, min, max, and std per district polygon.
    Median is preferred over mean for radiance aggregation because VIIRS
    radiance distributions are right-skewed (few very bright pixels in urban
    cores). Median is robust to outliers from gas flares, fires, and sensor
    saturation. (Elvidge et al. 2021, Section 3.1)
    all_touched=True ensures boundary pixels are included, following standard
    practice for administrative boundary zonal statistics.

    Note: Uses the numpy array + affine approach for zonal_stats instead of
    MemoryFile to avoid GDAL environment corruption after large raster
    operations (unpack/subset of ~11 GB global composites). The MemoryFile
    approach silently returns None for all stats when GDAL's internal state
    is corrupted by prior rasterio.mask.mask() calls on large files.
    """
    results = zonal_stats(
        gdf, filtered_array.astype("float32"),
        stats=["mean", "median", "count", "min", "max", "std"],
        nodata=np.nan,
        all_touched=True,
        affine=transform,
    )

    df = pd.DataFrame(results)
    df["district"] = gdf["district"].values
    df = df[["district", "mean", "median", "count", "min", "max", "std"]]
    df.columns = ["district", "mean_radiance", "median_radiance", "pixel_count",
                   "min_radiance", "max_radiance", "std_radiance"]
    return df


# ---------------------------------------------------------------------------
# Step 4: Trend modeling
# ---------------------------------------------------------------------------

def fit_log_linear_trend(yearly_df, district_name):
    """Fit log-linear OLS: log(radiance + epsilon) ~ year, with bootstrap CI.

    Thin wrapper around src.formulas.trend.fit_log_linear_trend() that
    extracts arrays from the DataFrame and adds the "district" key to the result.

    TREND MODEL methodology:
    Log-linear regression (log(radiance) ~ year) is used because ALAN growth
    is approximately exponential in developing regions. The slope coefficient
    β is converted to annual % change via (exp(β) - 1) × 100.
    Citation: Elvidge et al. (2021), Section 3; Li et al. (2020).
    """
    sub = yearly_df[yearly_df["district"] == district_name].sort_values("year")
    years = sub["year"].values.astype(float)
    radiance = sub["median_radiance"].values.astype(float)

    result = _core_fit_trend(years, radiance)
    result["district"] = district_name
    return result


# ---------------------------------------------------------------------------
# Step 4b: Data provenance tracking
# ---------------------------------------------------------------------------

def update_manifest(year, output_dir, cf_threshold):
    """Update data_manifest.json with processing record for reproducibility."""
    import json
    from datetime import datetime

    manifest_path = os.path.join(os.path.dirname(output_dir), "data_manifest.json")
    if not os.path.exists(manifest_path):
        manifest_path = os.path.join(".", "data_manifest.json")
    if not os.path.exists(manifest_path):
        log.warning("data_manifest.json not found; skipping manifest update")
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    entry = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "year_processed": year,
        "config_snapshot": {
            "cf_threshold": cf_threshold,
            "use_lit_mask": config.USE_LIT_MASK,
            "use_cf_filter": config.USE_CF_FILTER,
        },
    }
    manifest.setdefault("processing_history", []).append(entry)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Updated data_manifest.json for year %d", year)


# ---------------------------------------------------------------------------
# Step 5: Full pipeline
# ---------------------------------------------------------------------------

def unpack_subset_cleanup(gz_path, layer_name, year, gdf, output_dir):
    """Unpack a single .gz -> subset to Maharashtra -> delete the global TIF.

    YEAR-BY-YEAR PROCESSING methodology:
    VIIRS global composites are ~11 GB each (4 layers × 11 GB = ~44 GB per
    year). To avoid disk exhaustion on machines with limited storage, we
    process one layer at a time: unpack .gz -> clip to Maharashtra (~12 MB)
    -> delete global TIF before processing the next layer. This trades
    processing speed for disk efficiency, a necessary constraint for the
    13-year study period.
    """
    subset_dir = os.path.join(output_dir, "subsets", str(year))
    os.makedirs(subset_dir, exist_ok=True)
    out_path = os.path.join(subset_dir, f"maharashtra_{layer_name}_{year}.tif")

    if os.path.exists(out_path):
        log.info("Subset already exists: %s", os.path.basename(out_path))
        return out_path

    tif_path = gz_path[:-3]  # strip .gz
    unpacked_here = False

    try:
        # Unpack
        if not os.path.exists(tif_path):
            log.info("Unpacking %s ...", os.path.basename(gz_path))
            with gzip.open(gz_path, "rb") as f_in:
                with open(tif_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            unpacked_here = True
        else:
            log.info("Already unpacked: %s", os.path.basename(tif_path))

        # Subset
        subset_to_maharashtra(tif_path, gdf, out_path)
        return out_path

    except Exception as e:
        log.error("Failed to process %s for %d: %s", layer_name, year, e)
        return None

    finally:
        # Always remove the global TIF to free disk space
        if unpacked_here and os.path.exists(tif_path):
            os.remove(tif_path)
            log.info("Cleaned up global TIF: %s", os.path.basename(tif_path))


def process_single_year(year, viirs_dir, gdf, output_dir, cf_threshold=None):
    """Full pipeline for one year: unpack → subset → cleanup → filter → aggregate.

    Processes one layer at a time to minimise disk usage: each ~11 GB global
    TIF is unpacked, clipped to Maharashtra (~12 MB), then deleted before
    the next layer is processed.
    """
    if cf_threshold is None:
        cf_threshold = config.CF_COVERAGE_THRESHOLD
    year_dir = os.path.join(viirs_dir, str(year))
    if not os.path.isdir(year_dir):
        log.warning("Year directory not found: %s", year_dir)
        return None

    log.info("=" * 60)
    log.info("Processing year %d", year)
    log.info("=" * 60)

    # Identify .gz files and classify by layer
    gz_files = find_gz_files(year_dir)
    if not gz_files:
        log.error("No .gz files found for %d", year)
        return None

    gz_layers = identify_layers(gz_files)
    log.info("Found layers: %s", list(gz_layers.keys()))

    # Require at least median or average
    radiance_key = "median" if "median" in gz_layers else "average" if "average" in gz_layers else None
    if radiance_key is None:
        log.error("No radiance layer found for %d", year)
        return None

    # Process each layer: unpack → subset → delete global TIF
    subset_layers = {}
    for layer_name, gz_path in gz_layers.items():
        result = unpack_subset_cleanup(gz_path, layer_name, year, gdf, output_dir)
        if result:
            subset_layers[layer_name] = result

    # Filter
    radiance_path = subset_layers.get(radiance_key)
    if radiance_path is None:
        log.error("No radiance subset for %d", year)
        return None

    filtered, meta, transform = apply_quality_filters(
        radiance_path,
        lit_mask_path=subset_layers.get("lit_mask"),
        cf_cvg_path=subset_layers.get("cf_cvg"),
        cf_threshold=cf_threshold,
    )

    # Aggregate
    df = compute_district_stats(filtered, transform, gdf)
    df["year"] = year
    log.info("Year %d: %d districts processed", year, len(df))

    # Update data manifest for provenance
    update_manifest(year, output_dir, cf_threshold)

    return df


def run_full_pipeline(args):
    """Orchestrate the full district analysis pipeline."""
    from src.pipeline_steps import (
        step_load_boundaries,
        step_process_years,
        step_save_yearly_radiance,
        step_fit_trends,
        step_stability_analysis,
        step_breakpoint_detection,
        step_trend_diagnostics,
        step_quality_diagnostics,
        step_benchmark_comparison,
        step_radial_gradient_analysis,
        step_light_dome_modeling,
        step_generate_basic_maps,
        step_statewide_visualizations,
        step_graduated_classification,
        step_district_reports,
    )

    steps = []

    # Step 1: Load boundaries
    result, gdf = step_load_boundaries(args.shapefile_path)
    steps.append(result)
    if not result.ok:
        log.error("Pipeline aborted: %s", result.error)
        sys.exit(1)

    # Step 2: Parse years
    if "-" in args.years:
        start, end = args.years.split("-")
        years = list(range(int(start), int(end) + 1))
    else:
        years = [int(y) for y in args.years.split(",")]

    # Step 3: Process years (critical)
    result, yearly_df = step_process_years(years, args.viirs_dir, gdf, args.output_dir, args.cf_threshold)
    steps.append(result)
    if not result.ok:
        log.error("Pipeline aborted: no data processed")
        sys.exit(1)

    # Prepare output directories (entity-based)
    dirs = config.get_entity_dirs(args.output_dir, "district")
    csv_dir = dirs["csv"]
    maps_dir = dirs["maps"]
    diagnostics_dir = dirs["diagnostics"]

    # Step 4: Save yearly radiance (critical)
    result, yearly_path = step_save_yearly_radiance(yearly_df, csv_dir)
    steps.append(result)
    if not result.ok:
        log.error("Failed to save yearly radiance: %s", result.error)
        sys.exit(1)

    # Step 5: Fit trends (critical)
    result, trends_df = step_fit_trends(yearly_df, csv_dir)
    steps.append(result)
    if not result.ok:
        log.error("Pipeline aborted: trend fitting failed")
        sys.exit(1)

    # Step 6: Generate basic maps (non-critical)
    result, _ = step_generate_basic_maps(gdf, trends_df, yearly_df, args.output_dir, maps_dir)
    steps.append(result)
    if not result.ok:
        log.warning("Basic maps failed: %s", result.error)

    # Step 7: Stability analysis (non-critical)
    result, stability_df = step_stability_analysis(yearly_df, csv_dir, diagnostics_dir)
    steps.append(result)
    if not result.ok:
        log.warning("Stability analysis failed: %s", result.error)
        stability_df = None

    # Step 8: Breakpoint detection (non-critical)
    result, breakpoints_df = step_breakpoint_detection(yearly_df, csv_dir, diagnostics_dir)
    steps.append(result)
    if not result.ok:
        log.warning("Breakpoint detection failed: %s", result.error)
        breakpoints_df = None

    # Step 9: Trend diagnostics (non-critical)
    result, diag_df = step_trend_diagnostics(yearly_df, csv_dir, diagnostics_dir)
    steps.append(result)
    if not result.ok:
        log.warning("Trend diagnostics failed: %s", result.error)
        diag_df = None

    # Step 10: Quality diagnostics (non-critical)
    result, quality_df = step_quality_diagnostics(years, args.output_dir, gdf, csv_dir, diagnostics_dir)
    steps.append(result)
    if not result.ok:
        log.warning("Quality diagnostics failed: %s", result.error)
        quality_df = None

    # Step 11: Benchmark comparison (non-critical)
    result, _ = step_benchmark_comparison(trends_df, csv_dir)
    steps.append(result)
    if not result.ok:
        log.warning("Benchmark comparison failed: %s", result.error)

    # Step 12: Radial gradient analysis (non-critical)
    latest_year = yearly_df["year"].max()
    result, profiles_df = step_radial_gradient_analysis(args.output_dir, latest_year, csv_dir, maps_dir)
    steps.append(result)
    if not result.ok:
        log.warning("Radial gradient analysis failed: %s", result.error)
        profiles_df = None

    # Step 13: Light dome modeling (non-critical, requires profiles)
    if profiles_df is not None:
        result, dome_metrics = step_light_dome_modeling(profiles_df, csv_dir, maps_dir)
        steps.append(result)
        if not result.ok:
            log.warning("Light dome modeling failed: %s", result.error)

    # Step 14: Statewide visualizations (non-critical)
    result, _ = step_statewide_visualizations(yearly_df, trends_df, gdf, quality_df, args.output_dir, maps_dir)
    steps.append(result)
    if not result.ok:
        log.warning("Statewide visualizations failed: %s", result.error)

    # Step 15: Graduated classification (non-critical)
    result, _ = step_graduated_classification(yearly_df, csv_dir, maps_dir)
    steps.append(result)
    if not result.ok:
        log.warning("Graduated classification failed: %s", result.error)

    # Step 16: District reports (non-critical)
    reports_dir = dirs["reports"]
    result, _ = step_district_reports(yearly_df, trends_df, stability_df, gdf, args.output_dir, reports_dir)
    steps.append(result)
    if not result.ok:
        log.warning("District reports failed: %s", result.error)

    return trends_df, yearly_df


# ---------------------------------------------------------------------------
# Step 6: Visualization
# ---------------------------------------------------------------------------

def generate_maps(gdf, trends_df, yearly_df, output_dir):
    """Generate publication-quality maps and charts."""
    maps_dir = os.path.join(output_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)

    gdf_plot = gdf.merge(trends_df, on="district", how="left")

    # 1. Choropleth: Annual % change
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    gdf_plot.plot(
        column="annual_pct_change", ax=ax, legend=True,
        legend_kwds={"label": "Annual % Change in Radiance"},
        cmap="RdYlGn_r", edgecolor="black", linewidth=0.5,
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )
    for _, row in gdf_plot.iterrows():
        centroid = row.geometry.centroid
        ax.annotate(row["district"], xy=(centroid.x, centroid.y),
                    fontsize=5, ha="center", va="center")
    ax.set_title("Maharashtra: Annual % Change in ALAN (2012-2024)", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    path = os.path.join(maps_dir, "maharashtra_alan_trends.png")
    fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)

    # 2. Choropleth: Latest median radiance (log scale for visibility)
    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    radiance_col = "median_radiance_latest"
    # Use log-norm so Mumbai doesn't dominate the color scale
    vmin = max(gdf_plot[radiance_col].min(), 0.1)
    vmax = gdf_plot[radiance_col].max()
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    gdf_plot.plot(
        column=radiance_col, ax=ax, legend=True, norm=norm,
        legend_kwds={"label": "Median Radiance (nW/cm²/sr, log scale)"},
        cmap="YlOrRd", edgecolor="black", linewidth=0.5,
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )
    for _, row in gdf_plot.iterrows():
        centroid = row.geometry.centroid
        ax.annotate(row["district"], xy=(centroid.x, centroid.y),
                    fontsize=5, ha="center", va="center")
    ax.set_title("Maharashtra: Median Nighttime Radiance (Latest Year)", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    path = os.path.join(maps_dir, "maharashtra_radiance_latest.png")
    fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)

    # 3. Time series for selected districts
    highlight = list(config.TIMESERIES_HIGHLIGHT_DISTRICTS)
    available = yearly_df["district"].unique()
    highlight = [d for d in highlight if d in available]
    if not highlight:
        highlight = list(available[:5])

    fig, ax = plt.subplots(figsize=(12, 7))
    for d in highlight:
        sub = yearly_df[yearly_df["district"] == d].sort_values("year")
        ax.plot(sub["year"], sub["median_radiance"], "o-", label=d, markersize=4)
    ax.set_yscale("log")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Median Radiance (nW/cm²/sr, log scale)", fontsize=12)
    ax.set_title("ALAN Time Series: Selected Maharashtra Districts", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(maps_dir, "radiance_timeseries.png")
    fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)

    # Note: radiance heatmap consolidated into radiance_heatmap_log.png
    # (generated by visualization_suite.create_enhanced_radiance_heatmap)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Maharashtra VIIRS ALAN Trend Analysis (2012-2024)"
    )
    parser.add_argument("--viirs-dir", default=config.DEFAULT_VIIRS_DIR,
                        help="Root directory containing year folders with .gz files")
    parser.add_argument("--shapefile-path",
                        default=config.DEFAULT_SHAPEFILE_PATH,
                        help="Path to Maharashtra district boundaries (GeoJSON)")
    parser.add_argument("--output-dir", default=config.DEFAULT_OUTPUT_DIR,
                        help="Output directory for CSVs and maps")
    parser.add_argument("--cf-threshold", type=int, default=config.CF_COVERAGE_THRESHOLD,
                        help="Minimum cloud-free coverage threshold (default: %(default)s)")
    parser.add_argument("--years", default="2012-2024",
                        help="Year range (e.g., '2012-2024') or comma-separated years")
    parser.add_argument("--download-shapefiles", action="store_true",
                        help="Download Maharashtra shapefiles if not present")
    return parser.parse_args()


def _create_run_dir(base_output_dir, args):
    """Create a timestamped run directory and save the config snapshot.

    Structure:
        outputs/runs/2026-02-11_143022/
            config_snapshot.json   ← frozen settings for this run
            subsets/{year}/        ← shared rasters
            district/
              csv/                 ← district analysis results
              maps/                ← district visualizations
              reports/             ← district PDFs
              diagnostics/         ← district diagnostic plots
            city/
              csv/                 ← city analysis results
              maps/                ← city visualizations
              reports/             ← city PDFs
              diagnostics/
            site/
              csv/                 ← site analysis results
              maps/                ← site visualizations
              reports/             ← site PDFs
              diagnostics/
        outputs/latest → runs/2026-02-11_143022  (symlink)

    Returns the run-specific output directory path.
    """
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, "runs", timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Create entity subdirectories
    for entity in ["district", "city", "site"]:
        entity_dirs = config.get_entity_dirs(run_dir, entity)
        for d in entity_dirs.values():
            os.makedirs(d, exist_ok=True)

    # Save config snapshot
    snapshot = {
        "timestamp": timestamp,
        "viirs_dir": args.viirs_dir,
        "shapefile_path": args.shapefile_path,
        "cf_threshold": args.cf_threshold,
        "years": args.years,
        "use_lit_mask": config.USE_LIT_MASK,
        "use_cf_filter": config.USE_CF_FILTER,
        "log_epsilon": config.LOG_EPSILON,
        "bootstrap_resamples": config.BOOTSTRAP_RESAMPLES,
        "site_buffer_radius_km": config.SITE_BUFFER_RADIUS_KM,
        "alan_low_threshold": config.ALAN_LOW_THRESHOLD,
        "alan_medium_threshold": config.ALAN_MEDIUM_THRESHOLD,
    }
    snapshot_path = os.path.join(run_dir, "config_snapshot.json")
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    log.info("Config snapshot saved: %s", snapshot_path)

    # Update 'latest' symlink
    latest_link = os.path.join(base_output_dir, "latest")
    rel_target = os.path.join("runs", timestamp)
    try:
        if os.path.islink(latest_link):
            os.remove(latest_link)
        elif os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(rel_target, latest_link)
        log.info("Updated symlink: %s → %s", latest_link, rel_target)
    except OSError as e:
        log.warning("Could not create 'latest' symlink: %s", e)

    return run_dir


def main():
    args = parse_args()

    log.info("Maharashtra VIIRS ALAN Trend Analysis")
    log.info("Configuration:")
    log.info("  VIIRS dir:      %s", args.viirs_dir)
    log.info("  Shapefile:      %s", args.shapefile_path)
    log.info("  Output dir:     %s", args.output_dir)
    log.info("  CF threshold:   %d", args.cf_threshold)
    log.info("  Years:          %s", args.years)

    if args.download_shapefiles or not os.path.exists(args.shapefile_path):
        args.shapefile_path = download_shapefiles()

    # Create run-specific output directory with config snapshot
    base_output_dir = args.output_dir
    run_dir = _create_run_dir(base_output_dir, args)
    args.output_dir = run_dir
    log.info("  Run dir:        %s", run_dir)

    trends_df, yearly_df = run_full_pipeline(args)

    log.info("\nPipeline complete!")
    log.info("Outputs:")
    log.info("  Run dir: %s", run_dir)
    log.info("  CSV:     %s/csv/districts_trends.csv", run_dir)
    log.info("  CSV:     %s/csv/districts_yearly_radiance.csv", run_dir)
    log.info("  Maps:    %s/maps/", run_dir)
    log.info("  Latest:  %s/latest/", base_output_dir)


if __name__ == "__main__":
    main()
