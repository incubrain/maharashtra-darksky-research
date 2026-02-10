#!/usr/bin/env python3
"""
Maharashtra VIIRS Nighttime Lights: ALAN Trend Analysis (2012-2024)

Processes VIIRS DNB annual composites to compute district-level
Artificial Light at Night (ALAN) trends for Maharashtra, India.

Methods follow Section 3.1 of "Preserving India's Rural Night Skies".

Usage:
    python src/viirs_process.py [--viirs-dir ./viirs] [--shapefile-path ./data/shapefiles/maharashtra_district.geojson]
                                [--output-dir ./outputs] [--cf-threshold 5] [--years 2012-2024]
                                [--test-district Nagpur] [--test-year 2024] [--download-data]
"""

import argparse
import gzip
import logging
import os
import shutil
import sys
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

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Data download helpers
# ---------------------------------------------------------------------------

def download_shapefiles(out_dir="data/shapefiles"):
    """Download Maharashtra district boundaries (GeoJSON)."""
    url = config.SHAPEFILE_URL
    os.makedirs(out_dir, exist_ok=True)

    geojson_path = os.path.join(out_dir, config.SHAPEFILE_NAME)
    if os.path.exists(geojson_path):
        log.info("Boundary file already present at %s", geojson_path)
        return geojson_path

    log.info("Downloading Maharashtra district boundaries (GeoJSON)...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    with open(geojson_path, "wb") as f:
        f.write(r.content)

    log.info("Saved GeoJSON to %s", geojson_path)
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

    Handles NOAA EOG naming like:
      VNL_v21_npp_2013_global_vcmcfg_c202205302300.median_masked.dat.tif
      VNL_npp_2024_global_vcmslcfg_v2_c202502261200.cf_cvg.dat.tif
    """
    layers = {}
    for p in tif_paths:
        basename = os.path.basename(p).lower()
        # Check in order: most specific first to avoid false matches
        if ".median_masked." in basename:
            layers["median"] = p
        elif ".average_masked." in basename or ".avg_rade9h." in basename:
            layers["average"] = p
        elif ".cf_cvg." in basename:
            layers["cf_cvg"] = p
        elif ".lit_mask." in basename:
            layers["lit_mask"] = p
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
    """
    # Write temporary raster for rasterstats
    tmp_path = "/tmp/_viirs_filtered_tmp.tif"
    meta = {
        "driver": "GTiff",
        "height": filtered_array.shape[0],
        "width": filtered_array.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": np.nan,
    }
    with rasterio.open(tmp_path, "w", **meta) as dst:
        dst.write(filtered_array.astype("float32"), 1)

    results = zonal_stats(
        gdf, tmp_path,
        stats=["mean", "median", "count", "min", "max", "std"],
        nodata=np.nan,
        all_touched=True,
    )

    df = pd.DataFrame(results)
    df["district"] = gdf["district"].values
    df = df[["district", "mean", "median", "count", "min", "max", "std"]]
    df.columns = ["district", "mean_radiance", "median_radiance", "pixel_count",
                   "min_radiance", "max_radiance", "std_radiance"]

    os.remove(tmp_path)
    return df


# ---------------------------------------------------------------------------
# Step 4: Trend modeling
# ---------------------------------------------------------------------------

def fit_log_linear_trend(yearly_df, district_name):
    """Fit log-linear OLS: log(radiance + epsilon) ~ year, with bootstrap CI.

    TREND MODEL methodology:
    Log-linear regression (log(radiance) ~ year) is used because ALAN growth
    is approximately exponential in developing regions. The slope coefficient
    β is converted to annual % change via (exp(β) - 1) × 100.
    A small constant (config.LOG_EPSILON = 1e-6) prevents log(0) for pixels
    with zero radiance; this value is negligible relative to VIIRS minimum
    detectable radiance (~0.1 nW/cm²/sr).
    Bootstrap confidence intervals (1000 resamples) provide non-parametric
    uncertainty estimates robust to non-normal residual distributions.
    Citation: Elvidge et al. (2021), Section 3; Li et al. (2020).
    """
    sub = yearly_df[yearly_df["district"] == district_name].sort_values("year")
    if len(sub) < config.MIN_YEARS_FOR_TREND:
        return {
            "district": district_name,
            "annual_pct_change": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "r_squared": np.nan,
            "p_value": np.nan,
            "n_years": len(sub),
        }

    years = sub["year"].values.astype(float)
    radiance = sub["median_radiance"].values.astype(float)
    log_rad = np.log(radiance + config.LOG_EPSILON)

    X = sm.add_constant(years)
    model = sm.OLS(log_rad, X).fit()
    beta = model.params[1]
    annual_pct = (np.exp(beta) - 1) * 100

    # Bootstrap CI
    n_boot = config.BOOTSTRAP_RESAMPLES
    boot_pcts = []
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    for _ in range(n_boot):
        idx = rng.choice(len(years), size=len(years), replace=True)
        X_b = sm.add_constant(years[idx])
        y_b = log_rad[idx]
        try:
            m_b = sm.OLS(y_b, X_b).fit()
            boot_pcts.append((np.exp(m_b.params[1]) - 1) * 100)
        except Exception:
            continue

    boot_pcts = np.array(boot_pcts)
    ci_lo, ci_hi = config.BOOTSTRAP_CI_LEVEL
    ci_low = np.percentile(boot_pcts, ci_lo) if len(boot_pcts) > 0 else np.nan
    ci_high = np.percentile(boot_pcts, ci_hi) if len(boot_pcts) > 0 else np.nan

    return {
        "district": district_name,
        "annual_pct_change": annual_pct,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "r_squared": model.rsquared,
        "p_value": model.pvalues[1] if len(model.pvalues) > 1 else np.nan,
        "n_years": len(sub),
    }


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
    if cf_threshold is None:
        cf_threshold = config.CF_COVERAGE_THRESHOLD
    """Full pipeline for one year: unpack → subset → cleanup → filter → aggregate.

    Processes one layer at a time to minimise disk usage: each ~11 GB global
    TIF is unpacked, clipped to Maharashtra (~12 MB), then deleted before
    the next layer is processed.
    """
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
    """Orchestrate the full analysis pipeline."""
    from src.validate_names import validate_or_exit

    # Load shapefile
    gdf = gpd.read_file(args.shapefile_path)
    log.info("Loaded shapefile: %d districts", len(gdf))
    # ─── NORMALIZE DISTRICT COLUMN (LGD / Census-aligned) ─────────────
    if "dtname" in gdf.columns:
        gdf = gdf.rename(columns={"dtname": "district"})
    else:
        raise ValueError(
            f"Expected 'dtname' column not found. Columns: {list(gdf.columns)}"
        )

    gdf["district"] = gdf["district"].astype(str).str.strip()

    # ─── DEBUG / ONE-TIME AUDIT OUTPUT ────────────────────────────────
    districts = sorted(gdf["district"].unique())

    log.info("Loaded %d districts from shapefile:", len(districts))
    for d in districts:
        log.info("  - %s", d)

    # Validate district names across data sources
    validate_or_exit(args.shapefile_path, check_config=True)

    # Parse year range
    if "-" in args.years:
        start, end = args.years.split("-")
        years = list(range(int(start), int(end) + 1))
    else:
        years = [int(y) for y in args.years.split(",")]

    # Process each year
    all_yearly = []
    for year in years:
        df = process_single_year(
            year, args.viirs_dir, gdf, args.output_dir, args.cf_threshold
        )
        if df is not None:
            all_yearly.append(df)

    if not all_yearly:
        log.error("No data processed for any year!")
        sys.exit(1)

    yearly_df = pd.concat(all_yearly, ignore_index=True)
    log.info("Total records: %d (%d years × %d districts)",
             len(yearly_df), yearly_df["year"].nunique(),
             yearly_df["district"].nunique())

    # Save yearly radiance
    csv_dir = os.path.join(args.output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    yearly_path = os.path.join(csv_dir, "districts_yearly_radiance.csv")
    yearly_df.to_csv(yearly_path, index=False)
    log.info("Saved yearly data: %s", yearly_path)

    # Fit trends per district
    districts = yearly_df["district"].unique()
    trend_results = []
    for d in districts:
        result = fit_log_linear_trend(yearly_df, d)
        # Add latest year radiance
        latest = yearly_df[yearly_df["district"] == d].sort_values("year").iloc[-1]
        result["mean_radiance_latest"] = latest["mean_radiance"]
        result["median_radiance_latest"] = latest["median_radiance"]

        # Classify ALAN level
        med = latest["median_radiance"]
        if pd.isna(med):
            result["alan_class"] = "unknown"
        elif med < config.ALAN_LOW_THRESHOLD:
            result["alan_class"] = "low"
        elif med < config.ALAN_MEDIUM_THRESHOLD:
            result["alan_class"] = "medium"
        else:
            result["alan_class"] = "high"

        trend_results.append(result)

    trends_df = pd.DataFrame(trend_results)
    trends_path = os.path.join(csv_dir, "districts_trends.csv")
    trends_df.to_csv(trends_path, index=False)
    log.info("Saved trends: %s", trends_path)

    # Print summary
    log.info("\n" + "=" * 60)
    log.info("TREND SUMMARY")
    log.info("=" * 60)
    for _, row in trends_df.sort_values("annual_pct_change", ascending=False).iterrows():
        log.info("%-20s %+6.2f%% [%+.2f, %+.2f] | %.2f nW (%s)",
                 row["district"], row["annual_pct_change"],
                 row["ci_low"], row["ci_high"],
                 row["median_radiance_latest"], row["alan_class"])

    low_alan = trends_df[trends_df["alan_class"] == "low"]
    log.info("\nLow-ALAN districts (median < 1 nW/cm²/sr): %s",
             ", ".join(low_alan["district"].tolist()) if len(low_alan) > 0 else "None")

    # Generate maps
    generate_maps(gdf, trends_df, yearly_df, args.output_dir)

    # ── Temporal stability metrics (Task 3.1) ───────────────────────────
    from src.stability_metrics import compute_stability_metrics, plot_stability_scatter

    stability_metrics = compute_stability_metrics(
        yearly_df, entity_col="district",
        output_csv=os.path.join(csv_dir, "district_stability_metrics.csv"),
    )
    plot_stability_scatter(
        stability_metrics, entity_col="district",
        output_path=os.path.join(args.output_dir, "maps", "stability_scatter.png"),
    )

    # ── Breakpoint detection (Task 3.2) ───────────────────────────────
    from src.breakpoint_analysis import analyze_all_breakpoints, plot_breakpoint_timeline

    breakpoints = analyze_all_breakpoints(
        yearly_df, entity_col="district",
        output_csv=os.path.join(csv_dir, "district_breakpoints.csv"),
    )
    plot_breakpoint_timeline(
        breakpoints,
        output_path=os.path.join(args.output_dir, "maps", "breakpoint_timeline.png"),
    )

    # ── Trend model diagnostics (Task 3.3) ────────────────────────────
    from src.trend_diagnostics import (compute_all_diagnostics, plot_diagnostic_panel)

    diagnostics_dir = os.path.join(args.output_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)

    diag_df = compute_all_diagnostics(
        yearly_df, entity_col="district",
        output_csv=os.path.join(csv_dir, "trend_diagnostics.csv"),
    )

    # Generate diagnostic plots for flagged districts
    for _, row in diag_df.iterrows():
        r2 = row.get("r_squared", 1.0)
        outliers = row.get("outlier_years", "")
        n_outliers = len(outliers.split("; ")) if outliers else 0
        if (not np.isnan(r2) and r2 < 0.5) or n_outliers > 2:
            plot_diagnostic_panel(
                yearly_df, row["district"], entity_col="district",
                output_path=os.path.join(diagnostics_dir,
                                         f"{row['district']}_diagnostics.png"),
            )

    # ── Urban radial gradient analysis (Task 2.1) ─────────────────────
    latest_year = yearly_df["year"].max()
    subset_dir = os.path.join(args.output_dir, "subsets", str(latest_year))
    median_raster = os.path.join(subset_dir, f"maharashtra_median_{latest_year}.tif")

    if os.path.exists(median_raster):
        from src.gradient_analysis import extract_radial_profiles, plot_radial_decay_curves

        profiles = extract_radial_profiles(
            raster_path=median_raster,
            city_locations=config.URBAN_BENCHMARKS,
            output_csv=os.path.join(csv_dir, f"urban_radial_profiles_{latest_year}.csv"),
        )
        plot_radial_decay_curves(
            profiles,
            output_path=os.path.join(args.output_dir, "maps", "urban_radial_profiles.png"),
        )

    # ── Quality diagnostics (Task 4.1) ──────────────────────────────
    from src.quality_diagnostics import generate_quality_report, plot_quality_heatmap

    quality_dfs = []
    for year in years:
        subset_dir = os.path.join(args.output_dir, "subsets", str(year))
        median_path = os.path.join(subset_dir, f"maharashtra_median_{year}.tif")
        lit_path = os.path.join(subset_dir, f"maharashtra_lit_mask_{year}.tif")
        cf_path = os.path.join(subset_dir, f"maharashtra_cf_cvg_{year}.tif")
        if os.path.exists(median_path):
            qdf = generate_quality_report(
                median_path, lit_path, cf_path, gdf, year,
                output_csv=os.path.join(csv_dir, f"quality_report_{year}.csv"),
            )
            quality_dfs.append(qdf)

    if quality_dfs:
        all_quality = pd.concat(quality_dfs, ignore_index=True)
        all_quality.to_csv(os.path.join(csv_dir, "quality_all_years.csv"), index=False)
        plot_quality_heatmap(
            all_quality,
            output_path=os.path.join(args.output_dir, "maps", "quality_heatmap.png"),
        )

    # ── Benchmark comparison (Task 4.2) ──────────────────────────────
    from src.benchmark_comparison import compare_to_benchmarks

    compare_to_benchmarks(
        trends_df,
        output_csv=os.path.join(csv_dir, "benchmark_comparison.csv"),
    )

    # ── Statewide visualizations (Task 5.3) ──────────────────────────
    from src.visualization_suite import (
        create_multi_year_comparison_grid,
        create_growth_classification_map,
        create_enhanced_radiance_heatmap,
    )

    create_multi_year_comparison_grid(
        yearly_df, gdf,
        output_path=os.path.join(args.output_dir, "maps", "multi_year_comparison.png"),
    )
    create_growth_classification_map(
        trends_df, gdf,
        output_path=os.path.join(args.output_dir, "maps", "growth_classification.png"),
    )
    create_enhanced_radiance_heatmap(
        yearly_df,
        output_path=os.path.join(args.output_dir, "maps", "radiance_heatmap_log.png"),
    )

    if quality_dfs:
        from src.visualization_suite import create_data_quality_map
        create_data_quality_map(
            all_quality, gdf,
            output_path=os.path.join(args.output_dir, "maps", "data_quality_map.png"),
        )

    # ── Light dome modeling (Task 5.4) ───────────────────────────────
    if os.path.exists(median_raster):
        from src.light_dome_modeling import model_all_city_domes, plot_dome_comparison

        dome_metrics = model_all_city_domes(
            profiles,
            output_csv=os.path.join(csv_dir, "light_dome_metrics.csv"),
        )
        plot_dome_comparison(
            dome_metrics, profiles,
            output_path=os.path.join(args.output_dir, "maps", "light_dome_comparison.png"),
        )

    # ── Graduated classification (Task 7.2) ──────────────────────────
    from src.graduated_classification import (
        classify_temporal_trajectory,
        plot_tier_distribution,
        plot_tier_transition_matrix,
    )

    trajectory = classify_temporal_trajectory(
        yearly_df,
        output_csv=os.path.join(csv_dir, "graduated_classification.csv"),
    )
    if not trajectory.empty:
        plot_tier_distribution(
            trajectory,
            output_path=os.path.join(args.output_dir, "maps", "tier_distribution.png"),
        )
        first_year = yearly_df["year"].min()
        plot_tier_transition_matrix(
            trajectory, first_year, latest_year,
            output_path=os.path.join(args.output_dir, "maps",
                                     f"tier_transitions_{first_year}_{latest_year}.png"),
        )

    # ── District-level reports (Task 5.1) ────────────────────────────
    from src.district_reports import generate_all_district_reports

    generate_all_district_reports(
        yearly_df=yearly_df,
        trends_df=trends_df,
        stability_df=stability_metrics,
        gdf=gdf,
        output_dir=os.path.join(args.output_dir, config.OUTPUT_DIRS["district_reports"]),
    )

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

    # 2. Choropleth: Latest median radiance
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    gdf_plot.plot(
        column="median_radiance_latest", ax=ax, legend=True,
        legend_kwds={"label": "Median Radiance (nW/cm²/sr)"},
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

    # 4. Heatmap: districts × years
    pivot = yearly_df.pivot_table(
        index="district", columns="year", values="median_radiance"
    )
    pivot = pivot.sort_values(pivot.columns[-1], ascending=False)

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int), rotation=45, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("Median Radiance Heatmap: Districts × Years", fontsize=14)
    plt.colorbar(im, ax=ax, label="Median Radiance (nW/cm²/sr)")
    plt.tight_layout()
    path = os.path.join(maps_dir, "radiance_heatmap.png")
    fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Maharashtra VIIRS ALAN Trend Analysis (2012-2024)"
    )
    parser.add_argument("--viirs-dir", default="./viirs",
                        help="Root directory containing year folders with .gz files")
    parser.add_argument("--shapefile-path",
                        default="./data/shapefiles/maharashtra_district.geojson",
                        help="Path to Maharashtra district shapefile")
    parser.add_argument("--output-dir", default="./outputs",
                        help="Output directory for CSVs and maps")
    parser.add_argument("--cf-threshold", type=int, default=config.CF_COVERAGE_THRESHOLD,
                        help="Minimum cloud-free coverage threshold (default: %(default)s)")
    parser.add_argument("--years", default="2012-2024",
                        help="Year range (e.g., '2012-2024') or comma-separated years")
    parser.add_argument("--download-shapefiles", action="store_true",
                        help="Download Maharashtra shapefiles if not present")
    return parser.parse_args()


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

    trends_df, yearly_df = run_full_pipeline(args)

    log.info("\nPipeline complete!")
    log.info("Outputs:")
    log.info("  CSV:  %s/csv/districts_trends.csv", args.output_dir)
    log.info("  CSV:  %s/csv/districts_yearly_radiance.csv", args.output_dir)
    log.info("  Maps: %s/maps/", args.output_dir)


if __name__ == "__main__":
    main()
