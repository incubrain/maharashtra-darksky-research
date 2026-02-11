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
import logging
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _build_locations():
    """Build LOCATIONS dict from config.URBAN_BENCHMARKS and config.DARKSKY_SITES."""
    locations = {}
    for name, info in config.URBAN_BENCHMARKS.items():
        locations[name] = (info["lat"], info["lon"], "city", info["district"])
    for name, info in config.DARKSKY_SITES.items():
        locations[name] = (info["lat"], info["lon"], "site", info["district"])
    return locations


LOCATIONS = _build_locations()


def build_site_geodataframe(buffer_km=None):
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
    rows = []
    for name, (lat, lon, loc_type, district) in LOCATIONS.items():
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

    # Write filtered raster to temp file for zonal_stats
    tmp_fd, tmp_path = tempfile.mkstemp(suffix="_viirs_site_filtered.tif")
    os.close(tmp_fd)
    meta = {
        "driver": "GTiff", "height": filtered.shape[0], "width": filtered.shape[1],
        "count": 1, "dtype": "float32", "crs": crs, "transform": transform,
        "nodata": np.nan,
    }

    # Also write unfiltered for pixel count comparison
    tmp_fd2, tmp_unfilt = tempfile.mkstemp(suffix="_viirs_site_unfiltered.tif")
    os.close(tmp_fd2)

    try:
        with rasterio.open(tmp_path, "w", **meta) as dst:
            dst.write(filtered, 1)

        unfilt = np.where(np.isfinite(median_data), median_data, np.nan)
        with rasterio.open(tmp_unfilt, "w", **meta) as dst:
            dst.write(unfilt, 1)

        # Zonal stats on filtered
        results_filt = zonal_stats(
            gdf, tmp_path,
            stats=["mean", "median", "count", "min", "max", "std"],
            nodata=np.nan, all_touched=True,
        )
        # Zonal stats on unfiltered (for total pixel count / quality %)
        results_unfilt = zonal_stats(
            gdf, tmp_unfilt,
            stats=["count"],
            nodata=np.nan, all_touched=True,
        )
    finally:
        for p in [tmp_path, tmp_unfilt]:
            if os.path.exists(p):
                os.remove(p)

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
    df["alan_class"] = pd.cut(
        df["median_radiance"],
        bins=[-np.inf, config.ALAN_LOW_THRESHOLD, config.ALAN_MEDIUM_THRESHOLD, np.inf],
        labels=["low", "medium", "high"],
        right=False,
    )

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

    Returns DataFrame with annual_pct_change, ci_low, ci_high, r_squared, p_value.
    """
    import statsmodels.api as sm

    results = []
    for name in yearly_df["name"].unique():
        sub = yearly_df[yearly_df["name"] == name].sort_values("year")
        loc_type = sub["type"].iloc[0]
        district = sub["district"].iloc[0]

        if len(sub) < config.MIN_YEARS_FOR_SITE_TREND:
            results.append({
                "name": name, "type": loc_type, "district": district,
                "annual_pct_change": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "r_squared": np.nan, "p_value": np.nan, "n_years": len(sub),
            })
            continue

        years = sub["year"].values.astype(float)
        radiance = sub["median_radiance"].values.astype(float)
        log_rad = np.log(radiance + config.LOG_EPSILON)

        X = sm.add_constant(years)
        model = sm.OLS(log_rad, X).fit()
        beta = model.params[1]
        annual_pct = (np.exp(beta) - 1) * 100

        # Bootstrap CI
        rng = np.random.default_rng(config.BOOTSTRAP_SEED)
        boot_pcts = []
        for _ in range(config.BOOTSTRAP_RESAMPLES):
            idx = rng.choice(len(years), size=len(years), replace=True)
            try:
                m = sm.OLS(log_rad[idx], sm.add_constant(years[idx])).fit()
                boot_pcts.append((np.exp(m.params[1]) - 1) * 100)
            except Exception:
                continue

        boot_pcts = np.array(boot_pcts)
        ci_lo, ci_hi = config.BOOTSTRAP_CI_LEVEL
        ci_low = np.percentile(boot_pcts, ci_lo) if len(boot_pcts) > 0 else np.nan
        ci_high = np.percentile(boot_pcts, ci_hi) if len(boot_pcts) > 0 else np.nan

        latest = sub.iloc[-1]
        results.append({
            "name": name, "type": loc_type, "district": district,
            "annual_pct_change": annual_pct, "ci_low": ci_low, "ci_high": ci_high,
            "r_squared": model.rsquared,
            "p_value": model.pvalues[1] if len(model.pvalues) > 1 else np.nan,
            "n_years": len(sub),
            "median_radiance_latest": latest["median_radiance"],
            "mean_radiance_latest": latest["mean_radiance"],
        })

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
    log.info("Buffer radius: %d km | CF threshold: %d", args.buffer_km,
             args.cf_threshold)

    # Validate district names
    from src.validate_names import validate_or_exit
    validate_or_exit(args.shapefile_path, check_config=True)

    # Build site buffers
    gdf_sites = build_site_geodataframe(args.buffer_km)

    # Compute metrics for each year
    all_yearly = []
    for year in years:
        subset_dir = os.path.join(args.output_dir, "subsets", str(year))
        if not os.path.isdir(subset_dir):
            log.warning("No subsets for %d — skipping", year)
            continue
        df = compute_site_metrics(gdf_sites, subset_dir, year, args.cf_threshold)
        if df is not None:
            all_yearly.append(df)
            log.info("Year %d: %d sites processed", year, len(df))

    if not all_yearly:
        log.error("No data processed. Run viirs_process.py first.")
        return

    yearly_df = pd.concat(all_yearly, ignore_index=True)
    log.info("Total records: %d (%d years × %d sites)",
             len(yearly_df), yearly_df["year"].nunique(),
             yearly_df["name"].nunique())

    # Save yearly CSV
    csv_dir = os.path.join(args.output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    yearly_path = os.path.join(csv_dir, "site_yearly_radiance.csv")
    yearly_df.to_csv(yearly_path, index=False)
    log.info("Saved: %s", yearly_path)

    # Fit trends
    trends_df = fit_site_trends(yearly_df)

    # Add ALAN classification based on latest year median
    for i, row in trends_df.iterrows():
        if "median_radiance_latest" not in row or pd.isna(row.get("median_radiance_latest")):
            latest = yearly_df[(yearly_df["name"] == row["name"])].sort_values("year").iloc[-1]
            trends_df.at[i, "median_radiance_latest"] = latest["median_radiance"]
            trends_df.at[i, "mean_radiance_latest"] = latest["mean_radiance"]
        med = trends_df.at[i, "median_radiance_latest"]
        if pd.isna(med):
            trends_df.at[i, "alan_class"] = "unknown"
        elif med < config.ALAN_LOW_THRESHOLD:
            trends_df.at[i, "alan_class"] = "low"
        elif med < config.ALAN_MEDIUM_THRESHOLD:
            trends_df.at[i, "alan_class"] = "medium"
        else:
            trends_df.at[i, "alan_class"] = "high"

    trends_path = os.path.join(csv_dir, "site_trends.csv")
    trends_df.to_csv(trends_path, index=False)
    log.info("Saved: %s", trends_path)

    # Print summary
    log.info("\n" + "=" * 70)
    log.info("SITE TREND SUMMARY (%s)", args.years)
    log.info("=" * 70)

    cities_t = trends_df[trends_df["type"] == "city"].sort_values(
        "annual_pct_change", ascending=False)
    sites_t = trends_df[trends_df["type"] == "site"].sort_values(
        "annual_pct_change", ascending=False)

    log.info("\n--- CITIES ---")
    for _, r in cities_t.iterrows():
        log.info("  %-20s %+6.2f%% [%+.2f, %+.2f]  R²=%.3f  p=%.1e  latest=%.2f nW  [%s]",
                 r["name"], r["annual_pct_change"], r["ci_low"], r["ci_high"],
                 r["r_squared"], r["p_value"],
                 r["median_radiance_latest"], r["alan_class"])

    log.info("\n--- DARK-SKY SITES ---")
    for _, r in sites_t.iterrows():
        log.info("  %-30s %+6.2f%% [%+.2f, %+.2f]  R²=%.3f  p=%.1e  latest=%.2f nW  [%s]",
                 r["name"], r["annual_pct_change"], r["ci_low"], r["ci_high"],
                 r["r_squared"], r["p_value"],
                 r["median_radiance_latest"], r["alan_class"])

    city_avg = cities_t["annual_pct_change"].mean()
    site_avg = sites_t["annual_pct_change"].mean()
    log.info("\nCities avg growth: %+.2f%%/yr", city_avg)
    log.info("Sites avg growth:  %+.2f%%/yr", site_avg)

    low_alan = sites_t[sites_t["alan_class"] == "low"]
    log.info("\nDark-sky viable (median < 1 nW in %d): %d of %d sites",
             max(years), len(low_alan), len(sites_t))
    if len(low_alan) > 0:
        log.info("  %s", ", ".join(low_alan["name"].tolist()))

    # Generate maps (using latest year for static maps)
    latest_year = max(years)
    latest_subset_dir = os.path.join(args.output_dir, "subsets", str(latest_year))
    latest_metrics = yearly_df[yearly_df["year"] == latest_year]
    district_gdf = gpd.read_file(args.shapefile_path)
    generate_site_maps(gdf_sites, latest_metrics, district_gdf, latest_subset_dir,
                       args.output_dir, latest_year)

    # Generate time-series plots
    generate_site_timeseries(yearly_df, args.output_dir)

    # ── Spatial analysis enhancements ─────────────────────────────────
    csv_dir = os.path.join(args.output_dir, "csv")
    maps_dir = os.path.join(args.output_dir, "maps")
    median_path = os.path.join(latest_subset_dir, f"maharashtra_median_{latest_year}.tif")

    if os.path.exists(median_path):
        # Task 2.2: Inside vs outside buffer comparison
        from src.buffer_comparison import (compare_inside_outside_buffers,
                                           plot_inside_outside_comparison)
        buffer_comparison = compare_inside_outside_buffers(
            site_gdf=gdf_sites,
            raster_path=median_path,
            output_csv=os.path.join(csv_dir, f"site_buffer_comparison_{latest_year}.csv"),
        )
        plot_inside_outside_comparison(
            buffer_comparison,
            output_path=os.path.join(maps_dir, "site_buffer_comparison.png"),
        )

        # Task 2.3: Directional brightness analysis
        from src.directional_analysis import (compute_directional_brightness,
                                              plot_directional_polar)
        directional = compute_directional_brightness(
            raster_path=median_path,
            output_csv=os.path.join(csv_dir, f"directional_brightness_{latest_year}.csv"),
        )
        plot_directional_polar(
            directional,
            output_path=os.path.join(maps_dir, "directional_brightness_polar.pdf"),
        )

    # Task 2.4: Nearest city distance metrics
    from src.proximity_analysis import compute_nearest_city_distances
    proximity = compute_nearest_city_distances(
        output_csv=os.path.join(csv_dir, "site_proximity_metrics.csv"),
    )

    # ── Sky brightness estimation (Task 7.1) ─────────────────────────
    from src.sky_brightness_model import (
        compute_sky_brightness_metrics,
        plot_sky_brightness_distribution,
    )

    sky_metrics = compute_sky_brightness_metrics(
        latest_metrics,
        output_csv=os.path.join(csv_dir, f"sky_brightness_{latest_year}.csv"),
    )
    plot_sky_brightness_distribution(
        sky_metrics,
        output_path=os.path.join(maps_dir, "sky_brightness_distribution.png"),
    )

    # ── Temporal stability for sites (Task 3.1) ──────────────────────
    from src.stability_metrics import compute_stability_metrics, plot_stability_scatter

    site_stability = compute_stability_metrics(
        yearly_df, entity_col="name",
        output_csv=os.path.join(csv_dir, "site_stability_metrics.csv"),
    )
    plot_stability_scatter(
        site_stability, entity_col="name",
        output_path=os.path.join(maps_dir, "site_stability_scatter.png"),
    )

    # ── Breakpoint detection for sites (Task 3.2) ────────────────────
    from src.breakpoint_analysis import analyze_all_breakpoints

    site_breakpoints = analyze_all_breakpoints(
        yearly_df, entity_col="name",
        output_csv=os.path.join(csv_dir, "site_breakpoints.csv"),
    )

    # ── Benchmark comparison for sites (Task 4.2) ────────────────────
    from src.benchmark_comparison import compare_to_benchmarks

    compare_to_benchmarks(
        trends_df,
        output_csv=os.path.join(csv_dir, "site_benchmark_comparison.csv"),
    )

    # ── City vs site boxplot (Task 5.3) ──────────────────────────────
    from src.visualization_suite import create_city_vs_site_boxplot

    create_city_vs_site_boxplot(
        latest_metrics,
        output_path=os.path.join(maps_dir, "city_vs_site_boxplot.png"),
    )

    # ── Site-level deep-dive reports (Task 5.2) ──────────────────────
    from src.site_reports import generate_all_site_reports

    generate_all_site_reports(
        all_site_data=latest_metrics,
        yearly_df=yearly_df,
        output_dir=os.path.join(args.output_dir, config.OUTPUT_DIRS["site_reports"]),
    )

    log.info("\nDone! Outputs in %s/", args.output_dir)


if __name__ == "__main__":
    main()
