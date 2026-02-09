#!/usr/bin/env python3
"""
Site-level ALAN analysis for 5 cities and 11 dark-sky candidate sites.

Uses 2024 Maharashtra VIIRS subsets (already produced by viirs_process.py)
to compute radiance metrics at sub-district resolution via 10 km buffers
around point coordinates.

Usage:
    python src/site_analysis.py [--output-dir ./outputs] [--buffer-km 10]
                                [--cf-threshold 5]
"""

import argparse
import logging
import os

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Location data ────────────────────────────────────────────────────────
# (lat, lon, type, district)
LOCATIONS = {
    # Cities (urban high-ALAN benchmarks)
    "Mumbai":           (18.9600, 72.8200, "city", "Mumbai City"),
    "Pune":             (18.5167, 73.8554, "city", "Pune"),
    "Nagpur":           (21.1466, 79.0889, "city", "Nagpur"),
    "Thane":            (19.2183, 72.9781, "city", "Thane"),
    "Pimpri-Chinchwad": (18.6278, 73.8131, "city", "Pune"),

    # Dark-sky candidate sites
    "Lonar Crater":          (19.9761, 76.5079, "site", "Buldhana"),
    "Tadoba Tiger Reserve":  (20.2485, 79.4254, "site", "Chandrapur"),
    "Pench Tiger Reserve":   (21.6900, 79.2300, "site", "Nagpur"),
    "Udmal Tribal Village":  (20.5300, 73.3900, "site", "Nashik"),
    "Kaas Plateau":          (17.7200, 73.8228, "site", "Satara"),
    "Toranmal":              (21.7333, 74.4167, "site", "Nandurbar"),
    "Bhandardara":           (19.5375, 73.7695, "site", "Ahmednagar"),
    "Harihareshwar":         (17.9942, 73.0258, "site", "Raigad"),
    "Yawal Wildlife Sanctuary": (21.3781, 75.8750, "site", "Jalgaon"),
    "Melghat Tiger Reserve": (21.4458, 77.1972, "site", "Amravati"),
    "Bhimashankar":          (19.0739, 73.5352, "site", "Pune"),
}


def build_site_geodataframe(buffer_km=10):
    """Create GeoDataFrame with 10 km circular buffers around each site."""
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


def compute_site_metrics(gdf, subset_dir, year=2024, cf_threshold=5):
    """Compute filtered radiance stats for each site buffer."""
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
    tmp_path = "/tmp/_viirs_site_filtered.tif"
    meta = {
        "driver": "GTiff", "height": filtered.shape[0], "width": filtered.shape[1],
        "count": 1, "dtype": "float32", "crs": crs, "transform": transform,
        "nodata": np.nan,
    }
    with rasterio.open(tmp_path, "w", **meta) as dst:
        dst.write(filtered, 1)

    # Also write unfiltered for pixel count comparison
    tmp_unfilt = "/tmp/_viirs_site_unfiltered.tif"
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

    os.remove(tmp_path)
    os.remove(tmp_unfilt)

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
        bins=[-np.inf, 1.0, 5.0, np.inf],
        labels=["low", "medium", "high"],
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
    fig.savefig(path, dpi=300, bbox_inches="tight")
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
    ax.axvline(x=1.0, color="orange", linestyle="--", linewidth=1, label="Low-ALAN threshold (1 nW)")

    for bar, val in zip(bars, df_sorted["median_radiance"]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)

    city_patch = mpatches.Patch(color="crimson", label="Cities")
    site_patch = mpatches.Patch(color="forestgreen", label="Dark-sky sites")
    ax.legend(handles=[city_patch, site_patch, ax.lines[0]], loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(maps_dir, "site_comparison_chart.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
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
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)


def main():
    parser = argparse.ArgumentParser(
        description="Site-level ALAN analysis (5 cities + 11 dark-sky sites)"
    )
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--shapefile-path",
                        default="./data/shapefiles/maharashtra_district.shp")
    parser.add_argument("--buffer-km", type=float, default=10)
    parser.add_argument("--cf-threshold", type=int, default=5)
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()

    subset_dir = os.path.join(args.output_dir, "subsets", str(args.year))
    log.info("Site-level ALAN analysis (%d)", args.year)
    log.info("Buffer radius: %d km", args.buffer_km)

    # Build site buffers
    gdf_sites = build_site_geodataframe(args.buffer_km)

    # Compute metrics
    metrics_df = compute_site_metrics(gdf_sites, subset_dir, args.year,
                                      args.cf_threshold)
    if metrics_df is None:
        log.error("Failed to compute metrics. Run viirs_process.py --years %d first.",
                  args.year)
        return

    # Save CSV
    csv_dir = os.path.join(args.output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"site_metrics_{args.year}.csv")
    metrics_df.to_csv(csv_path, index=False)
    log.info("Saved: %s", csv_path)

    # Print summary
    log.info("\n" + "=" * 70)
    log.info("SITE METRICS SUMMARY (%d)", args.year)
    log.info("=" * 70)

    cities_df = metrics_df[metrics_df["type"] == "city"].sort_values(
        "median_radiance", ascending=False)
    sites_df = metrics_df[metrics_df["type"] == "site"].sort_values(
        "median_radiance", ascending=True)

    log.info("\n--- CITIES (urban benchmarks) ---")
    for _, r in cities_df.iterrows():
        log.info("  %-20s median=%.2f  mean=%.2f  pixels=%d  quality=%.0f%%  [%s]",
                 r["name"], r["median_radiance"], r["mean_radiance"],
                 r["valid_pixels"], r["quality_pct"], r["alan_class"])

    log.info("\n--- DARK-SKY SITES ---")
    for _, r in sites_df.iterrows():
        log.info("  %-30s median=%.2f  mean=%.2f  pixels=%d  quality=%.0f%%  [%s]",
                 r["name"], r["median_radiance"], r["mean_radiance"],
                 r["valid_pixels"], r["quality_pct"], r["alan_class"])

    city_avg = cities_df["median_radiance"].mean()
    site_avg = sites_df["median_radiance"].mean()
    log.info("\nCities average: %.2f nW/cm²/sr", city_avg)
    log.info("Sites average:  %.2f nW/cm²/sr", site_avg)
    log.info("Ratio: %.1fx", city_avg / site_avg if site_avg > 0 else float("inf"))

    low_alan = sites_df[sites_df["alan_class"] == "low"]
    log.info("\nDark-sky viable (median < 1 nW): %d of %d sites",
             len(low_alan), len(sites_df))
    if len(low_alan) > 0:
        log.info("  %s", ", ".join(low_alan["name"].tolist()))

    # Generate maps
    district_gdf = gpd.read_file(args.shapefile_path)
    generate_site_maps(gdf_sites, metrics_df, district_gdf, subset_dir,
                       args.output_dir, args.year)

    log.info("\nDone! Outputs in %s/", args.output_dir)


if __name__ == "__main__":
    main()
