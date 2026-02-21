#!/usr/bin/env python3
"""
Site-level ALAN analysis for 5 cities and 11 dark-sky candidate sites.

Uses Maharashtra VIIRS subsets (already produced by viirs_process.py)
to compute radiance metrics at sub-district resolution via 10 km buffers
around point coordinates, then fits log-linear trends over the year range.

Usage (run from project root):
    python3 -m src.site.site_analysis [--output-dir ./outputs] [--buffer-km 10]
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
from src import viirs_utils
from src.formulas.trend import fit_log_linear_trend as _core_fit_trend
from src.formulas.classification import classify_alan, classify_alan_series

log = get_pipeline_logger(__name__)

# Threshold for switching between individual-item and aggregate visualizations.
# <= MANY_THRESHOLD: show per-location charts (original behavior for 43 config cities)
# > MANY_THRESHOLD: show distribution/aggregate charts (for 629 census towns)
MANY_THRESHOLD = 50

# ALAN classification colors for point markers
ALAN_COLORS = {"high": "#d62728", "medium": "#ff7f0e", "low": "#2ca02c"}


def _build_locations():
    """Build LOCATIONS dict from config.URBAN_BENCHMARKS and config.DARKSKY_SITES."""
    locations = {}
    for name, info in config.URBAN_BENCHMARKS.items():
        locations[name] = (info["lat"], info["lon"], "city", info["district"])
    for name, info in config.DARKSKY_SITES.items():
        locations[name] = (info["lat"], info["lon"], "site", info["district"])
    return locations


def _build_locations_filtered(entity_type="all", city_source="config"):
    """Build LOCATIONS dict, optionally filtered by entity type.

    Parameters
    ----------
    entity_type : str
        "city", "site", or "all".
    city_source : str
        "config" uses config.URBAN_BENCHMARKS (43 hand-picked cities).
        "census" uses geocoded census towns (up to 629).
    """
    locations = {}
    if entity_type in ("city", "all"):
        if city_source == "census":
            from src.census_town_locations import load_census_town_locations
            towns = load_census_town_locations()
            for name, info in towns.items():
                locations[name] = (info["lat"], info["lon"], "city", info["district"])
            log.info("Loaded %d census towns as city locations", len(towns))
        else:
            for name, info in config.URBAN_BENCHMARKS.items():
                locations[name] = (info["lat"], info["lon"], "city", info["district"])
    if entity_type in ("site", "all"):
        for name, info in config.DARKSKY_SITES.items():
            locations[name] = (info["lat"], info["lon"], "site", info["district"])
    return locations


LOCATIONS = _build_locations()


def build_site_geodataframe(buffer_km=None, entity_type="all", city_source="config"):
    """Create GeoDataFrame with circular buffers around each site.

    BUFFER ANALYSIS methodology:
    Following Wang et al. (2022), a 10 km radius buffer around point sites
    captures the local ALAN environment while being large enough to contain
    sufficient VIIRS pixels (~440 pixels at 450 m resolution) for robust
    statistics. The buffer is computed in UTM Zone 43N (EPSG:32643) for
    metric accuracy, then reprojected to WGS84 for raster extraction.

    LAND CLIPPING: Buffers for coastal sites (e.g., Mumbai, Ratnagiri) are
    clipped to the Maharashtra land boundary (union of district polygons) to
    exclude ocean pixels that would otherwise dilute radiance statistics
    with near-zero background noise. This is essential for sites within
    buffer_km of the coastline.

    Citation: Wang, J. et al. (2022). Protected area buffer analysis.
    """
    if buffer_km is None:
        buffer_km = config.SITE_BUFFER_RADIUS_KM
    locations = _build_locations_filtered(entity_type, city_source=city_source)
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

    # Clip buffers to land boundary (Maharashtra district union)
    land_boundary = _get_land_boundary_utm()
    if land_boundary is not None:
        original_areas = gdf_utm.geometry.area.copy()
        gdf_utm["geometry"] = gdf_utm.geometry.intersection(land_boundary)
        clipped_areas = gdf_utm.geometry.area
        for idx, row in gdf_utm.iterrows():
            pct = clipped_areas[idx] / original_areas[idx] * 100 if original_areas[idx] > 0 else 100
            if pct < 99:
                log.info("  Land-clipped '%s': %.0f%% of buffer is land (%.0f → %.0f km²)",
                         row["name"], pct,
                         original_areas[idx] / 1e6, clipped_areas[idx] / 1e6)

    gdf_buffered = gdf_utm.to_crs(epsg=4326)

    log.info("Created %d site buffers (%d km radius, land-clipped)", len(gdf_buffered), buffer_km)
    for _, row in gdf_buffered.iterrows():
        area_km2 = gdf_utm[gdf_utm["name"] == row["name"]].geometry.area.values[0] / 1e6
        log.info("  %-30s %s  %.0f km²  (%.4f°N, %.4f°E)",
                 row["name"], row["type"], area_km2, row["lat"], row["lon"])

    return gdf_buffered


def _get_land_boundary_utm():
    """Load Maharashtra district shapefile and return union in UTM 43N.

    Returns None if shapefile is not available (graceful fallback).
    """
    shapefile_path = config.DEFAULT_SHAPEFILE_PATH
    if not os.path.exists(shapefile_path):
        log.warning("Shapefile not found at %s — skipping land clipping", shapefile_path)
        return None
    try:
        districts = gpd.read_file(shapefile_path)
        districts_utm = districts.to_crs(epsg=32643)
        land_union = districts_utm.union_all()
        log.info("Loaded land boundary from %s", shapefile_path)
        return land_union
    except Exception as exc:
        log.warning("Could not load land boundary: %s — skipping land clipping", exc)
        return None


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

    # NOTE: DBS (Dynamic Background Subtraction) is intentionally NOT applied
    # here. Site metrics use RAW radiance with quality filtering (lit_mask +
    # cf_cvg) as the correct approach. DBS computes a single P1.0 floor from
    # the entire state raster, which over-subtracts from dark sites and
    # under-subtracts from bright ones. DBS is reserved for visualization
    # paths only (gradient analysis, annual map frames).

    valid = np.isfinite(median_data) & (lit_data > 0) & (cf_data >= cf_threshold)
    total_valid = valid.sum()

    filtered = np.where(valid, median_data, np.nan)
    log.info("Quality filter: %d pixels pass (lit_mask & cf_cvg >= %d)",
             total_valid, cf_threshold)

    # Use numpy array + affine approach for zonal_stats to avoid GDAL
    # environment corruption after large raster operations (unpack/subset
    # of ~11 GB global composites). See compute_district_stats() note.
    results_filt = zonal_stats(
        gdf, filtered.astype("float32"),
        stats=["mean", "median", "count", "min", "max", "std"],
        nodata=np.nan, all_touched=True,
        affine=transform,
    )

    unfilt = np.where(np.isfinite(median_data), median_data, np.nan)
    results_unfilt = zonal_stats(
        gdf, unfilt.astype("float32"),
        stats=["count"],
        nodata=np.nan, all_touched=True,
        affine=transform,
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
    """Generate maps overlaying analysis locations on Maharashtra with radiance data.

    Renders maps appropriate for the entity type present in the data
    (cities or dark-sky sites — each processed independently).

    For cities: the state-level map shows only the top 20 by radiance
    for legibility, and per-district maps show all cities in each district.
    """
    maps_dir = os.path.join(output_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)

    merged = gdf_sites.merge(metrics_df[["name", "median_radiance", "alan_class"]],
                             on="name")

    # Determine entity type for labelling
    entity_types = merged["type"].unique()
    is_city = "city" in entity_types
    entity_label = "Cities" if is_city else "Dark-Sky Sites"
    entity_color = "crimson" if is_city else "forestgreen"
    entity_edge = "darkred" if is_city else "darkgreen"
    outline_color = "cyan" if is_city else "lime"

    # For city state map: limit to top 20 by radiance for legibility
    if is_city and len(merged) > 20:
        top20 = merged.nlargest(20, "median_radiance")
        state_label = f"Top 20 {entity_label} by Radiance"
    else:
        top20 = merged
        state_label = entity_label

    # ── Map 1: State overlay ────────────────────────────────────────
    many = len(merged) > MANY_THRESHOLD
    fig, ax = plt.subplots(figsize=(14, 11))
    district_gdf.plot(ax=ax, color="whitesmoke", edgecolor="grey", linewidth=0.5)

    if many:
        # Scatter dots colored by radiance for many-town case
        cx = merged.geometry.centroid.x
        cy = merged.geometry.centroid.y
        rad = merged["median_radiance"].fillna(0).values
        sc = ax.scatter(cx, cy, c=rad, cmap="YlOrRd", s=np.clip(rad * 3, 5, 80),
                        edgecolors="black", linewidths=0.3, alpha=0.8, zorder=5,
                        vmin=0, vmax=max(rad.max(), 1))
        plt.colorbar(sc, ax=ax, shrink=0.5, label="Median Radiance (nW/cm²/sr)")

        # Label top 10 by radiance
        label_set = merged.nlargest(10, "median_radiance")
        texts = []
        for _, row in label_set.iterrows():
            c = row.geometry.centroid
            txt = ax.text(c.x, c.y, f"{row['name']}\n{row['median_radiance']:.1f}",
                          fontsize=5.5, fontweight="bold", ha="center", va="center",
                          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="grey",
                                    alpha=0.85, linewidth=0.4))
            texts.append(txt)
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.4),
                        expand=(1.3, 1.5), force_text=(0.5, 0.8))
        except ImportError:
            pass
        state_label = f"{len(merged)} {entity_label}"
    else:
        # Original buffer overlay for small counts
        top20.plot(ax=ax, color=entity_color, alpha=0.4, edgecolor=entity_edge,
                   linewidth=1.2)
        texts = []
        for _, row in top20.iterrows():
            centroid = row.geometry.centroid
            txt = ax.text(
                centroid.x, centroid.y,
                f"{row['name']}\n{row['median_radiance']:.1f}",
                fontsize=5.5, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="grey",
                          alpha=0.85, linewidth=0.4),
            )
            texts.append(txt)
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.4),
                        expand=(1.3, 1.5), force_text=(0.5, 0.8))
        except ImportError:
            pass
        ax.legend(
            handles=[mpatches.Patch(color=entity_color, alpha=0.4,
                                    label=f"{state_label} (10 km buffer)")],
            loc="lower left", fontsize=9,
        )

    ax.set_title(f"Maharashtra: {state_label} ALAN Analysis ({year})", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    path = os.path.join(maps_dir, "overlay_map.png")
    fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)

    # ── Map 2: Radiance chart ──────────────────────────────────────
    df_sorted = metrics_df.sort_values("median_radiance", ascending=True)

    if len(df_sorted) > MANY_THRESHOLD:
        # 3-panel distribution figure for many towns
        fig, axes = plt.subplots(1, 3, figsize=(18, 8),
                                 gridspec_kw={"width_ratios": [2, 2, 1]})

        # Panel 1: Histogram of radiance (log-x scale)
        ax1 = axes[0]
        rad_vals = df_sorted["median_radiance"].dropna()
        rad_vals = rad_vals[rad_vals > 0]
        ax1.hist(rad_vals, bins=30, color=entity_color, edgecolor="grey",
                 linewidth=0.5, alpha=0.8)
        ax1.set_xscale("log")
        ax1.axvline(x=config.ALAN_LOW_THRESHOLD, color="green", linestyle="--",
                     linewidth=1.5, label=f"Low ({config.ALAN_LOW_THRESHOLD} nW)")
        ax1.axvline(x=config.ALAN_MEDIUM_THRESHOLD, color="orange", linestyle="--",
                     linewidth=1.5, label=f"Medium ({config.ALAN_MEDIUM_THRESHOLD} nW)")
        ax1.set_xlabel("Median Radiance (nW/cm²/sr)", fontsize=11)
        ax1.set_ylabel("Number of Towns", fontsize=11)
        ax1.set_title("Radiance Distribution", fontsize=12)
        ax1.legend(fontsize=8)
        ax1.grid(axis="y", alpha=0.3)

        # Panel 2: Box plot by district (top 15 districts by town count)
        ax2 = axes[1]
        district_counts = df_sorted["district"].value_counts()
        top_districts = district_counts.head(15).index.tolist()
        subset = df_sorted[df_sorted["district"].isin(top_districts)]
        # Order districts by median radiance
        district_order = (subset.groupby("district")["median_radiance"]
                          .median().sort_values().index.tolist())
        box_data = [subset[subset["district"] == d]["median_radiance"].dropna().values
                    for d in district_order]
        bp = ax2.boxplot(box_data, vert=False, labels=district_order, patch_artist=True,
                         widths=0.6, medianprops=dict(color="black", linewidth=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(entity_color)
            patch.set_alpha(0.6)
        ax2.axvline(x=config.ALAN_LOW_THRESHOLD, color="green", linestyle="--",
                     linewidth=1, alpha=0.7)
        ax2.axvline(x=config.ALAN_MEDIUM_THRESHOLD, color="orange", linestyle="--",
                     linewidth=1, alpha=0.7)
        ax2.set_xlabel("Median Radiance (nW/cm²/sr)", fontsize=11)
        ax2.set_title(f"By District (top {len(district_order)})", fontsize=12)
        ax2.grid(axis="x", alpha=0.3)

        # Panel 3: ALAN classification pie chart
        ax3 = axes[2]
        if "alan_class" in df_sorted.columns:
            class_counts = df_sorted["alan_class"].value_counts()
        else:
            class_counts = pd.Series(classify_alan_series(
                df_sorted["median_radiance"])).value_counts()
        pie_colors = [ALAN_COLORS.get(c, "#999999") for c in class_counts.index]
        wedges, texts_pie, autotexts = ax3.pie(
            class_counts.values, labels=class_counts.index,
            colors=pie_colors, autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 10})
        for at in autotexts:
            at.set_fontweight("bold")
        ax3.set_title("ALAN Classification", fontsize=12)

        fig.suptitle(f"ALAN Radiance: {len(df_sorted)} {entity_label} ({year})",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
    else:
        # Original horizontal bar chart for small counts
        fig, ax = plt.subplots(figsize=(12, max(8, len(df_sorted) * 0.3)))
        bars = ax.barh(df_sorted["name"], df_sorted["median_radiance"],
                       color=entity_color, edgecolor="grey", linewidth=0.5)
        ax.set_xlabel("Median Radiance (nW/cm²/sr)", fontsize=12)
        ax.set_title(f"ALAN Radiance: {entity_label} ({year})", fontsize=14)
        ax.axvline(x=config.ALAN_LOW_THRESHOLD, color="orange", linestyle="--", linewidth=1,
                   label=f"Low-ALAN threshold ({config.ALAN_LOW_THRESHOLD} nW)")
        for bar, val in zip(bars, df_sorted["median_radiance"]):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=8)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

    path = os.path.join(maps_dir, "radiance_chart.png")
    fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)

    # ── Map 3: Radiance raster with overlay (top 20 for cities) ───────
    median_path = os.path.join(subset_dir, f"maharashtra_median_{year}.tif")
    with rasterio.open(median_path) as src:
        raster = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    fig, ax = plt.subplots(figsize=(14, 11))
    raster_log = np.log10(np.clip(raster, 0.01, None))
    raster_log = np.where(np.isfinite(raster_log), raster_log, np.nan)
    im = ax.imshow(raster_log, extent=extent, cmap="magma", vmin=-2, vmax=2,
                   origin="upper", aspect="auto")
    district_gdf.boundary.plot(ax=ax, edgecolor="white", linewidth=0.3, alpha=0.5)

    if many:
        # Small scatter dots on raster for many-town case
        cx = top20.geometry.centroid.x
        cy = top20.geometry.centroid.y
        ax.scatter(cx, cy, s=15, c="cyan", edgecolors="white",
                   linewidths=0.3, alpha=0.9, zorder=5)
    else:
        top20.plot(ax=ax, facecolor="none", edgecolor=outline_color, linewidth=1.5)

    for _, row in top20.iterrows():
        c = row.geometry.centroid
        ax.annotate(row["name"], xy=(c.x, c.y), fontsize=5, color="white",
                    ha="center", va="bottom", fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.6, label="log₁₀(Radiance nW/cm²/sr)")
    ax.set_title(f"Maharashtra: Nighttime Radiance — {state_label} ({year})",
                 fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    path = os.path.join(maps_dir, "radiance_overlay.png")
    fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)

    # ── Per-district maps (cities only) ───────────────────────────────
    if is_city and "district" in merged.columns:
        _generate_per_district_maps(merged, district_gdf, maps_dir, year)


def _generate_per_district_maps(merged, district_gdf, maps_dir, year):
    """Generate a map for each district showing all its cities/towns.

    Uses ALAN-class-colored point markers (not buffer polygons) so that
    markers remain proportional to the district area regardless of how
    many towns are present.
    """
    district_maps_dir = os.path.join(maps_dir, "districts")
    os.makedirs(district_maps_dir, exist_ok=True)

    districts_with_cities = merged["district"].unique()
    log.info("Generating per-district maps for %d districts...", len(districts_with_cities))

    for district_name in sorted(districts_with_cities):
        district_cities = merged[merged["district"] == district_name].copy()
        if district_cities.empty:
            continue

        # Get district boundary
        district_boundary = district_gdf[
            district_gdf["district"].str.lower() == district_name.lower()
        ]
        if district_boundary.empty:
            log.warning("District '%s' not found in shapefile — skipping", district_name)
            continue

        fig, ax = plt.subplots(figsize=(10, 9))
        district_boundary.plot(ax=ax, color="whitesmoke", edgecolor="grey", linewidth=1)

        # Plot point markers colored by ALAN class, sized by radiance
        centroids_x = district_cities.geometry.centroid.x
        centroids_y = district_cities.geometry.centroid.y
        colors = district_cities["alan_class"].map(ALAN_COLORS).fillna("#999999")

        radiance_vals = district_cities["median_radiance"].fillna(0).values
        rad_min, rad_max = radiance_vals.min(), radiance_vals.max()
        if rad_max > rad_min:
            sizes = 30 + (radiance_vals - rad_min) / (rad_max - rad_min) * 170
        else:
            sizes = np.full(len(radiance_vals), 80)

        ax.scatter(centroids_x, centroids_y, c=colors.values, s=sizes,
                   edgecolors="black", linewidths=0.5, alpha=0.8, zorder=5)

        # Label only top 5 by radiance to avoid clutter
        top_n = district_cities.nlargest(min(5, len(district_cities)), "median_radiance")
        for _, row in top_n.iterrows():
            centroid = row.geometry.centroid
            ax.annotate(
                f"{row['name']}\n{row['median_radiance']:.1f} nW",
                xy=(centroid.x, centroid.y), fontsize=6.5, fontweight="bold",
                ha="center", va="bottom",
                xytext=(0, 8), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="grey",
                          alpha=0.85, linewidth=0.4),
            )

        # Legend for ALAN classes
        legend_handles = [
            mpatches.Patch(color=ALAN_COLORS["high"], label=f"High (>{config.ALAN_MEDIUM_THRESHOLD} nW)"),
            mpatches.Patch(color=ALAN_COLORS["medium"], label=f"Medium ({config.ALAN_LOW_THRESHOLD}–{config.ALAN_MEDIUM_THRESHOLD} nW)"),
            mpatches.Patch(color=ALAN_COLORS["low"], label=f"Low (<{config.ALAN_LOW_THRESHOLD} nW)"),
        ]
        ax.legend(handles=legend_handles, loc="lower left", fontsize=8,
                  title="ALAN Class", title_fontsize=9)

        n_towns = len(district_cities)
        ax.set_title(f"{district_name} District: {n_towns} Towns ALAN ({year})", fontsize=13)
        ax.set_axis_off()
        plt.tight_layout()

        safe_name = district_name.lower().replace(" ", "_")
        path = os.path.join(district_maps_dir, f"{safe_name}_cities.png")
        fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight")
        plt.close(fig)

    log.info("Saved %d district maps in %s", len(districts_with_cities), district_maps_dir)


def fit_site_trends(yearly_df):
    """Fit log-linear OLS trends per site, with bootstrap CI.

    Uses src.formulas.trend.fit_log_linear_trend() for core computation.
    Sites with low data quality (median quality_pct < 5% or median
    valid_pixels < 30) are flagged as unreliable — their trends are
    still computed but marked with quality_flag='low_quality'.

    Returns DataFrame with annual_pct_change, ci_low, ci_high, r_squared,
    p_value, quality_flag.
    """
    # Quality gate thresholds
    MIN_QUALITY_PCT = 5.0    # Minimum median quality percentage across years
    MIN_VALID_PIXELS = 30    # Minimum median valid pixel count across years

    results = []
    for name in yearly_df["name"].unique():
        sub = yearly_df[yearly_df["name"] == name].sort_values("year")
        loc_type = sub["type"].iloc[0]
        district = sub["district"].iloc[0]

        # Compute quality metrics for this site
        med_quality_pct = sub["quality_pct"].median() if "quality_pct" in sub.columns else 100.0
        med_valid_pixels = sub["valid_pixels"].median() if "valid_pixels" in sub.columns else 999

        quality_flag = "ok"
        if med_quality_pct < MIN_QUALITY_PCT or med_valid_pixels < MIN_VALID_PIXELS:
            quality_flag = "low_quality"
            log.warning(
                "Site '%s': low data quality (median quality=%.1f%%, median pixels=%d) "
                "— trend is unreliable",
                name, med_quality_pct, med_valid_pixels,
            )

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
            "median_quality_pct": round(med_quality_pct, 1),
            "median_valid_pixels": int(med_valid_pixels),
            "quality_flag": quality_flag,
        }

        # Add latest radiance if we had enough data
        if not np.isnan(core_result["annual_pct_change"]):
            latest = sub.iloc[-1]
            row["median_radiance_latest"] = latest["median_radiance"]
            row["mean_radiance_latest"] = latest["mean_radiance"]

        results.append(row)

    return pd.DataFrame(results)


def generate_site_timeseries(yearly_df, output_dir):
    """Generate time-series plot for the entity type present in yearly_df.

    For many towns (> MANY_THRESHOLD): 2-panel figure with district-level
    median lines (top 10) and state-wide IQR bands.
    For few locations: per-site line plot (original behavior).
    For dark-sky sites: adds ALAN threshold line.
    """
    maps_dir = os.path.join(output_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)

    entity_types = yearly_df["type"].unique()
    is_city = "city" in entity_types
    entity_label = "Cities" if is_city else "Dark-Sky Candidate Sites"
    prefix = "city" if is_city else "site"

    names = yearly_df["name"].unique()
    if len(names) == 0:
        log.info("Skipping timeseries — no data in yearly_df")
        return

    if len(names) > MANY_THRESHOLD:
        # 2-panel: district medians + state IQR bands
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

        # Panel 1: District-level median radiance over time (top 10)
        district_yearly = (yearly_df.groupby(["district", "year"])["median_radiance"]
                           .median().reset_index())
        latest_year = yearly_df["year"].max()
        latest_district = district_yearly[district_yearly["year"] == latest_year]
        top_districts = latest_district.nlargest(10, "median_radiance")["district"].tolist()

        cmap = plt.cm.tab10
        for i, dist in enumerate(top_districts):
            sub = district_yearly[district_yearly["district"] == dist].sort_values("year")
            ax1.plot(sub["year"], sub["median_radiance"], "o-", label=dist,
                     color=cmap(i), markersize=4, linewidth=1.5)

        ax1.set_ylabel("District Median Radiance\n(nW/cm²/sr)", fontsize=11)
        ax1.set_title(f"ALAN Time Series: Top 10 Districts by Radiance "
                      f"({len(names)} {entity_label})", fontsize=13)
        ax1.legend(fontsize=8, ncol=2, loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Panel 2: State-wide percentile bands
        year_stats = (yearly_df.groupby("year")["median_radiance"]
                      .agg(["median", lambda x: x.quantile(0.25),
                            lambda x: x.quantile(0.75),
                            lambda x: x.quantile(0.10),
                            lambda x: x.quantile(0.90)])
                      .reset_index())
        year_stats.columns = ["year", "p50", "p25", "p75", "p10", "p90"]

        ax2.fill_between(year_stats["year"], year_stats["p10"], year_stats["p90"],
                         alpha=0.15, color="steelblue", label="P10–P90")
        ax2.fill_between(year_stats["year"], year_stats["p25"], year_stats["p75"],
                         alpha=0.3, color="steelblue", label="P25–P75 (IQR)")
        ax2.plot(year_stats["year"], year_stats["p50"], "o-", color="darkblue",
                 linewidth=2, markersize=5, label="Median")

        ax2.set_xlabel("Year", fontsize=11)
        ax2.set_ylabel("Radiance (nW/cm²/sr)", fontsize=11)
        ax2.set_title(f"State-wide Radiance Distribution Over Time "
                      f"({len(names)} towns)", fontsize=13)
        ax2.legend(fontsize=9, loc="upper left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
    else:
        # Original per-site line plot for small counts
        if is_city and len(names) > 15:
            latest_year = yearly_df["year"].max()
            latest = yearly_df[yearly_df["year"] == latest_year]
            plot_names = latest.nlargest(15, "median_radiance")["name"].tolist()
            title_suffix = f" (top {len(plot_names)} by radiance)"
        else:
            plot_names = list(names)
            title_suffix = ""

        fig, ax = plt.subplots(figsize=(14, 8))
        for name in plot_names:
            sub = yearly_df[yearly_df["name"] == name].sort_values("year")
            ax.plot(sub["year"], sub["median_radiance"], "o-", label=name, markersize=4)

        if not is_city:
            ax.axhline(y=config.ALAN_LOW_THRESHOLD, color="orange", linestyle="--", linewidth=1,
                       label=f"Low-ALAN threshold ({config.ALAN_LOW_THRESHOLD} nW)")

        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Median Radiance (nW/cm²/sr)", fontsize=12)
        ax.set_title(f"ALAN Time Series: {entity_label}{title_suffix}", fontsize=14)
        ax.legend(fontsize=7 if not is_city else 8, ncol=2, loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    path = os.path.join(maps_dir, f"{prefix}_timeseries.png")
    fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)


def generate_annual_frames(years, output_dir, gdf_sites, district_gdf):
    """Generate annual map frames for animation with YoY statistics."""
    frame_dir = os.path.join(output_dir, "maps", "frames")
    os.makedirs(frame_dir, exist_ok=True)
    log.info("Generating annual frames in %s...", frame_dir)

    # Pre-calculate state statistics
    state_stats = {}
    for year in years:
        subset_dir = os.path.join(output_dir, "subsets", str(year))
        median_path = os.path.join(subset_dir, f"maharashtra_median_{year}.tif")
        
        if not os.path.exists(median_path):
            log.warning("Missing raster for %d, skipping frame", year)
            continue
            
        with rasterio.open(median_path) as src:
            data = src.read(1)
            # Apply DBS
            data_corrected = viirs_utils.apply_dynamic_background_subtraction(data, year=year)
            
            # Valid mask for statistics (finite and >0)
            valid_mask = np.isfinite(data_corrected) & (data_corrected > 0)
            if valid_mask.sum() > 0:
                # Re-calculate stats on corrected data
                median_val = np.median(data_corrected[valid_mask])
                state_stats[year] = median_val
            else:
                state_stats[year] = np.nan

    baseline_year = min(years)
    baseline_val = state_stats.get(baseline_year, np.nan)
    
    cities = gdf_sites[gdf_sites["type"] == "city"]
    sites = gdf_sites[gdf_sites["type"] == "site"]

    # Generate frame for each year
    for i, year in enumerate(years):
        if year not in state_stats:
            continue
            
        current_val = state_stats[year]
        prev_year = years[i-1] if i > 0 else None
        prev_val = state_stats.get(prev_year, np.nan)
        
        # Calculate changes
        growth_vs_base = ((current_val - baseline_val) / baseline_val * 100) if baseline_val else np.nan
        yoy_change = ((current_val - prev_val) / prev_val * 100) if prev_val else np.nan
        
        # Load raster for plotting
        subset_dir = os.path.join(output_dir, "subsets", str(year))
        median_path = os.path.join(subset_dir, f"maharashtra_median_{year}.tif")
        
        with rasterio.open(median_path) as src:
            raster = src.read(1)
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
            
        # Apply Dynamic Background Subtraction for display
        valid_mask = np.isfinite(raster) & (raster > 0)
        if valid_mask.sum() > 0:
            p01 = np.percentile(raster[valid_mask], 1)
            raster = np.maximum(0, raster - p01)
            
        # Plot
        fig, ax = plt.subplots(figsize=(14, 11))
        
        # Log-scale radiance
        raster_log = np.log10(np.clip(raster, 0.01, None))
        raster_log = np.where(np.isfinite(raster_log), raster_log, np.nan)
        
        im = ax.imshow(raster_log, extent=extent, cmap="magma", vmin=-2, vmax=2,
                       origin="upper", aspect="auto")
                       
        district_gdf.boundary.plot(ax=ax, edgecolor="white", linewidth=0.3, alpha=0.5)
        # sites/cities plotting removed for cleaner map
        
        # Annotations
        stats_text = (
            f"Year: {year}\n"
            f"State Median Radiance: {current_val:.2f} nW/cm²/sr\n"
            f"Growth vs {baseline_year}: {growth_vs_base:+.1f}%\n"
            f"(YoY: {yoy_change:+.1f}%)" if prev_year else f"Year: {year}\nState Median Radiance: {current_val:.2f} nW/cm²/sr\n(Baseline)"
        )
        
        # Add text box in bottom-right
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                fontsize=14, fontweight="bold", color="white",
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", fc="black", alpha=0.7, ec="white"))

        cbar = plt.colorbar(im, ax=ax, shrink=0.6, label="log₁₀(Radiance nW/cm²/sr)")
        ax.set_title(f"Maharashtra ALAN Evolution ({year})", fontsize=16)
        ax.set_axis_off()
        
        plt.tight_layout()
        path = os.path.join(frame_dir, f"frame_{year}.png")
        fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight", facecolor='black')
        plt.close(fig)
        log.info("Generated frame: %s", path)


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
    parser.add_argument("--city-source", choices=["config", "census"], default="config",
                        help="City locations source: 'config' (43 hand-picked) or 'census' (geocoded census towns)")
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
    log.info("Buffer radius: %d km | CF threshold: %d | Type: %s | City source: %s",
             args.buffer_km, args.cf_threshold, args.type, args.city_source)

    # Validate district names
    from src.validate_names import validate_or_exit
    validate_or_exit(args.shapefile_path, check_config=True)

    # Determine which entity types to process
    entity_types = [args.type] if args.type != "all" else ["city", "site"]

    for entity_type in entity_types:
        log.info("Running %s analysis...", entity_type)
        _run_entity_pipeline(args, years, entity_type, city_source=args.city_source)

    log.info("\nDone! Outputs in %s/", args.output_dir)


def _run_entity_pipeline(args, years, entity_type, city_source="config"):
    """Run the site/city analysis pipeline for a specific entity type."""
    # Import step functions
    from src.site.site_pipeline_steps import (
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
    result, gdf_sites = step_build_site_buffers(args.buffer_km, entity_type, city_source=city_source)
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

    # Step 11: Site-level deep-dive reports
    result, _ = step_site_reports(latest_metrics, yearly_df, reports_dir, entity_type)
    steps.append(result)
    if not result.ok:
        log.warning("Report generation failed: %s", result.error)


if __name__ == "__main__":
    main()
