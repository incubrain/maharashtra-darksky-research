"""
Protected area buffer comparison: inside vs. outside ALAN analysis.

Compares ALAN inside site boundaries vs. outside buffer zones to assess
light spillover pressure on protected areas.

METHODOLOGY:
Following Wang et al. (2022) protected area buffer methodology:
"A buffer zone comparison quantifies the difference in light pressure
between the interior of protected areas and surrounding developed land."
"""

from src.logging_config import get_pipeline_logger
import os

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tempfile
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import Point

from src import config
from src import viirs_utils

log = get_pipeline_logger(__name__)


def compare_inside_outside_buffers(site_gdf, raster_path, year=None, buffer_km=None,
                                   output_csv=None):
    """Compare ALAN inside site boundary vs. outside buffer zone.

    Args:
        site_gdf: GeoDataFrame with site polygons (buffered points).
        raster_path: Path to radiance raster.
        buffer_km: Distance for outside buffer (default: config value).
        output_csv: Path to save results.

    Returns:
        DataFrame with columns: [site, inside_mean, inside_median,
        outside_mean, outside_median, ratio, gradient].
    """
    if buffer_km is None:
        buffer_km = config.PROTECTED_AREA_BUFFER_KM

    results = []
    for _, row in site_gdf.iterrows():
        name = row["name"]
        inside_geom = row.geometry

        # Project to UTM for metric buffer
        inside_gdf = gpd.GeoDataFrame(
            [{"geometry": inside_geom}], crs="EPSG:4326"
        ).to_crs(epsg=config.MAHARASHTRA_UTM_EPSG)

        # Outside zone: buffer(inside + buffer_km) - inside
        outer = inside_gdf.geometry.iloc[0].buffer(buffer_km * 1000)
        outside_geom = outer.difference(inside_gdf.geometry.iloc[0])

        outside_gdf = gpd.GeoDataFrame(
            [{"geometry": outside_geom}],
            crs=f"EPSG:{config.MAHARASHTRA_UTM_EPSG}"
        ).to_crs("EPSG:4326")

        inside_wgs = gpd.GeoDataFrame(
            [{"geometry": inside_geom}], crs="EPSG:4326"
        )

        # Prepare corrected raster
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            meta = src.meta.copy()
            # Apply DBS
            data_corrected = viirs_utils.apply_dynamic_background_subtraction(data, year=year)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix="_buffer_corr.tif")
        os.close(tmp_fd)
        try:
            with rasterio.open(tmp_path, "w", **meta) as dst:
                dst.write(data_corrected, 1)

            # Zonal stats for inside
            stats_in = zonal_stats(
                inside_wgs, tmp_path,
                stats=["mean", "median", "count"],
                nodata=np.nan, all_touched=True,
            )
            # Zonal stats for outside
            stats_out = zonal_stats(
                outside_gdf, tmp_path,
                stats=["mean", "median", "count"],
                nodata=np.nan, all_touched=True,
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        in_med = stats_in[0].get("median", np.nan) if stats_in else np.nan
        out_med = stats_out[0].get("median", np.nan) if stats_out else np.nan
        in_mean = stats_in[0].get("mean", np.nan) if stats_in else np.nan
        out_mean = stats_out[0].get("mean", np.nan) if stats_out else np.nan

        ratio = out_med / in_med if in_med and in_med > 0 else np.nan
        gradient = (out_med - in_med) / buffer_km if not np.isnan(in_med) else np.nan

        results.append({
            "site": name,
            "inside_mean": round(in_mean, 4) if not np.isnan(in_mean) else np.nan,
            "inside_median": round(in_med, 4) if not np.isnan(in_med) else np.nan,
            "outside_mean": round(out_mean, 4) if not np.isnan(out_mean) else np.nan,
            "outside_median": round(out_med, 4) if not np.isnan(out_med) else np.nan,
            "ratio": round(ratio, 3) if not np.isnan(ratio) else np.nan,
            "gradient": round(gradient, 4) if not np.isnan(gradient) else np.nan,
        })

    df = pd.DataFrame(results)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info("Saved buffer comparison: %s", output_csv)

    return df


def plot_inside_outside_comparison(comparison_df, output_path):
    """Bar chart comparing inside vs. outside ALAN for each site."""
    df = comparison_df.dropna(subset=["inside_median", "outside_median"])
    df = df.sort_values("inside_median", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(df))
    width = 0.35

    bars_in = ax.barh(x - width / 2, df["inside_median"], width,
                      color="forestgreen", alpha=0.7, label="Inside buffer")
    bars_out = ax.barh(x + width / 2, df["outside_median"], width,
                       color="orange", alpha=0.7, label="Outside buffer")

    ax.axvline(x=config.ALAN_LOW_THRESHOLD, color="red", linestyle="--",
               linewidth=1, label=f"Low-ALAN threshold ({config.ALAN_LOW_THRESHOLD} nW)")

    ax.set_yticks(x)
    ax.set_yticklabels(df["site"], fontsize=9)
    ax.set_xlabel("Median Radiance (nW/cm²/sr)", fontsize=12)
    ax.set_title("ALAN Inside vs. Outside Site Buffers", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)
