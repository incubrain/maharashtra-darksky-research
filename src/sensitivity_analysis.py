"""
Automated sensitivity analysis for key parameters.

Tests how results change with different cf_threshold and buffer_km values.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats

from src import config

log = logging.getLogger(__name__)


def run_cf_threshold_sensitivity(subset_dir, gdf, year=2024,
                                 param_values=None, output_csv=None):
    """Test sensitivity to cloud-free coverage threshold.

    Args:
        subset_dir: Directory with subset rasters for the given year.
        gdf: District GeoDataFrame.
        year: Year to analyse.
        param_values: List of cf_threshold values to test.
        output_csv: Path to save results.

    Returns:
        DataFrame with results for each parameter value.
    """
    if param_values is None:
        param_values = [3, 5, 7, 10]

    median_path = os.path.join(subset_dir, f"maharashtra_median_{year}.tif")
    lit_path = os.path.join(subset_dir, f"maharashtra_lit_mask_{year}.tif")
    cf_path = os.path.join(subset_dir, f"maharashtra_cf_cvg_{year}.tif")

    for p in [median_path, lit_path, cf_path]:
        if not os.path.exists(p):
            log.error("Missing raster for sensitivity analysis: %s", p)
            return pd.DataFrame()

    with rasterio.open(median_path) as src:
        median_data = src.read(1).astype("float32")
        transform = src.transform
        crs = src.crs

    with rasterio.open(lit_path) as src:
        lit_data = src.read(1)

    with rasterio.open(cf_path) as src:
        cf_data = src.read(1)

    results = []
    for cf_val in param_values:
        valid = np.isfinite(median_data) & (lit_data > 0) & (cf_data >= cf_val)
        filtered = np.where(valid, median_data, np.nan)

        tmp_path = f"/tmp/_sensitivity_cf{cf_val}.tif"
        meta = {
            "driver": "GTiff", "height": filtered.shape[0],
            "width": filtered.shape[1], "count": 1, "dtype": "float32",
            "crs": crs, "transform": transform, "nodata": np.nan,
        }
        with rasterio.open(tmp_path, "w", **meta) as dst:
            dst.write(filtered.astype("float32"), 1)

        stats = zonal_stats(gdf, tmp_path,
                            stats=["mean", "median", "count"],
                            nodata=np.nan, all_touched=True)
        os.remove(tmp_path)

        for i, s in enumerate(stats):
            results.append({
                "cf_threshold": cf_val,
                "district": gdf["district"].iloc[i],
                "mean_radiance": s.get("mean"),
                "median_radiance": s.get("median"),
                "valid_pixels": s.get("count", 0),
            })

    df = pd.DataFrame(results)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info("Saved cf_threshold sensitivity: %s", output_csv)

    return df


def plot_sensitivity_results(sensitivity_df, param_name, output_path):
    """Multi-panel plot showing parameter sensitivity."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Mean radiance vs param value (aggregated across districts)
    ax = axes[0]
    agg = sensitivity_df.groupby(param_name).agg(
        mean_rad=("median_radiance", "mean"),
        std_rad=("median_radiance", "std"),
        total_pixels=("valid_pixels", "sum"),
    ).reset_index()
    ax.errorbar(agg[param_name], agg["mean_rad"], yerr=agg["std_rad"],
                fmt="o-", capsize=4, color="steelblue", linewidth=2)
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("Mean Median Radiance (nW/cm²/sr)", fontsize=12)
    ax.set_title(f"Radiance vs. {param_name}", fontsize=13)
    ax.grid(True, alpha=0.3)

    # Panel 2: Valid pixels vs param value
    ax = axes[1]
    ax.bar(agg[param_name].astype(str), agg["total_pixels"],
           color="steelblue", alpha=0.7)
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("Total Valid Pixels", fontsize=12)
    ax.set_title(f"Data Availability vs. {param_name}", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Sensitivity Analysis: {param_name}", fontsize=14, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)
