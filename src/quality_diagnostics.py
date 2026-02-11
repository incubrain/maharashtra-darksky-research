"""
Pixel-level quality diagnostics: per-district filter breakdown.

Details how many pixels pass/fail each quality filter (nodata, lit_mask,
cf_cvg) per district per year.
"""

import logging
import os

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats

from src import config
from src import viirs_utils

log = logging.getLogger(__name__)


def generate_quality_report(median_path, lit_path, cf_path, gdf, year,
                            output_csv=None):
    """Per-district breakdown of pixel filtering results.

    Args:
        median_path: Path to median radiance raster.
        lit_path: Path to lit_mask raster.
        cf_path: Path to cf_cvg raster.
        gdf: District GeoDataFrame.
        year: Year being processed.
        output_csv: Path to save report.

    Returns:
        DataFrame with filter breakdown per district.
    """
    with rasterio.open(median_path) as src:
        median_data = src.read(1).astype("float32")
        # Apply DBS
        median_data = viirs_utils.apply_dynamic_background_subtraction(median_data, year=year)
        transform = src.transform
        crs = src.crs

    lit_data = None
    if lit_path and os.path.exists(lit_path):
        with rasterio.open(lit_path) as src:
            lit_data = src.read(1)

    cf_data = None
    if cf_path and os.path.exists(cf_path):
        with rasterio.open(cf_path) as src:
            cf_data = src.read(1)

    # Create mask arrays for each filter
    mask_finite = np.isfinite(median_data).astype("float32")
    mask_lit = (lit_data > 0).astype("float32") if lit_data is not None else np.ones_like(median_data)
    mask_cf = (cf_data >= config.CF_COVERAGE_THRESHOLD).astype("float32") if cf_data is not None else np.ones_like(median_data)
    mask_all = (mask_finite * mask_lit * mask_cf)

    # Write temp rasters for zonal_stats
    tmp_dir = "/tmp/_quality_diag"
    os.makedirs(tmp_dir, exist_ok=True)
    meta = {
        "driver": "GTiff", "height": median_data.shape[0],
        "width": median_data.shape[1], "count": 1, "dtype": "float32",
        "crs": crs, "transform": transform, "nodata": np.nan,
    }

    layers = {
        "total": np.ones_like(median_data, dtype="float32"),
        "finite": mask_finite,
        "lit": mask_lit,
        "cf": mask_cf,
        "passed": mask_all,
    }

    stats_dict = {}
    for name, arr in layers.items():
        tmp_path = os.path.join(tmp_dir, f"_qd_{name}.tif")
        with rasterio.open(tmp_path, "w", **meta) as dst:
            dst.write(arr, 1)
        results = zonal_stats(gdf, tmp_path, stats=["sum", "count"],
                              nodata=np.nan, all_touched=True)
        stats_dict[name] = results
        os.remove(tmp_path)

    os.rmdir(tmp_dir)

    rows = []
    for i, district in enumerate(gdf["district"].values):
        total = stats_dict["total"][i].get("count", 0) or 0
        passed = int(stats_dict["passed"][i].get("sum", 0) or 0)
        finite = int(stats_dict["finite"][i].get("sum", 0) or 0)
        lit = int(stats_dict["lit"][i].get("sum", 0) or 0)
        cf = int(stats_dict["cf"][i].get("sum", 0) or 0)

        rows.append({
            "district": district,
            "year": year,
            "total_pixels": total,
            "passed_all_filters": passed,
            "failed_nodata": total - finite,
            "failed_lit_mask": total - lit,
            "failed_cf_cvg": total - cf,
            "quality_percentage": round(passed / total * 100, 1) if total > 0 else 0.0,
        })

    df = pd.DataFrame(rows)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info("Saved quality report: %s", output_csv)

    return df


def plot_quality_heatmap(all_years_quality_df, output_path):
    """Heatmap: districts (rows) x years (columns), colored by quality %."""
    pivot = all_years_quality_df.pivot_table(
        index="district", columns="year", values="quality_percentage"
    )
    pivot = pivot.sort_values(pivot.columns[-1], ascending=False)

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int), rotation=45, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("Data Quality Heatmap: % Pixels Passing All Filters", fontsize=14)
    plt.colorbar(im, ax=ax, label="Quality %")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)
