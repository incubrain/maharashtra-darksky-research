"""
Ecological and land-use overlay analysis.

Cross-references ALAN radiance data with land cover classes and
ecological zones to assess light pollution impact on different
habitat types and landscape contexts.
"""

from src.logging_config import get_pipeline_logger
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
<<<<<<< HEAD
from src import viirs_utils
=======
from src.formulas.ecology import (
    LAND_COVER_CLASSES as _LAND_COVER_CLASSES,
    ECOLOGICAL_SENSITIVITY as _ECOLOGICAL_SENSITIVITY,
)
>>>>>>> vk/de3e-comprehensive-de

log = get_pipeline_logger(__name__)

# Re-export from src.formulas.ecology for backwards compatibility
LAND_COVER_CLASSES = _LAND_COVER_CLASSES
ECOLOGICAL_SENSITIVITY = _ECOLOGICAL_SENSITIVITY


def compute_landcover_alan_stats(radiance_raster, landcover_raster,
                                 year=None, output_csv=None):
    """Cross-tabulate ALAN levels by land cover class.

    Args:
        radiance_raster: Path to VIIRS radiance raster.
        landcover_raster: Path to land cover classification raster.
        output_csv: Optional path to save results.

    Returns:
        DataFrame with land cover class statistics.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(radiance_raster) as rad_src:
        rad_data = rad_src.read(1).astype("float32")
        rad_transform = rad_src.transform
        rad_crs = rad_src.crs
        rad_shape = rad_data.shape

    # Apply Dynamic Background Subtraction
    rad_data = viirs_utils.apply_dynamic_background_subtraction(rad_data, year=year)

    with rasterio.open(landcover_raster) as lc_src:
        lc_data = lc_src.read(1)
        lc_crs = lc_src.crs
        lc_transform = lc_src.transform

        # Reproject land cover to match radiance grid if needed
        if lc_crs != rad_crs or lc_data.shape != rad_shape:
            lc_reproj = np.empty(rad_shape, dtype=lc_data.dtype)
            reproject(
                lc_data, lc_reproj,
                src_transform=lc_transform, src_crs=lc_crs,
                dst_transform=rad_transform, dst_crs=rad_crs,
                resampling=Resampling.nearest,
            )
            lc_data = lc_reproj

    # Mask invalid radiance pixels
    valid = np.isfinite(rad_data) & (rad_data >= 0)

    rows = []
    for class_id, class_name in LAND_COVER_CLASSES.items():
        mask = (lc_data == class_id) & valid
        pixel_count = mask.sum()

        if pixel_count == 0:
            continue

        radiance_vals = rad_data[mask]
        sensitivity = ECOLOGICAL_SENSITIVITY.get(class_name, 0.5)

        rows.append({
            "land_cover_class": class_name,
            "pixel_count": int(pixel_count),
            "mean_radiance": round(float(np.mean(radiance_vals)), 3),
            "median_radiance": round(float(np.median(radiance_vals)), 3),
            "std_radiance": round(float(np.std(radiance_vals)), 3),
            "max_radiance": round(float(np.max(radiance_vals)), 3),
            "pct_above_low": round(
                float((radiance_vals > config.ALAN_LOW_THRESHOLD).sum() / pixel_count * 100), 1
            ),
            "pct_above_medium": round(
                float((radiance_vals > config.ALAN_MEDIUM_THRESHOLD).sum() / pixel_count * 100), 1
            ),
            "ecological_sensitivity": sensitivity,
            "impact_score": round(
                float(np.mean(radiance_vals) * sensitivity), 3
            ),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("impact_score", ascending=False)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info("Saved land cover ALAN stats: %s", output_csv)

    return df


def compute_district_landcover_profile(gdf, radiance_raster, landcover_raster,
                                       year=None, output_csv=None):
    """Per-district breakdown of ALAN by land cover type.

    Args:
        gdf: District GeoDataFrame.
        radiance_raster: Path to VIIRS radiance raster.
        landcover_raster: Path to land cover raster.
        output_csv: Optional output path.

    Returns:
        DataFrame with district × land cover ALAN breakdown.
    """
    import rasterio
    from rasterio.mask import mask as rasterio_mask
    from shapely.geometry import mapping

    rows = []
    for _, district_row in gdf.iterrows():
        district_name = district_row["district"]
        geom = [mapping(district_row.geometry)]

        try:
            with rasterio.open(radiance_raster) as rad_src:
                rad_clip, _ = rasterio_mask(rad_src, geom, crop=True, nodata=np.nan)
                rad_clip = rad_clip[0].astype("float32")
                # Apply DBS to the clipped area (or global if we had the full raster, 
                # but local background subtraction is also valid/consistent here)
                rad_clip = viirs_utils.apply_dynamic_background_subtraction(rad_clip, year=year)

            with rasterio.open(landcover_raster) as lc_src:
                lc_clip, _ = rasterio_mask(lc_src, geom, crop=True, nodata=0)
                lc_clip = lc_clip[0]
        except Exception as e:
            log.debug("Skipping %s: %s", district_name, e)
            continue

        # Align shapes (use minimum extent)
        min_h = min(rad_clip.shape[0], lc_clip.shape[0])
        min_w = min(rad_clip.shape[1], lc_clip.shape[1])
        rad_clip = rad_clip[:min_h, :min_w]
        lc_clip = lc_clip[:min_h, :min_w]

        valid = np.isfinite(rad_clip)

        for class_id, class_name in LAND_COVER_CLASSES.items():
            class_mask = (lc_clip == class_id) & valid
            count = class_mask.sum()
            if count < 10:
                continue

            vals = rad_clip[class_mask]
            rows.append({
                "district": district_name,
                "land_cover_class": class_name,
                "pixel_count": int(count),
                "mean_radiance": round(float(np.mean(vals)), 3),
                "median_radiance": round(float(np.median(vals)), 3),
            })

    df = pd.DataFrame(rows)
    if output_csv and not df.empty:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info("Saved district land cover profile: %s", output_csv)

    return df


def plot_landcover_alan_comparison(landcover_stats_df, output_path):
    """Bar chart comparing ALAN levels across land cover types.

    Args:
        landcover_stats_df: Output from compute_landcover_alan_stats().
        output_path: Path to save figure.
    """
    if landcover_stats_df.empty:
        log.warning("No land cover stats for comparison plot")
        return

    df = landcover_stats_df.sort_values("median_radiance", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Median radiance by land cover
    ax = axes[0]
    colors = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(df)))
    ax.barh(df["land_cover_class"], df["median_radiance"], color=colors,
            edgecolor="grey", linewidth=0.5)
    ax.set_xlabel("Median Radiance (nW/cm²/sr)", fontsize=11)
    ax.set_title("ALAN by Land Cover Type", fontsize=13)
    ax.grid(axis="x", alpha=0.3)

    # Panel 2: Ecological impact score
    ax = axes[1]
    df_sorted = df.sort_values("impact_score", ascending=True)
    bar_colors = ["#d73027" if s > 0.6 else "#fee08b" if s > 0.3 else "#1a9850"
                  for s in df_sorted["ecological_sensitivity"]]
    ax.barh(df_sorted["land_cover_class"], df_sorted["impact_score"],
            color=bar_colors, edgecolor="grey", linewidth=0.5)
    ax.set_xlabel("Impact Score (radiance × sensitivity)", fontsize=11)
    ax.set_title("Ecological Impact Score", fontsize=13)
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("ALAN × Land Cover Analysis", fontsize=14, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)
