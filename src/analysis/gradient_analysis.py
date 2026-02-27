"""
Urban-rural radial gradient analysis for ALAN decay profiles.

Extracts mean/median radiance at concentric rings around city centres
to quantify light spillover extent and identify "dark zones".

METHODOLOGY:
Radial extraction at concentric distances from city centres characterises
the spatial decay of urban nighttime light. Standard approach in NTL
literature for analysing urban-rural light gradients.
Ref: Bennie, J. et al. (2014). Contrasting trends in light pollution
     across Europe. Scientific Reports, 4, 3789.

NOTE ON DBS USAGE (finding Z3, review 2026-02-27):
This module applies Dynamic Background Subtraction (DBS) via
viirs_utils.apply_dynamic_background_subtraction() to radiance rasters
before gradient extraction. DBS is intentionally used here (and in
ecological_overlay.py) to remove the background noise floor for
visualization and spatial analysis purposes. The main zonal statistics
path in viirs_process.py does NOT apply DBS — raw quality-filtered
radiance is preserved for trend analysis. This is by design:
  - DBS for gradient/visualization: removes noise floor for cleaner
    spatial decay curves
  - No DBS for trend/stability: preserves absolute radiance for temporal
    comparisons
Ref: Levin, N. et al. (2020). Remote Sensing of Night Lights: A Review.
     Remote Sensing of Environment, 237, 111443.

VIEWING ANGLE CAVEAT (finding Z2, review 2026-02-27):
VIIRS DNB viewing geometry introduces ~50% radiance variability between
nadir and edge-of-swath observations. This anisotropic effect is NOT
corrected in this analysis. Radial profiles assume isotropic measurement,
which may introduce systematic bias for cities observed at high view angles.
Ref: Zheng, Q. et al. (2019). Remote Sensing, 11(18), 2132.
"""

from src.logging_config import get_pipeline_logger
import os
import tempfile

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import Point

from src import config
from src import viirs_utils

log = get_pipeline_logger(__name__)


def extract_radial_profiles(raster_path, city_locations, year=None, radii_km=None, output_csv=None):
    """Extract mean/median radiance at concentric rings around each city.

    Args:
        raster_path: Path to radiance raster (e.g., maharashtra_median_2024.tif).
        city_locations: Dict from config.URBAN_CITIES.
        radii_km: List of distances (default: config.URBAN_GRADIENT_RADII_KM).
        output_csv: Path to save results (optional).

    Returns:
        DataFrame with columns: [city, distance_km, mean_radiance,
        median_radiance, pixel_count, std_radiance].
    """
    if radii_km is None:
        radii_km = config.URBAN_GRADIENT_RADII_KM

    results = []
    for city_name, info in city_locations.items():
        point = Point(info["lon"], info["lat"])
        gdf_point = gpd.GeoDataFrame(
            [{"geometry": point}], crs="EPSG:4326"
        )
        gdf_utm = gdf_point.to_crs(epsg=config.MAHARASHTRA_UTM_EPSG)

        prev_radius = 0
        for radius_km in sorted(radii_km):
            # Create annulus: buffer(radius) - buffer(prev_radius)
            outer = gdf_utm.geometry.iloc[0].buffer(radius_km * 1000)
            inner = gdf_utm.geometry.iloc[0].buffer(prev_radius * 1000) if prev_radius > 0 else None
            annulus = outer.difference(inner) if inner else outer

            # Reproject to WGS84 for raster extraction
            annulus_gdf = gpd.GeoDataFrame(
                [{"geometry": annulus}], crs=f"EPSG:{config.MAHARASHTRA_UTM_EPSG}"
            ).to_crs("EPSG:4326")

            # Prepare corrected raster
            with rasterio.open(raster_path) as src:
                data = src.read(1)
                meta = src.meta.copy()
                # Apply DBS
                data_corrected = viirs_utils.apply_dynamic_background_subtraction(data, year=year)

            tmp_fd, tmp_path = tempfile.mkstemp(suffix="_gradient_corr.tif")
            os.close(tmp_fd)
            try:
                with rasterio.open(tmp_path, "w", **meta) as dst:
                    dst.write(data_corrected, 1)

                stats = zonal_stats(
                    annulus_gdf, tmp_path,
                    stats=["mean", "median", "count", "std"],
                    nodata=np.nan, all_touched=True,
                )
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            if stats and stats[0]["count"] and stats[0]["count"] > 0:
                results.append({
                    "city": city_name,
                    "distance_km": radius_km,
                    "mean_radiance": stats[0]["mean"],
                    "median_radiance": stats[0]["median"],
                    "pixel_count": stats[0]["count"],
                    "std_radiance": stats[0]["std"],
                })
            else:
                results.append({
                    "city": city_name,
                    "distance_km": radius_km,
                    "mean_radiance": np.nan,
                    "median_radiance": np.nan,
                    "pixel_count": 0,
                    "std_radiance": np.nan,
                })

            prev_radius = radius_km

    df = pd.DataFrame(results)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info("Saved radial profiles: %s", output_csv)

    return df


def plot_radial_decay_curves(profiles_df, output_path):
    """Plot radiance vs. distance for all cities on one figure.

    Creates line plot with cities as different colours, log Y-axis,
    horizontal line at ALAN_LOW_THRESHOLD, annotations showing
    "dark zone" distance where curve crosses threshold.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for city in profiles_df["city"].unique():
        sub = profiles_df[profiles_df["city"] == city].sort_values("distance_km")
        ax.plot(sub["distance_km"], sub["median_radiance"], "o-",
                label=city, markersize=5, linewidth=2)

    # Threshold line
    ax.axhline(y=config.ALAN_LOW_THRESHOLD, color="orange", linestyle="--",
               linewidth=1.5, label=f"Low-ALAN threshold ({config.ALAN_LOW_THRESHOLD} nW)")

    ax.set_yscale("log")
    ax.set_xlabel("Distance from City Centre (km)", fontsize=12)
    ax.set_ylabel("Median Radiance (nW/cm²/sr, log scale)", fontsize=12)
    ax.set_title("Urban Light Dome Decay Profiles", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)
