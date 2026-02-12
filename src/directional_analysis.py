"""
Directional brightness analysis: N/S/E/W quadrant ALAN measurement.

Measures radiance in cardinal direction wedges around each site to
identify dominant light pollution sources.
"""

from src.logging_config import get_pipeline_logger
import math
import os

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rasterstats import zonal_stats
from shapely.geometry import Point, Polygon

from src import config
from src.formulas.spatial import DIRECTION_DEFINITIONS

log = get_pipeline_logger(__name__)


def _create_wedge(centre_x, centre_y, radius, start_angle, end_angle, n_points=64):
    """Create a wedge-shaped polygon (sector of a circle).

    Args:
        centre_x, centre_y: Centre coordinates.
        radius: Wedge radius.
        start_angle, end_angle: Angles in degrees (0=N, 90=E, clockwise).
        n_points: Points along the arc for smoothness.

    Returns:
        Shapely Polygon.
    """
    angles = np.linspace(
        math.radians(90 - end_angle),
        math.radians(90 - start_angle),
        n_points,
    )
    arc_points = [(centre_x + radius * math.cos(a),
                   centre_y + radius * math.sin(a)) for a in angles]
    coords = [(centre_x, centre_y)] + arc_points + [(centre_x, centre_y)]
    return Polygon(coords)


def compute_directional_brightness(site_locations=None, raster_path=None,
                                   buffer_km=None, output_csv=None):
    """Compute mean radiance in N/S/E/W quadrants around each site.

    Args:
        site_locations: Dict from config.DARKSKY_SITES (default).
        raster_path: Path to radiance raster.
        buffer_km: Buffer radius to analyse (default: config value).
        output_csv: Path to save results.

    Returns:
        DataFrame with columns: [site, north_mean, south_mean, east_mean,
        west_mean, dominant_direction, max_min_ratio].
    """
    if site_locations is None:
        site_locations = config.DARKSKY_SITES
    if buffer_km is None:
        buffer_km = config.SITE_BUFFER_RADIUS_KM

    results = []
    for site_name, info in site_locations.items():
        point = Point(info["lon"], info["lat"])
        gdf_point = gpd.GeoDataFrame(
            [{"geometry": point}], crs="EPSG:4326"
        ).to_crs(epsg=config.MAHARASHTRA_UTM_EPSG)

        cx = gdf_point.geometry.iloc[0].x
        cy = gdf_point.geometry.iloc[0].y
        radius_m = buffer_km * 1000

        row = {"site": site_name}
        dir_values = {}

        for dir_name, dir_def in DIRECTION_DEFINITIONS.items():
            start_deg = dir_def["start_angle"]
            end_deg = dir_def["end_angle"]
            wedge = _create_wedge(cx, cy, radius_m, start_deg, end_deg)
            wedge_gdf = gpd.GeoDataFrame(
                [{"geometry": wedge}],
                crs=f"EPSG:{config.MAHARASHTRA_UTM_EPSG}"
            ).to_crs("EPSG:4326")

            stats = zonal_stats(
                wedge_gdf, raster_path,
                stats=["mean"], nodata=np.nan, all_touched=True,
            )
            val = stats[0].get("mean", np.nan) if stats else np.nan
            row[f"{dir_name}_mean"] = round(val, 4) if not np.isnan(val) else np.nan
            if not np.isnan(val):
                dir_values[dir_name] = val

        # Determine dominant direction
        if dir_values:
            row["dominant_direction"] = max(dir_values, key=dir_values.get).upper()
            max_val = max(dir_values.values())
            min_val = min(dir_values.values())
            row["max_min_ratio"] = round(max_val / min_val, 2) if min_val > 0 else np.nan
        else:
            row["dominant_direction"] = "N/A"
            row["max_min_ratio"] = np.nan

        results.append(row)

    df = pd.DataFrame(results)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info("Saved directional brightness: %s", output_csv)

    return df


def plot_directional_polar(directional_df, output_path):
    """Polar plot showing directional brightness for each site."""
    sites = directional_df["site"].values
    n_sites = len(sites)
    cols = min(4, n_sites)
    rows = math.ceil(n_sites / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows),
                             subplot_kw={"projection": "polar"})
    if n_sites == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    dir_angles = {"north_mean": 0, "east_mean": np.pi / 2,
                  "south_mean": np.pi, "west_mean": 3 * np.pi / 2}

    for i, (_, row) in enumerate(directional_df.iterrows()):
        ax = axes[i]
        angles = list(dir_angles.values())
        values = [row.get(d, 0) or 0 for d in dir_angles.keys()]
        # Close the polygon
        angles.append(angles[0])
        values.append(values[0])

        ax.plot(angles, values, "o-", linewidth=2, color="steelblue")
        ax.fill(angles, values, alpha=0.25, color="steelblue")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids([0, 90, 180, 270], ["N", "E", "S", "W"])
        ax.set_title(f"{row['site']}\n({row.get('dominant_direction', 'N/A')})",
                     fontsize=8, pad=12)

    # Hide unused axes
    for j in range(n_sites, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Directional Brightness Analysis (nW/cm²/sr)", fontsize=14, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)
