"""
Nearest city distance metrics for dark-sky candidate sites.

Computes great-circle distances and bearings from each site to the
nearest urban benchmark, enabling regression of ALAN ~ distance.
"""

from src.logging_config import get_pipeline_logger
import math
import os

import numpy as np
import pandas as pd

from src import config
from src.formulas.spatial import EARTH_RADIUS_KM

log = get_pipeline_logger(__name__)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Compute great-circle distance between two points (WGS84, in km)."""
    R = EARTH_RADIUS_KM
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_deg(lat1, lon1, lat2, lon2):
    """Compute initial bearing from point 1 to point 2 (degrees, 0=N)."""
    dlon = math.radians(lon2 - lon1)
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r) -
         math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _bearing_to_cardinal(bearing):
    """Convert bearing (degrees) to 8-point cardinal direction."""
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(bearing / 45) % 8
    return directions[idx]


def compute_nearest_city_distances(site_locations=None, city_locations=None,
                                   output_csv=None):
    """For each site, find distance and direction to nearest city.

    Args:
        site_locations: Dict from config.DARKSKY_SITES (default).
        city_locations: Dict from config.URBAN_BENCHMARKS (default).
        output_csv: Path to save results (optional).

    Returns:
        DataFrame with columns: [site, nearest_city_name, distance_km,
        cardinal_direction, bearing_degrees].
    """
    if site_locations is None:
        site_locations = config.DARKSKY_SITES
    if city_locations is None:
        city_locations = config.URBAN_BENCHMARKS

    results = []
    for site_name, site_info in site_locations.items():
        slat, slon = site_info["lat"], site_info["lon"]
        best_dist = float("inf")
        best_city = None
        best_bearing = 0.0

        for city_name, city_info in city_locations.items():
            clat, clon = city_info["lat"], city_info["lon"]
            dist = _haversine_km(slat, slon, clat, clon)
            if dist < best_dist:
                best_dist = dist
                best_city = city_name
                best_bearing = _bearing_deg(slat, slon, clat, clon)

        results.append({
            "site": site_name,
            "nearest_city_name": best_city,
            "distance_km": round(best_dist, 1),
            "cardinal_direction": _bearing_to_cardinal(best_bearing),
            "bearing_degrees": round(best_bearing, 1),
        })

    df = pd.DataFrame(results)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info("Saved proximity metrics: %s", output_csv)

    return df
