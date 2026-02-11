"""
Shared fixtures for VIIRS pipeline tests.

Provides synthetic raster data, GeoDataFrames, and temporary directories
so each test module can focus on verifying pipeline logic against known inputs.
"""

import os
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box


# ---------------------------------------------------------------------------
# Constants for synthetic test geometry
# ---------------------------------------------------------------------------
# Two simple rectangular "districts" side by side within Maharashtra bbox
WEST, SOUTH, EAST, NORTH = 76.0, 19.0, 78.0, 21.0
MID_LON = (WEST + EAST) / 2  # 77.0

# Raster resolution matching VIIRS (~15 arc-sec ≈ 0.00417°)
RES = 0.00417
WIDTH = int((EAST - WEST) / RES)
HEIGHT = int((NORTH - SOUTH) / RES)
TRANSFORM = from_bounds(WEST, SOUTH, EAST, NORTH, WIDTH, HEIGHT)


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after test."""
    with tempfile.TemporaryDirectory(prefix="viirs_test_") as d:
        yield d


@pytest.fixture
def two_district_gdf():
    """GeoDataFrame with two rectangular districts: 'DistrictA' (west) and 'DistrictB' (east)."""
    geom_a = box(WEST, SOUTH, MID_LON, NORTH)
    geom_b = box(MID_LON, SOUTH, EAST, NORTH)
    return gpd.GeoDataFrame(
        {"district": ["DistrictA", "DistrictB"], "geometry": [geom_a, geom_b]},
        crs="EPSG:4326",
    )


def _write_raster(path, data, transform=TRANSFORM):
    """Helper: write a 2D float32 array as a single-band GeoTIFF."""
    meta = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": np.nan,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data.astype("float32"), 1)
    return path


@pytest.fixture
def uniform_rasters(tmp_dir):
    """Create median, lit_mask, and cf_cvg rasters with uniform known values.

    - median: 5.0 everywhere (known radiance)
    - lit_mask: 1 everywhere (all lit)
    - cf_cvg: 10 everywhere (passes default threshold of 5)
    """
    median_data = np.full((HEIGHT, WIDTH), 5.0, dtype="float32")
    lit_data = np.ones((HEIGHT, WIDTH), dtype="float32")
    cf_data = np.full((HEIGHT, WIDTH), 10.0, dtype="float32")

    median_path = _write_raster(os.path.join(tmp_dir, "median.tif"), median_data)
    lit_path = _write_raster(os.path.join(tmp_dir, "lit_mask.tif"), lit_data)
    cf_path = _write_raster(os.path.join(tmp_dir, "cf_cvg.tif"), cf_data)

    return {
        "median": median_path,
        "lit_mask": lit_path,
        "cf_cvg": cf_path,
        "median_value": 5.0,
    }


@pytest.fixture
def partial_mask_rasters(tmp_dir):
    """Rasters where the western half fails lit_mask (= 0) and eastern half passes.

    This lets us verify that filtering correctly excludes pixels.
    - median: 10.0 everywhere
    - lit_mask: 0 in west half, 1 in east half
    - cf_cvg: 10 everywhere
    """
    median_data = np.full((HEIGHT, WIDTH), 10.0, dtype="float32")
    lit_data = np.zeros((HEIGHT, WIDTH), dtype="float32")
    lit_data[:, WIDTH // 2:] = 1.0  # east half is lit
    cf_data = np.full((HEIGHT, WIDTH), 10.0, dtype="float32")

    median_path = _write_raster(os.path.join(tmp_dir, "median.tif"), median_data)
    lit_path = _write_raster(os.path.join(tmp_dir, "lit_mask.tif"), lit_data)
    cf_path = _write_raster(os.path.join(tmp_dir, "cf_cvg.tif"), cf_data)

    return {
        "median": median_path,
        "lit_mask": lit_path,
        "cf_cvg": cf_path,
        "median_value": 10.0,
    }


@pytest.fixture
def trend_dataframe():
    """Synthetic yearly radiance data for two districts with known growth rates.

    DistrictA: starts at 2.0 nW, grows at ~8% per year (exponential)
    DistrictB: starts at 0.5 nW, stays roughly constant (0% growth)
    """
    years = list(range(2012, 2025))  # 13 years
    rows = []
    rng = np.random.default_rng(42)

    for year in years:
        # DistrictA: exponential growth ~8%/yr
        t = year - 2012
        rad_a = 2.0 * np.exp(0.08 * t) + rng.normal(0, 0.05)
        rows.append({
            "district": "DistrictA",
            "year": year,
            "mean_radiance": rad_a * 1.1,
            "median_radiance": max(rad_a, 0.01),
            "pixel_count": 1000,
            "min_radiance": rad_a * 0.5,
            "max_radiance": rad_a * 2.0,
            "std_radiance": rad_a * 0.3,
        })

        # DistrictB: roughly flat at 0.5
        rad_b = 0.5 + rng.normal(0, 0.02)
        rows.append({
            "district": "DistrictB",
            "year": year,
            "mean_radiance": rad_b * 1.1,
            "median_radiance": max(rad_b, 0.01),
            "pixel_count": 800,
            "min_radiance": rad_b * 0.5,
            "max_radiance": rad_b * 2.0,
            "std_radiance": rad_b * 0.3,
        })

    return pd.DataFrame(rows)
