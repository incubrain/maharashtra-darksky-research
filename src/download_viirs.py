#!/usr/bin/env python3
"""
Download VIIRS DNB Annual Composites from NOAA EOG.

NOAA EOG requires free registration at https://eogdata.mines.edu/products/vnl/
After registering, you can download via browser or use this script with your
bearer token.

Usage:
    # Option 1: Manual download
    # Go to https://eogdata.mines.edu/nighttime_light/annual/v22/<YEAR>/
    # Download the .tif.gz files into ./viirs/<YEAR>/

    # Option 2: With bearer token (after registering at eogdata.mines.edu)
    python src/download_viirs.py --token YOUR_BEARER_TOKEN --years 2012-2024

    # Option 3: Generate synthetic test data for pipeline validation
    python src/download_viirs.py --generate-test-data --years 2023-2024
"""

import argparse
import gzip
import os
import sys

import numpy as np


# VIIRS version mapping (NOAA changed versions over time)
YEAR_VERSION = {
    2012: "v21", 2013: "v21",
    2014: "v22", 2015: "v22", 2016: "v22", 2017: "v22",
    2018: "v22", 2019: "v22", 2020: "v22", 2021: "v22",
    2022: "v22", 2023: "v22", 2024: "v22",
}

# Layers to download per year
LAYERS = ["average_masked", "median_masked", "cf_cvg", "lit_mask"]


def generate_test_data(viirs_dir, years, shapefile_path=None):
    """Generate synthetic VIIRS-like rasters for pipeline testing.

    Creates realistic-looking Maharashtra-extent rasters with:
    - Urban areas (Mumbai, Pune, Nagpur) having high radiance
    - Rural/tribal areas having low radiance
    - Gradual growth trend over years
    """
    import rasterio
    from rasterio.transform import from_bounds
    import geopandas as gpd

    # Maharashtra approximate bounds
    west, south, east, north = 72.5, 15.5, 81.0, 22.1
    res = 0.004166667  # ~15 arc-seconds (VIIRS resolution)
    width = int((east - west) / res)
    height = int((north - south) / res)
    transform = from_bounds(west, south, east, north, width, height)

    print(f"Generating test rasters: {width}x{height} pixels ({res}° resolution)")

    # Known city approximate locations (lon, lat) and relative brightness
    cities = {
        "Mumbai": (72.88, 19.08, 50.0),
        "Pune": (73.86, 18.52, 25.0),
        "Nagpur": (79.09, 21.15, 15.0),
        "Nashik": (73.79, 20.00, 8.0),
        "Aurangabad": (75.34, 19.88, 7.0),
        "Solapur": (75.92, 17.68, 6.0),
        "Kolhapur": (74.24, 16.70, 5.0),
        "Thane": (72.98, 19.20, 30.0),
        "Navi Mumbai": (73.02, 19.03, 20.0),
    }

    # Load shapefile for masking if available
    mask = None
    if shapefile_path and os.path.exists(shapefile_path):
        gdf = gpd.read_file(shapefile_path)
        from rasterio.features import geometry_mask
        mask = ~geometry_mask(
            gdf.geometry, out_shape=(height, width), transform=transform, all_touched=True
        )
        print(f"Using shapefile mask: {mask.sum()} pixels inside Maharashtra")

    rng = np.random.default_rng(42)

    for year in years:
        year_dir = os.path.join(viirs_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)

        growth_factor = 1.0 + 0.08 * (year - 2012)  # ~8% annual growth from baseline

        # Create base radiance field
        base = rng.exponential(0.3, size=(height, width)).astype("float32")

        # Add city lights with Gaussian falloff
        y_coords = np.linspace(north, south, height)
        x_coords = np.linspace(west, east, width)
        xx, yy = np.meshgrid(x_coords, y_coords)

        for city, (lon, lat, brightness) in cities.items():
            dist = np.sqrt((xx - lon)**2 + (yy - lat)**2)
            city_light = brightness * growth_factor * np.exp(-dist**2 / (2 * 0.15**2))
            base += city_light.astype("float32")

        # Apply Maharashtra mask
        if mask is not None:
            base = np.where(mask, base, np.nan)

        # Add year-to-year noise
        noise = rng.normal(0, 0.1, size=base.shape).astype("float32")
        base = np.clip(base + noise, 0, None)

        meta = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": np.nan,
        }

        # median_masked (main analysis layer)
        median_path = os.path.join(year_dir, f"median_masked_{year}.tif")
        with rasterio.open(median_path, "w", **meta) as dst:
            dst.write(base, 1)

        # average_masked (slightly noisier)
        avg = base * rng.uniform(0.9, 1.1, size=base.shape).astype("float32")
        avg_path = os.path.join(year_dir, f"average_masked_{year}.tif")
        with rasterio.open(avg_path, "w", **meta) as dst:
            dst.write(avg, 1)

        # cf_cvg (cloud-free coverage: 0-365)
        cf = rng.integers(0, 30, size=(height, width)).astype("float32")
        cf = np.where(mask, cf, 0) if mask is not None else cf
        cf_path = os.path.join(year_dir, f"cf_cvg_{year}.tif")
        with rasterio.open(cf_path, "w", **meta) as dst:
            dst.write(cf, 1)

        # lit_mask (binary: 0 or 1)
        lit = (base > 0.5).astype("float32")
        lit_path = os.path.join(year_dir, f"lit_mask_{year}.tif")
        with rasterio.open(lit_path, "w", **meta) as dst:
            dst.write(lit, 1)

        # Compress to .gz
        for tif_path in [median_path, avg_path, cf_path, lit_path]:
            gz_path = tif_path + ".gz"
            with open(tif_path, "rb") as f_in:
                with gzip.open(gz_path, "wb") as f_out:
                    f_out.write(f_in.read())
            os.remove(tif_path)  # keep only .gz
            print(f"  Created: {os.path.basename(gz_path)} "
                  f"({os.path.getsize(gz_path) / 1e6:.1f} MB)")

        print(f"Year {year}: done (growth factor={growth_factor:.2f})")

    print(f"\nTest data generated in {viirs_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Download/generate VIIRS data")
    parser.add_argument("--viirs-dir", default="./viirs")
    parser.add_argument("--years", default="2012-2024")
    parser.add_argument("--token", help="NOAA EOG bearer token for download")
    parser.add_argument("--generate-test-data", action="store_true",
                        help="Generate synthetic test rasters")
    parser.add_argument("--shapefile-path",
                        default="./data/shapefiles/maharashtra_district.shp")
    args = parser.parse_args()

    if "-" in args.years:
        start, end = args.years.split("-")
        years = list(range(int(start), int(end) + 1))
    else:
        years = [int(y) for y in args.years.split(",")]

    if args.generate_test_data:
        generate_test_data(args.viirs_dir, years, args.shapefile_path)
    elif args.token:
        print("Token-based download not yet implemented.")
        print("Please download manually from https://eogdata.mines.edu/nighttime_light/annual/v22/")
        sys.exit(1)
    else:
        print("Please specify --generate-test-data or --token.")
        print("VIIRS data requires free registration at https://eogdata.mines.edu/products/vnl/")
        sys.exit(1)


if __name__ == "__main__":
    main()
