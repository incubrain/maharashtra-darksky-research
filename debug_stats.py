import os
import rasterio
import numpy as np
from src import config

def quick_stats(year=2024):
    path = os.path.join(config.OUTPUT_DIR, "subsets", str(year), f"maharashtra_median_{year}.tif")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    with rasterio.open(path) as src:
        data = src.read(1)
        valid = data[np.isfinite(data) & (data > 0)]
        print(f"--- Stats for {year} ---")
        print(f"Min:    {np.min(valid):.4f}")
        print(f"P01:    {np.percentile(valid, 1):.4f}")
        print(f"Median: {np.median(valid):.4f}")
        print(f"Max:    {np.max(valid):.4f}")

if __name__ == "__main__":
    quick_stats(2012)
    quick_stats(2020)
    quick_stats(2024)
