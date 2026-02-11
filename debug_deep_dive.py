import os
import matplotlib.pyplot as plt
import rasterio
import numpy as np
import logging
from src import config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def plot_histograms(years=[2012, 2020, 2024]):
    """Compare radiance distributions to identify background shifts."""
    plt.figure(figsize=(12, 6))
    
    for year in years:
        path = os.path.join(config.OUTPUT_DIR, "subsets", str(year), f"maharashtra_median_{year}.tif")
        if not os.path.exists(path):
            log.warning(f"Path missing: {path}")
            continue
            
        with rasterio.open(path) as src:
            data = src.read(1).flatten()
            
        # Filter for small values to see the noise floor
        valid = data[(data > 0) & (data < 2.0) & np.isfinite(data)]
        
        plt.hist(valid, bins=100, alpha=0.5, label=f"Year {year}", density=True)
        
    plt.axvline(0.25, color='r', linestyle='--', label="Original Threshold (0.25)")
    plt.title("Radiance Distribution Deep Dive (0.0 - 2.0 nW)")
    plt.xlabel("Radiance (nW/cm²/sr)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(config.OUTPUT_DIR, "debug_radiance_histograms.png")
    plt.savefig(out_path)
    log.info(f"Histogram saved to {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_histograms()
