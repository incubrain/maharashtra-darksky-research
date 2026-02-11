import os
import logging
import numpy as np
import rasterio
import pandas as pd
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

def audit_all_years():
    """Perform a data consistency check across all available years."""
    # We'll use the subsets directory to find all processed years
    subset_root = os.path.join(config.OUTPUT_DIR, "subsets")
    if not os.path.exists(subset_root):
        log.error("Subsets directory not found: %s", subset_root)
        return

    years = sorted([int(d) for d in os.listdir(subset_root) if d.isdigit()])
    log.info("Auditing years: %s", years)

    audit_results = []

    for year in years:
        median_path = os.path.join(subset_root, str(year), f"maharashtra_median_{year}.tif")
        if not os.path.exists(median_path):
            continue

        with rasterio.open(median_path) as src:
            data = src.read(1)
            
        # Get stats on finite pixels > 0
        valid_mask = np.isfinite(data) & (data > 0)
        valid_data = data[valid_mask]

        if len(valid_data) == 0:
            log.warning("Year %d has no valid data > 0", year)
            continue

        p01 = np.percentile(valid_data, 1)
        p05 = np.percentile(valid_data, 5)
        median = np.median(valid_data)
        minimum = np.min(valid_data)
        
        # Percentage of pixels below certain thresholds (0.25, 0.50)
        # This helps identify if a background floor exists
        pct_below_025 = (np.sum(valid_data < 0.25) / len(valid_data)) * 100
        pct_below_050 = (np.sum(valid_data < 0.50) / len(valid_data)) * 100

        log.info(f"Year {year} | Min: {minimum:.4f} | P01: {p01:.4f} | Median: {median:.4f} | <0.25: {pct_below_025:.1f}%")

        audit_results.append({
            "year": year,
            "min": minimum,
            "p01": p01,
            "p05": p05,
            "median": median,
            "pct_below_025": pct_below_025,
            "pct_below_050": pct_below_050
        })

    df = pd.DataFrame(audit_results)
    csv_path = os.path.join(config.OUTPUT_DIR, "data_audit_report.csv")
    df.to_csv(csv_path, index=False)
    log.info("Full audit report saved to: %s", csv_path)

if __name__ == "__main__":
    audit_all_years()
