"""
Sky brightness estimation from VIIRS radiance.

Converts satellite-measured upward radiance (nW/cm2/sr) to approximate
zenith sky brightness in mag/arcsec2, enabling comparison with
ground-based SQM measurements and Bortle scale classification.

Based on Falchi et al. (2016) methodology:
  "The new world atlas of artificial night sky brightness."
  Science Advances, 2(6), e1600377.

IMPORTANT LIMITATIONS:
- **Local-pixel-only approximation (CF1, F1):** This conversion uses only
  the local pixel's upward radiance. Cinzano & Falchi (2012) show that sky
  brightness at any point integrates scattered light from sources within a
  ~195 km radius. At dark sites near cities, ignoring this integration can
  produce errors of 1-3 mag/arcsec². For accurate sky brightness
  assessments, use the Falchi et al. (2016) World Atlas GeoTIFF instead.
  Ref: Cinzano, P. & Falchi, F. (2012). Monthly Notices of the Royal
  Astronomical Society, 427(4), 3337-3357.
- **LED spectral bias (F4, K2):** VIIRS DNB misses blue-shifted LED light,
  causing Bortle classifications to drift 1-2 classes over the study period
  as municipalities transition from HPS to LED lighting.
  Ref: Kyba et al. (2023), Science, 379(6629), 265-268.
- **No elevation correction (CF4):** Maharashtra ranges from sea level to
  ~1400 m; atmospheric column depth affects scattering efficiency but is
  not accounted for here.
"""

from src.logging_config import get_pipeline_logger
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.formulas.sky_brightness import (
    NATURAL_SKY_BRIGHTNESS as _NATURAL_SKY_BRIGHTNESS,
    RADIANCE_TO_MCD as _RADIANCE_TO_MCD,
    REFERENCE_MCD as _REFERENCE_MCD,
    BORTLE_THRESHOLDS as _BORTLE_THRESHOLDS,
)

log = get_pipeline_logger(__name__)

# Re-export from src.formulas.sky_brightness for backwards compatibility
NATURAL_SKY_BRIGHTNESS = _NATURAL_SKY_BRIGHTNESS
RADIANCE_TO_MCD = _RADIANCE_TO_MCD
REFERENCE_MCD = _REFERENCE_MCD
BORTLE_THRESHOLDS = _BORTLE_THRESHOLDS


def radiance_to_sky_brightness(radiance_nw):
    """Convert VIIRS radiance (nW/cm²/sr) to zenith sky brightness.

    Uses the Falchi et al. (2016) empirical relationship between
    upward satellite-measured radiance and ground-level sky brightness.

    NOTE: This is a local-pixel-only approximation. True sky brightness
    integrates scattered light from sources within ~195 km (Cinzano &
    Falchi 2012). Results may err by 1-3 mag at dark sites near cities.
    See module docstring for full list of limitations.

    Args:
        radiance_nw: Radiance in nW/cm²/sr (scalar or array).

    Returns:
        Sky brightness in mag/arcsec² (scalar or array).
    """
    radiance_nw = np.asarray(radiance_nw, dtype=float)

    # Convert to mcd/m² (artificial component)
    artificial_mcd = radiance_nw * RADIANCE_TO_MCD

    # Add natural sky background (~0.100 mcd/m² for 22.0 mag/arcsec²)
    natural_mcd = REFERENCE_MCD * 10 ** (-0.4 * NATURAL_SKY_BRIGHTNESS)
    total_mcd = artificial_mcd + natural_mcd

    # Convert to mag/arcsec²
    mag = -2.5 * np.log10(total_mcd / REFERENCE_MCD)

    return mag


def classify_bortle(mag_arcsec2):
    """Classify sky brightness on the Bortle scale (1-9).

    Args:
        mag_arcsec2: Sky brightness in mag/arcsec².

    Returns:
        Tuple of (bortle_class, description).
    """
    mag = float(mag_arcsec2)
    for bortle_class in range(1, 10):
        low, high, desc = BORTLE_THRESHOLDS[bortle_class]
        if low <= mag < high:
            return bortle_class, desc
    return 9, "Inner-city sky"


def compute_sky_brightness_metrics(metrics_df, output_csv=None):
    """Add sky brightness columns to site/district metrics.

    Args:
        metrics_df: DataFrame with 'median_radiance' column.
        output_csv: Optional path to save results.

    Returns:
        DataFrame with added sky_brightness and bortle columns.
    """
    df = metrics_df.copy()

    if "median_radiance" not in df.columns:
        log.warning("No 'median_radiance' column; cannot compute sky brightness")
        return df

    radiance = df["median_radiance"].values
    valid = np.isfinite(radiance)

    mag = np.full(len(radiance), np.nan)
    bortle_class = np.full(len(radiance), np.nan)
    bortle_desc = [""] * len(radiance)

    mag[valid] = radiance_to_sky_brightness(radiance[valid])

    for i in range(len(radiance)):
        if valid[i]:
            bc, bd = classify_bortle(mag[i])
            bortle_class[i] = bc
            bortle_desc[i] = bd

    df["sky_brightness_mag"] = np.round(mag, 2)
    df["bortle_class"] = bortle_class
    df["bortle_description"] = bortle_desc

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info("Saved sky brightness metrics: %s", output_csv)

    return df


def plot_sky_brightness_distribution(metrics_df, output_path):
    """Histogram of sky brightness values with Bortle scale overlay.

    Args:
        metrics_df: DataFrame with 'sky_brightness_mag' column.
        output_path: Path to save figure.
    """
    if "sky_brightness_mag" not in metrics_df.columns:
        log.warning("No sky_brightness_mag column for distribution plot")
        return

    mag = metrics_df["sky_brightness_mag"].dropna().values

    if len(mag) == 0:
        log.warning("No valid sky brightness data for plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Adaptive bin count: avoid near-empty histograms with few data points
    n_bins = min(20, max(5, len(mag) // 2))
    ax.hist(mag, bins=n_bins, color="midnightblue", alpha=0.7, edgecolor="white")

    # Overlay Bortle boundaries
    bortle_colors = {
        1: "#000033", 2: "#000066", 3: "#003366", 4: "#336699",
        5: "#6699CC", 6: "#99CCFF", 7: "#FFCC99", 8: "#FF9966", 9: "#FF6633",
    }
    for bc, (low, high, desc) in BORTLE_THRESHOLDS.items():
        if low > 0:
            ax.axvline(x=low, color=bortle_colors[bc], linestyle="--",
                       linewidth=0.8, alpha=0.7)
            ax.text(low, ax.get_ylim()[1] * 0.95, f"B{bc}",
                    fontsize=7, ha="center", color=bortle_colors[bc])

    ax.set_xlabel("Sky Brightness (mag/arcsec²)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Sky Brightness Distribution with Bortle Scale", fontsize=14)
    ax.invert_xaxis()  # Brighter (lower mag) on right
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)
