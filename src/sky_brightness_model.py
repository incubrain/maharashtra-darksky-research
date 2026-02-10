"""
Sky brightness estimation from VIIRS radiance.

Converts satellite-measured upward radiance (nW/cm2/sr) to approximate
zenith sky brightness in mag/arcsec2, enabling comparison with
ground-based SQM measurements and Bortle scale classification.

Based on Falchi et al. (2016) methodology:
  "The new world atlas of artificial night sky brightness."
  Science Advances, 2(6), e1600377.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)

# Conversion constants from Falchi et al. (2016)
# Natural sky brightness: ~21.6 mag/arcsec² (typical dark site)
NATURAL_SKY_BRIGHTNESS = 21.6  # mag/arcsec²

# Empirical scaling: 1 nW/cm²/sr upward radiance ≈ 0.18 mcd/m² at zenith
# mcd/m² to mag/arcsec²: m = -2.5 * log10(L / 108000)
# where L is in mcd/m² and 108000 mcd/m² ≈ 0 mag/arcsec²
RADIANCE_TO_MCD = 0.18  # mcd/m² per nW/cm²/sr (empirical)
REFERENCE_MCD = 108000  # mcd/m² corresponding to 0 mag/arcsec²

# Bortle scale thresholds (approximate mag/arcsec²)
BORTLE_THRESHOLDS = {
    1: (21.75, np.inf, "Excellent dark-sky site"),
    2: (21.50, 21.75, "Typical dark-sky site"),
    3: (21.25, 21.50, "Rural sky"),
    4: (20.50, 21.25, "Rural/suburban transition"),
    5: (19.50, 20.50, "Suburban sky"),
    6: (18.50, 19.50, "Bright suburban sky"),
    7: (18.00, 18.50, "Suburban/urban transition"),
    8: (17.00, 18.00, "City sky"),
    9: (0.00, 17.00, "Inner-city sky"),
}


def radiance_to_sky_brightness(radiance_nw):
    """Convert VIIRS radiance (nW/cm²/sr) to zenith sky brightness.

    Uses the Falchi et al. (2016) empirical relationship between
    upward satellite-measured radiance and ground-level sky brightness.

    Args:
        radiance_nw: Radiance in nW/cm²/sr (scalar or array).

    Returns:
        Sky brightness in mag/arcsec² (scalar or array).
    """
    radiance_nw = np.asarray(radiance_nw, dtype=float)

    # Convert to mcd/m² (artificial component)
    artificial_mcd = radiance_nw * RADIANCE_TO_MCD

    # Add natural sky background (~0.171 mcd/m² for 21.6 mag/arcsec²)
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

    ax.hist(mag, bins=20, color="midnightblue", alpha=0.7, edgecolor="white")

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
