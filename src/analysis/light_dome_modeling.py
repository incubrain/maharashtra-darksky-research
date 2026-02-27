"""
Urban radiance footprint modeling.

Fits exponential decay models to urban radial profiles to quantify
the spatial extent of city radiance footprints.

TERMINOLOGY NOTE (finding CF2, review 2026-02-27):
This module models the spatial decay of upward VIIRS-detected radiance
from urban centres — an "urban radiance footprint", NOT a true atmospheric
"light dome". Cinzano & Falchi (2012) show that atmospheric light
propagation follows a ~d^(-2.5) power law and integrates over ~195 km,
whereas our exponential decay model describes urban morphology (built-up
area tapering at city edges). The two phenomena are physically distinct:
  - Radiance footprint: ground-level light emission pattern (this module)
  - Light dome: sky glow visible to an observer from scattered light
Ref: Cinzano, P. & Falchi, F. (2012). MNRAS, 427(4), 3337-3357.
"""

from src.logging_config import get_pipeline_logger
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src import config
from src.formulas.fitting import EXP_DECAY_BOUNDS, EXP_DECAY_MAXFEV, LIGHT_DOME_BACKGROUND_THRESHOLD

log = get_pipeline_logger(__name__)


def _exp_decay(d, peak, decay_rate, background):
    """Exponential decay model: radiance(d) = peak * exp(-decay * d) + bg."""
    return peak * np.exp(-decay_rate * d) + background


def fit_light_dome_model(radial_profile_df, city_name, background_threshold=None):
    """Fit exponential decay model to urban radial profile.

    Args:
        radial_profile_df: Output from extract_radial_profiles().
        city_name: Name of city to model.
        background_threshold: Radiance defining "background" (nW/cm²/sr).

    Returns:
        Dict with dome metrics.
    """
    if background_threshold is None:
        background_threshold = LIGHT_DOME_BACKGROUND_THRESHOLD

    sub = radial_profile_df[radial_profile_df["city"] == city_name].sort_values("distance_km")

    if len(sub) < 3:
        return {
            "city": city_name, "peak_radiance": np.nan,
            "dome_radius_km": np.nan, "decay_rate": np.nan,
            "half_distance_km": np.nan, "model_r_squared": np.nan,
            "effective_area_km2": np.nan,
        }

    distances = sub["distance_km"].values.astype(float)
    radiance = sub["median_radiance"].values.astype(float)

    # Remove NaN
    valid = np.isfinite(radiance)
    distances = distances[valid]
    radiance = radiance[valid]

    if len(distances) < 3:
        return {
            "city": city_name, "peak_radiance": np.nan,
            "dome_radius_km": np.nan, "decay_rate": np.nan,
            "half_distance_km": np.nan, "model_r_squared": np.nan,
            "effective_area_km2": np.nan,
        }

    try:
        p0 = [radiance[0], 0.1, 0.1]
        popt, _ = curve_fit(_exp_decay, distances, radiance, p0=p0,
                            bounds=EXP_DECAY_BOUNDS,
                            maxfev=EXP_DECAY_MAXFEV)
        peak, decay_rate, background = popt

        fitted = _exp_decay(distances, *popt)
        ss_res = np.sum((radiance - fitted) ** 2)
        ss_tot = np.sum((radiance - np.mean(radiance)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Dome radius: where model = background_threshold
        if peak > background_threshold and decay_rate > 0 and background < background_threshold:
            dome_radius = -np.log((background_threshold - background) / peak) / decay_rate
            dome_radius = max(0, dome_radius)
        else:
            dome_radius = np.nan

        # Half distance: where radiance = peak/2 + background
        if decay_rate > 0:
            half_dist = np.log(2) / decay_rate
        else:
            half_dist = np.nan

        effective_area = np.pi * dome_radius ** 2 if not np.isnan(dome_radius) else np.nan

    except Exception as e:
        log.warning("Dome model fit failed for %s: %s", city_name, e)
        return {
            "city": city_name, "peak_radiance": radiance[0],
            "dome_radius_km": np.nan, "decay_rate": np.nan,
            "half_distance_km": np.nan, "model_r_squared": np.nan,
            "effective_area_km2": np.nan,
        }

    return {
        "city": city_name,
        "peak_radiance": round(peak, 2),
        "dome_radius_km": round(dome_radius, 1) if not np.isnan(dome_radius) else np.nan,
        "decay_rate": round(decay_rate, 4),
        "half_distance_km": round(half_dist, 1) if not np.isnan(half_dist) else np.nan,
        "model_r_squared": round(r_squared, 3),
        "effective_area_km2": round(effective_area, 0) if not np.isnan(effective_area) else np.nan,
    }


def model_all_city_domes(radial_profiles_df, output_csv=None):
    """Fit light dome models for all cities."""
    results = []
    for city in radial_profiles_df["city"].unique():
        dome = fit_light_dome_model(radial_profiles_df, city)
        results.append(dome)

    df = pd.DataFrame(results)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info("Saved light dome metrics: %s", output_csv)

    return df


def plot_dome_comparison(dome_metrics_df, radial_profiles_df, output_path):
    """Combined visualization of light domes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Radial profiles with fitted curves
    ax = axes[0]
    for city in radial_profiles_df["city"].unique():
        sub = radial_profiles_df[radial_profiles_df["city"] == city].sort_values("distance_km")
        ax.plot(sub["distance_km"], sub["median_radiance"], "o-",
                label=city, markersize=4, linewidth=1.5)
    ax.set_xlabel("Distance from Centre (km)", fontsize=11)
    ax.set_ylabel("Median Radiance (nW/cm²/sr)", fontsize=11)
    ax.set_title("Urban Radial Profiles", fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Dome radius comparison
    ax = axes[1]
    df = dome_metrics_df.dropna(subset=["dome_radius_km"]).sort_values("dome_radius_km", ascending=True)
    if not df.empty:
        ax.barh(df["city"], df["dome_radius_km"], color="steelblue", alpha=0.7)
        ax.set_xlabel("Dome Radius (km)", fontsize=11)
        ax.set_title("Light Dome Extent", fontsize=13)
        ax.grid(axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Insufficient data for dome modeling",
                ha="center", va="center", transform=ax.transAxes)

    fig.suptitle("Urban Light Dome Analysis", fontsize=14, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)
