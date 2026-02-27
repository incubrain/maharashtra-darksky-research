"""
ALAN classification and stability classification functions.

All functions are pure (no I/O, no side effects).
"""

import numpy as np
import pandas as pd

from src import config


def classify_alan(median_radiance, low_threshold=None, medium_threshold=None):
    """Classify a single radiance value into ALAN level.

    Uses left-inclusive intervals (right=False semantics):
        radiance < low_threshold  → "low"
        low_threshold <= radiance < medium_threshold → "medium"
        radiance >= medium_threshold → "high"
        NaN → "unknown"

    Parameters
    ----------
    median_radiance : float
        Median radiance in nW/cm²/sr.
    low_threshold : float, optional
        Defaults to config.ALAN_LOW_THRESHOLD (1.0).
    medium_threshold : float, optional
        Defaults to config.ALAN_MEDIUM_THRESHOLD (5.0).

    Returns
    -------
    str
        One of "low", "medium", "high", "unknown".
    """
    if low_threshold is None:
        low_threshold = config.ALAN_LOW_THRESHOLD
    if medium_threshold is None:
        medium_threshold = config.ALAN_MEDIUM_THRESHOLD

    if pd.isna(median_radiance):
        return "unknown"
    if median_radiance < low_threshold:
        return "low"
    if median_radiance < medium_threshold:
        return "medium"
    return "high"


def classify_alan_series(radiance_series, low_threshold=None, medium_threshold=None):
    """Classify a Series/array of radiance values into ALAN levels.

    Uses pd.cut with right=False for left-inclusive intervals, consistent
    with classify_alan() scalar behavior.

    Parameters
    ----------
    radiance_series : array-like
        Radiance values in nW/cm²/sr.
    low_threshold : float, optional
        Defaults to config.ALAN_LOW_THRESHOLD (1.0).
    medium_threshold : float, optional
        Defaults to config.ALAN_MEDIUM_THRESHOLD (5.0).

    Returns
    -------
    pd.Categorical
        Categorical series with labels ["low", "medium", "high"].
    """
    if low_threshold is None:
        low_threshold = config.ALAN_LOW_THRESHOLD
    if medium_threshold is None:
        medium_threshold = config.ALAN_MEDIUM_THRESHOLD

    return pd.cut(
        radiance_series,
        bins=[-np.inf, low_threshold, medium_threshold, np.inf],
        labels=["low", "medium", "high"],
        right=False,
    )


def classify_stability(cv, stable_threshold=None, erratic_threshold=None):
    """Classify temporal stability by coefficient of variation.

    NOTE (findings SE1, SE5): The default thresholds (0.2/0.5) are
    project-specific heuristics, not published VIIRS standards. Small &
    Elvidge (2022) use a multi-moment approach with five zones instead.
    Results should be treated as exploratory classifications.

    Parameters
    ----------
    cv : float
        Coefficient of variation (std / mean).
    stable_threshold : float, optional
        Defaults to 0.2 (project heuristic, not a published threshold).
    erratic_threshold : float, optional
        Defaults to 0.5 (project heuristic, not a published threshold).

    Returns
    -------
    str
        One of "stable", "moderate", "erratic".
    """
    from src.formulas.diagnostics_thresholds import (
        CV_STABLE_THRESHOLD,
        CV_ERRATIC_THRESHOLD,
    )

    if stable_threshold is None:
        stable_threshold = CV_STABLE_THRESHOLD
    if erratic_threshold is None:
        erratic_threshold = CV_ERRATIC_THRESHOLD

    if pd.isna(cv):
        return "unknown"
    if cv < stable_threshold:
        return "stable"
    if cv < erratic_threshold:
        return "moderate"
    return "erratic"


# Tier colors for graduated ALAN percentile classification plots.
# Extracted from graduated_classification.py.
TIER_COLORS = {
    "Pristine": "#1a9850",
    "Low": "#91cf60",
    "Medium": "#fee08b",
    "High": "#fc8d59",
    "Very High": "#d73027",
}
