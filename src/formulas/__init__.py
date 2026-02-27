"""
Centralized scientific formulas, constants, and classification functions.

This subpackage consolidates all domain-specific constants and pure
computational functions used across the VIIRS ALAN analysis pipeline.
config.py retains runtime parameters, paths, and location definitions;
this package holds the science.
"""

from src.formulas.classification import (
    classify_alan,
    classify_alan_series,
    classify_stability,
    TIER_COLORS,
)
from src.formulas.trend import fit_log_linear_trend
from src.formulas.sky_brightness import (
    NATURAL_SKY_BRIGHTNESS,
    RADIANCE_TO_MCD,
    REFERENCE_MCD,
    BORTLE_THRESHOLDS,
)
from src.formulas.spatial import (
    EARTH_RADIUS_KM,
    DIRECTION_DEFINITIONS,
)
from src.formulas.ecology import (
    LAND_COVER_CLASSES,
    ECOLOGICAL_SENSITIVITY,
)
from src.formulas.benchmarks import (
    PUBLISHED_BENCHMARKS,
    BENCHMARK_INTERPRETATION_THRESHOLD,
)
from src.formulas.diagnostics_thresholds import (
    OUTLIER_Z_THRESHOLD,
    DW_WARNING_LOW,
    DW_WARNING_HIGH,
    JB_P_THRESHOLD,
    COOKS_D_THRESHOLD,
    R_SQUARED_WARNING,
    CV_STABLE_THRESHOLD,
    CV_ERRATIC_THRESHOLD,
)
from src.formulas.fitting import (
    EXP_DECAY_BOUNDS,
    EXP_DECAY_MAXFEV,
    LIGHT_DOME_BACKGROUND_THRESHOLD,
)
from src.formulas.quality import (
    CF_CVG_VALID_RANGE,
)

__all__ = [
    # classification
    "classify_alan",
    "classify_alan_series",
    "classify_stability",
    "TIER_COLORS",
    # trend
    "fit_log_linear_trend",
    # sky brightness
    "NATURAL_SKY_BRIGHTNESS",
    "RADIANCE_TO_MCD",
    "REFERENCE_MCD",
    "BORTLE_THRESHOLDS",
    # spatial
    "EARTH_RADIUS_KM",
    "DIRECTION_DEFINITIONS",
    # ecology
    "LAND_COVER_CLASSES",
    "ECOLOGICAL_SENSITIVITY",
    # benchmarks
    "PUBLISHED_BENCHMARKS",
    "BENCHMARK_INTERPRETATION_THRESHOLD",
    # diagnostics thresholds
    "OUTLIER_Z_THRESHOLD",
    "DW_WARNING_LOW",
    "DW_WARNING_HIGH",
    "JB_P_THRESHOLD",
    "COOKS_D_THRESHOLD",
    "R_SQUARED_WARNING",
    "CV_STABLE_THRESHOLD",
    "CV_ERRATIC_THRESHOLD",
    # fitting
    "EXP_DECAY_BOUNDS",
    "EXP_DECAY_MAXFEV",
    "LIGHT_DOME_BACKGROUND_THRESHOLD",
    # quality
    "CF_CVG_VALID_RANGE",
]
