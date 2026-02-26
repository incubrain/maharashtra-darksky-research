"""
Tests for the src.formulas subpackage.

Covers classification functions, trend fitting with raw arrays,
and validation of all exported constants.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
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
from src.formulas.spatial import EARTH_RADIUS_KM, DIRECTION_DEFINITIONS
from src.formulas.ecology import LAND_COVER_CLASSES, ECOLOGICAL_SENSITIVITY
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
from src.formulas.quality import LIT_MASK_THRESHOLD, CF_CVG_VALID_RANGE


# ── classify_alan ──────────────────────────────────────────────────────


class TestClassifyAlan:
    """Tests for classify_alan() scalar classification."""

    @pytest.mark.parametrize(
        "radiance, expected",
        [
            (0.0, "low"),
            (0.5, "low"),
            (0.99, "low"),
            (1.0, "medium"),    # boundary: >= low_threshold
            (1.01, "medium"),
            (3.0, "medium"),
            (4.99, "medium"),
            (5.0, "high"),      # boundary: >= medium_threshold
            (5.01, "high"),
            (50.0, "high"),
            (1000.0, "high"),
        ],
    )
    def test_threshold_classification(self, radiance, expected):
        assert classify_alan(radiance) == expected

    def test_nan_returns_unknown(self):
        assert classify_alan(np.nan) == "unknown"
        assert classify_alan(float("nan")) == "unknown"

    def test_custom_thresholds(self):
        assert classify_alan(2.0, low_threshold=3.0, medium_threshold=10.0) == "low"
        assert classify_alan(5.0, low_threshold=3.0, medium_threshold=10.0) == "medium"
        assert classify_alan(15.0, low_threshold=3.0, medium_threshold=10.0) == "high"

    def test_uses_config_defaults(self):
        """Verify defaults match config.py values."""
        assert classify_alan(config.ALAN_LOW_THRESHOLD - 0.01) == "low"
        assert classify_alan(config.ALAN_LOW_THRESHOLD) == "medium"
        assert classify_alan(config.ALAN_MEDIUM_THRESHOLD) == "high"


# ── classify_alan_series ───────────────────────────────────────────────


class TestClassifyAlanSeries:
    """Tests for classify_alan_series() vectorized classification."""

    def test_basic_series(self):
        values = [0.5, 3.0, 10.0]
        result = classify_alan_series(values)
        assert list(result) == ["low", "medium", "high"]

    def test_consistency_with_scalar(self):
        """Every element should match classify_alan() for the same value."""
        test_values = [0.0, 0.5, 0.99, 1.0, 1.01, 3.0, 4.99, 5.0, 50.0]
        result = classify_alan_series(test_values)
        for val, cls in zip(test_values, result):
            assert cls == classify_alan(val), f"Mismatch at {val}: {cls} != {classify_alan(val)}"

    def test_pandas_series_input(self):
        s = pd.Series([0.5, 3.0, 10.0])
        result = classify_alan_series(s)
        assert len(result) == 3


# ── classify_stability ─────────────────────────────────────────────────


class TestClassifyStability:
    """Tests for classify_stability() CV-based classification."""

    @pytest.mark.parametrize(
        "cv, expected",
        [
            (0.0, "stable"),
            (0.1, "stable"),
            (0.19, "stable"),
            (0.2, "moderate"),    # boundary
            (0.3, "moderate"),
            (0.49, "moderate"),
            (0.5, "erratic"),     # boundary
            (0.8, "erratic"),
            (2.0, "erratic"),
        ],
    )
    def test_cv_classification(self, cv, expected):
        assert classify_stability(cv) == expected

    def test_nan_returns_unknown(self):
        assert classify_stability(np.nan) == "unknown"

    def test_custom_thresholds(self):
        assert classify_stability(0.15, stable_threshold=0.1) == "moderate"
        assert classify_stability(0.15, stable_threshold=0.2) == "stable"


# ── TIER_COLORS ────────────────────────────────────────────────────────


class TestTierColors:
    def test_all_labels_present(self):
        expected = set(config.ALAN_PERCENTILE_LABELS)
        assert set(TIER_COLORS.keys()) == expected

    def test_values_are_hex_colors(self):
        for label, color in TIER_COLORS.items():
            assert color.startswith("#"), f"{label} color should be hex"
            assert len(color) == 7, f"{label} color should be #RRGGBB"


# ── fit_log_linear_trend (raw arrays) ──────────────────────────────────


class TestFitLogLinearTrend:
    """Tests for the shared trend fitting function on raw arrays."""

    def test_exponential_growth_detected(self):
        """8% annual growth should be detected within ±3%."""
        years = np.arange(2012, 2025)
        radiance = 5.0 * (1.08 ** (years - 2012))
        result = fit_log_linear_trend(years, radiance)
        assert result["annual_pct_change"] == pytest.approx(8.0, abs=3.0)

    def test_flat_trend_near_zero(self):
        years = np.arange(2012, 2025)
        radiance = np.full(len(years), 5.0)
        result = fit_log_linear_trend(years, radiance)
        assert abs(result["annual_pct_change"]) < 1.0

    def test_insufficient_years_returns_nan(self):
        result = fit_log_linear_trend([2020], [5.0], min_years=2)
        assert np.isnan(result["annual_pct_change"])
        assert result["n_years"] == 1

    def test_two_years_minimum(self):
        result = fit_log_linear_trend([2020, 2021], [5.0, 5.4], min_years=2)
        assert not np.isnan(result["annual_pct_change"])

    def test_returns_residuals(self):
        years = np.arange(2012, 2025)
        radiance = 5.0 * (1.08 ** (years - 2012))
        result = fit_log_linear_trend(years, radiance)
        assert len(result["residuals"]) == len(years)

    def test_returns_beta(self):
        years = np.arange(2012, 2025)
        radiance = 5.0 * (1.08 ** (years - 2012))
        result = fit_log_linear_trend(years, radiance)
        assert not np.isnan(result["beta"])

    def test_r_squared_high_for_clean_data(self):
        years = np.arange(2012, 2025)
        radiance = 5.0 * (1.08 ** (years - 2012))
        result = fit_log_linear_trend(years, radiance)
        assert result["r_squared"] > 0.95

    def test_p_value_significant_for_trend(self):
        years = np.arange(2012, 2025)
        radiance = 5.0 * (1.08 ** (years - 2012))
        result = fit_log_linear_trend(years, radiance)
        assert result["p_value"] < 0.05

    def test_bootstrap_ci_contains_point_estimate(self):
        years = np.arange(2012, 2025)
        radiance = 5.0 * (1.08 ** (years - 2012))
        result = fit_log_linear_trend(years, radiance)
        assert result["ci_low"] <= result["annual_pct_change"] <= result["ci_high"]

    def test_bootstrap_ci_reasonable_width(self):
        years = np.arange(2012, 2025)
        radiance = 5.0 * (1.08 ** (years - 2012))
        result = fit_log_linear_trend(years, radiance)
        ci_width = result["ci_high"] - result["ci_low"]
        assert ci_width < 20.0

    def test_different_seeds_produce_similar_estimates(self):
        """Point estimates should agree within 0.5%; CIs should overlap."""
        years = np.arange(2012, 2025)
        radiance = 5.0 * (1.08 ** (years - 2012))
        r1 = fit_log_linear_trend(years, radiance, seed=42)
        r2 = fit_log_linear_trend(years, radiance, seed=123)
        r3 = fit_log_linear_trend(years, radiance, seed=789)
        # Point estimates identical (deterministic OLS)
        assert r1["annual_pct_change"] == pytest.approx(r2["annual_pct_change"], abs=0.01)
        # CIs should overlap
        assert r1["ci_low"] < r2["ci_high"] and r2["ci_low"] < r1["ci_high"]
        assert r1["ci_low"] < r3["ci_high"] and r3["ci_low"] < r1["ci_high"]

    def test_log_epsilon_prevents_log_zero(self):
        years = np.arange(2012, 2025)
        radiance = np.zeros(len(years))
        result = fit_log_linear_trend(years, radiance)
        assert np.isfinite(result["annual_pct_change"])


# ── Constants validation ───────────────────────────────────────────────


class TestSkyBrightnessConstants:
    # NATURAL_SKY_BRIGHTNESS == 22.0 is verified with full Falchi citation
    # in test_research_validation.py — no need to duplicate here.

    def test_radiance_to_mcd_positive(self):
        assert RADIANCE_TO_MCD > 0

    def test_reference_mcd_positive(self):
        assert REFERENCE_MCD > 0

    def test_bortle_covers_all_classes(self):
        assert set(BORTLE_THRESHOLDS.keys()) == set(range(1, 10))

    def test_bortle_thresholds_ordered(self):
        """Higher Bortle class = brighter sky = lower mag/arcsec²."""
        for cls in range(1, 9):
            mag_min_curr = BORTLE_THRESHOLDS[cls][0]
            mag_min_next = BORTLE_THRESHOLDS[cls + 1][0]
            assert mag_min_curr > mag_min_next, (
                f"Bortle {cls} min ({mag_min_curr}) should be > "
                f"Bortle {cls + 1} min ({mag_min_next})"
            )


class TestSpatialConstants:
    def test_earth_radius(self):
        assert EARTH_RADIUS_KM == pytest.approx(6371.0)

    def test_four_directions(self):
        assert set(DIRECTION_DEFINITIONS.keys()) == {"north", "east", "south", "west"}

    def test_direction_angles_complete(self):
        for direction, defn in DIRECTION_DEFINITIONS.items():
            assert "start_angle" in defn
            assert "end_angle" in defn
            assert "radian" in defn


class TestBenchmarkConstants:
    def test_required_benchmarks_present(self):
        assert "global_average" in PUBLISHED_BENCHMARKS
        assert "india_national" in PUBLISHED_BENCHMARKS

    def test_benchmark_has_required_fields(self):
        for name, bm in PUBLISHED_BENCHMARKS.items():
            assert "source" in bm, f"{name} missing source"
            assert "annual_growth_pct" in bm, f"{name} missing annual_growth_pct"
            assert "ci_low" in bm, f"{name} missing ci_low"
            assert "ci_high" in bm, f"{name} missing ci_high"
            assert bm["ci_low"] < bm["annual_growth_pct"] < bm["ci_high"]

    def test_interpretation_threshold_positive(self):
        assert BENCHMARK_INTERPRETATION_THRESHOLD > 0


class TestDiagnosticsThresholds:
    def test_outlier_z_positive(self):
        assert OUTLIER_Z_THRESHOLD > 0

    def test_dw_bounds_symmetric_around_2(self):
        assert DW_WARNING_LOW < 2.0 < DW_WARNING_HIGH

    def test_jb_threshold_standard_alpha(self):
        assert JB_P_THRESHOLD == 0.05

    def test_cooks_d_positive(self):
        assert COOKS_D_THRESHOLD > 0

    def test_r_squared_between_0_and_1(self):
        assert 0 < R_SQUARED_WARNING < 1

    def test_cv_thresholds_ordered(self):
        assert CV_STABLE_THRESHOLD < CV_ERRATIC_THRESHOLD


class TestFittingConstants:
    def test_exp_decay_bounds_shape(self):
        lower, upper = EXP_DECAY_BOUNDS
        assert len(lower) == 3
        assert len(upper) == 3

    def test_maxfev_positive(self):
        assert EXP_DECAY_MAXFEV > 0

    def test_background_threshold_positive(self):
        assert LIGHT_DOME_BACKGROUND_THRESHOLD > 0


class TestQualityConstants:
    def test_lit_mask_threshold_positive(self):
        assert LIT_MASK_THRESHOLD > 0

    def test_cf_cvg_range_valid(self):
        assert CF_CVG_VALID_RANGE[0] < CF_CVG_VALID_RANGE[1]
        assert CF_CVG_VALID_RANGE[0] >= 0


class TestEcologyConstants:
    def test_land_cover_has_entries(self):
        assert len(LAND_COVER_CLASSES) >= 5

    def test_sensitivity_values_in_range(self):
        for name, val in ECOLOGICAL_SENSITIVITY.items():
            assert 0 <= val <= 1, f"{name} sensitivity {val} out of [0,1]"

    def test_sensitivity_keys_match_land_cover_values(self):
        lc_names = set(LAND_COVER_CLASSES.values())
        sens_names = set(ECOLOGICAL_SENSITIVITY.keys())
        assert lc_names == sens_names
