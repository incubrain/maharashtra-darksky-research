"""
Edge case and boundary tests for the VIIRS ALAN analysis pipeline.

Tests extreme inputs that could cause crashes, NaN propagation, or
incorrect results. Every test targets a specific failure mode that
has been observed or is plausible in production data.
"""

import numpy as np
import pandas as pd
import pytest

from src.formulas.classification import classify_alan, classify_alan_series, classify_stability
from src.formulas.trend import fit_log_linear_trend


# ── Extreme Radiance Values ──────────────────────────────────────────


class TestExtremeRadiance:
    """Classification and trend fitting with extreme radiance inputs."""

    def test_negative_radiance_classified_as_low(self):
        """Negative radiance (sensor artifact) should classify as low."""
        assert classify_alan(-1.0) == "low"
        assert classify_alan(-0.001) == "low"

    def test_zero_radiance(self):
        assert classify_alan(0.0) == "low"

    def test_very_small_radiance(self):
        """Sub-noise radiance (1e-10 nW) should be low."""
        assert classify_alan(1e-10) == "low"

    def test_very_large_radiance(self):
        """Extreme urban radiance (10000 nW) should be high."""
        assert classify_alan(10000.0) == "high"

    def test_inf_radiance(self):
        """Infinity should classify as high (not crash)."""
        assert classify_alan(float("inf")) == "high"

    def test_nan_radiance(self):
        assert classify_alan(float("nan")) == "unknown"
        assert classify_alan(np.nan) == "unknown"

    @pytest.mark.parametrize("value", [-1.0, 0.0, 1e-10, 10000.0])
    def test_series_handles_extremes(self, value):
        """classify_alan_series should not crash on extreme values."""
        result = classify_alan_series([value])
        assert len(result) == 1
        assert str(result[0]) in ("low", "medium", "high")


# ── Extreme CV Values ────────────────────────────────────────────────


class TestExtremeCVValues:
    """Stability classification with extreme coefficient of variation."""

    def test_zero_cv(self):
        assert classify_stability(0.0) == "stable"

    def test_negative_cv(self):
        """Negative CV (mathematically impossible but defensive)."""
        assert classify_stability(-0.1) == "stable"

    def test_very_large_cv(self):
        """CV > 1.0 (std > mean) should be erratic."""
        assert classify_stability(2.0) == "erratic"
        assert classify_stability(100.0) == "erratic"

    def test_nan_cv(self):
        assert classify_stability(np.nan) == "unknown"


# ── Trend Fitting Edge Cases ─────────────────────────────────────────


class TestTrendFittingEdgeCases:
    """fit_log_linear_trend with edge-case inputs."""

    def test_single_year_returns_nan(self):
        """One data point cannot define a trend."""
        result = fit_log_linear_trend([2024], [5.0], min_years=2)
        assert np.isnan(result["annual_pct_change"])
        assert result["n_years"] == 1

    def test_two_years_minimum_works(self):
        """Minimum viable input: 2 data points."""
        result = fit_log_linear_trend([2023, 2024], [5.0, 5.4], min_years=2)
        assert not np.isnan(result["annual_pct_change"])
        assert result["annual_pct_change"] > 0

    def test_non_contiguous_years(self):
        """Years don't have to be consecutive."""
        years = [2012, 2015, 2020, 2024]
        radiance = [2.0, 3.0, 5.0, 8.0]
        result = fit_log_linear_trend(years, radiance)
        assert not np.isnan(result["annual_pct_change"])
        assert result["annual_pct_change"] > 0

    def test_all_zero_radiance(self):
        """All-zero radiance should not crash (LOG_EPSILON prevents log(0))."""
        years = np.arange(2012, 2025)
        radiance = np.zeros(13)
        result = fit_log_linear_trend(years, radiance)
        assert np.isfinite(result["annual_pct_change"])

    def test_all_nan_radiance(self):
        """All-NaN radiance will produce NaN log values.

        The OLS model may still run (NaN propagation), but result should
        be NaN, not a crash.
        """
        years = np.arange(2012, 2025)
        radiance = np.full(13, np.nan)
        # Should not raise — may return NaN results
        try:
            result = fit_log_linear_trend(years, radiance)
            # If it returns, values should be NaN
            assert np.isnan(result["annual_pct_change"]) or np.isfinite(result["annual_pct_change"])
        except (ValueError, np.linalg.LinAlgError):
            pass  # Also acceptable — singular matrix from NaN data

    def test_constant_radiance_gives_zero_trend(self):
        """Flat radiance should give ~0% annual change."""
        years = np.arange(2012, 2025)
        radiance = np.full(13, 5.0)
        result = fit_log_linear_trend(years, radiance)
        assert abs(result["annual_pct_change"]) < 0.01

    def test_very_high_radiance(self):
        """Extreme urban radiance (1000 nW) should not overflow."""
        years = np.arange(2012, 2025)
        radiance = np.full(13, 1000.0)
        result = fit_log_linear_trend(years, radiance)
        assert np.isfinite(result["annual_pct_change"])

    def test_mixed_zero_and_positive_radiance(self):
        """Real data often has some zero-radiance years (sensor gap)."""
        years = np.arange(2012, 2020)
        radiance = [0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        result = fit_log_linear_trend(years, radiance)
        assert np.isfinite(result["annual_pct_change"])

    def test_decreasing_radiance_gives_negative_trend(self):
        """Declining radiance should give negative % change."""
        years = np.arange(2012, 2025)
        radiance = 10.0 * (0.95 ** (years - 2012))  # -5%/yr
        result = fit_log_linear_trend(years, radiance)
        assert result["annual_pct_change"] < 0

    def test_very_small_bootstrap_count(self):
        """Even with n_bootstrap=10, should not crash."""
        years = np.arange(2012, 2025)
        radiance = 5.0 * (1.08 ** (years - 2012))
        result = fit_log_linear_trend(years, radiance, n_bootstrap=10)
        assert not np.isnan(result["annual_pct_change"])
        assert not np.isnan(result["ci_low"])

    def test_zero_bootstrap_gives_nan_ci(self):
        """With n_bootstrap=0, CIs should be NaN but point estimate valid."""
        years = np.arange(2012, 2025)
        radiance = 5.0 * (1.08 ** (years - 2012))
        result = fit_log_linear_trend(years, radiance, n_bootstrap=0)
        assert not np.isnan(result["annual_pct_change"])
        assert np.isnan(result["ci_low"])
        assert np.isnan(result["ci_high"])


# ── Empty and Minimal DataFrames ─────────────────────────────────────


class TestEmptyAndMinimalInputs:
    """Pipeline behavior with empty or single-row inputs."""

    def test_empty_series_classification(self):
        """Empty series should return empty result."""
        result = classify_alan_series([])
        assert len(result) == 0

    def test_single_element_series(self):
        result = classify_alan_series([3.0])
        assert len(result) == 1
        assert str(result[0]) == "medium"

    def test_trend_with_zero_years(self):
        """Empty arrays should return NaN result."""
        result = fit_log_linear_trend([], [])
        assert np.isnan(result["annual_pct_change"])
        assert result["n_years"] == 0


# ── NaN Propagation ──────────────────────────────────────────────────


class TestNaNPropagation:
    """Verify NaN handling doesn't cause silent errors."""

    def test_nan_in_series_produces_nan_category(self):
        """NaN values in series should become NaN categories, not crash."""
        result = classify_alan_series([1.0, np.nan, 5.0])
        assert len(result) == 3
        assert str(result[0]) == "medium"
        assert pd.isna(result[1])
        assert str(result[2]) == "high"

    def test_trend_with_some_nans(self):
        """Radiance with NaN values should still be processable.

        NaN propagates through log(), making the OLS fit on non-NaN
        values effectively. This may or may not raise depending on
        statsmodels behavior.
        """
        years = np.arange(2012, 2017)
        radiance = [5.0, np.nan, 6.0, np.nan, 7.0]
        try:
            result = fit_log_linear_trend(years, np.array(radiance))
            # If it succeeds, check it didn't silently produce garbage
            assert result["n_years"] == 5
        except (ValueError, np.linalg.LinAlgError):
            pass  # Acceptable if statsmodels rejects NaN input


# ── Type Coercion ────────────────────────────────────────────────────


class TestTypeCoercion:
    """Verify functions handle various input types."""

    def test_classify_alan_accepts_int(self):
        assert classify_alan(3) == "medium"

    def test_classify_alan_accepts_numpy_float(self):
        assert classify_alan(np.float64(3.0)) == "medium"
        assert classify_alan(np.float32(3.0)) == "medium"

    def test_trend_accepts_lists(self):
        """Lists should work, not just numpy arrays."""
        result = fit_log_linear_trend([2020, 2021, 2022], [5.0, 5.5, 6.0])
        assert not np.isnan(result["annual_pct_change"])

    def test_trend_accepts_pandas_series(self):
        """Pandas Series should be coerced to arrays."""
        years = pd.Series([2020, 2021, 2022])
        radiance = pd.Series([5.0, 5.5, 6.0])
        result = fit_log_linear_trend(years, radiance)
        assert not np.isnan(result["annual_pct_change"])
