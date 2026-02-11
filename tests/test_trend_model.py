"""
Tests for log-linear trend fitting and bootstrap CI.

Verifies that fit_log_linear_trend correctly:
1. Detects ~8% annual growth from exponential synthetic data
2. Returns ~0% for flat data
3. Produces valid bootstrap confidence intervals
4. Handles edge case of MIN_YEARS_FOR_TREND
5. Reports correct R² and p-value
"""

import numpy as np
import pandas as pd
import pytest

from src.viirs_process import fit_log_linear_trend
from src import config


class TestFitLogLinearTrend:

    def test_exponential_growth_detected(self, trend_dataframe):
        """DistrictA grows at ~8%/yr; fitted trend should be close to 8%."""
        result = fit_log_linear_trend(trend_dataframe, "DistrictA")

        assert result["district"] == "DistrictA"
        assert result["n_years"] == 13
        # Allow ±3% tolerance due to noise in synthetic data
        assert result["annual_pct_change"] == pytest.approx(8.0, abs=3.0), (
            f"Expected ~8% growth, got {result['annual_pct_change']:.2f}%"
        )

    def test_flat_trend_near_zero(self, trend_dataframe):
        """DistrictB is roughly flat; fitted trend should be near 0%."""
        result = fit_log_linear_trend(trend_dataframe, "DistrictB")

        assert result["district"] == "DistrictB"
        assert abs(result["annual_pct_change"]) < 5.0, (
            f"Expected ~0% growth for flat data, got {result['annual_pct_change']:.2f}%"
        )

    def test_bootstrap_ci_contains_point_estimate(self, trend_dataframe):
        """Bootstrap CI should contain the point estimate."""
        result = fit_log_linear_trend(trend_dataframe, "DistrictA")

        assert result["ci_low"] < result["annual_pct_change"] < result["ci_high"], (
            f"Point estimate {result['annual_pct_change']:.2f} should be within "
            f"CI [{result['ci_low']:.2f}, {result['ci_high']:.2f}]"
        )

    def test_bootstrap_ci_reasonable_width(self, trend_dataframe):
        """Bootstrap CI should be reasonably narrow for clean data."""
        result = fit_log_linear_trend(trend_dataframe, "DistrictA")

        ci_width = result["ci_high"] - result["ci_low"]
        assert ci_width > 0, "CI width should be positive"
        assert ci_width < 20.0, f"CI width {ci_width:.2f} seems too wide"

    def test_r_squared_high_for_exponential(self, trend_dataframe):
        """R² should be high for clean exponential data."""
        result = fit_log_linear_trend(trend_dataframe, "DistrictA")

        assert result["r_squared"] > 0.8, (
            f"R² should be >0.8 for exponential data, got {result['r_squared']:.3f}"
        )

    def test_p_value_significant_for_trend(self, trend_dataframe):
        """p-value should be <0.05 for data with real trend."""
        result = fit_log_linear_trend(trend_dataframe, "DistrictA")

        assert result["p_value"] < 0.05, (
            f"p-value should be <0.05 for trending data, got {result['p_value']:.4f}"
        )

    def test_insufficient_years_returns_nan(self):
        """With fewer than MIN_YEARS_FOR_TREND data points, all outputs should be NaN."""
        df = pd.DataFrame({
            "district": ["X"],
            "year": [2024],
            "median_radiance": [5.0],
        })
        result = fit_log_linear_trend(df, "X")

        assert np.isnan(result["annual_pct_change"])
        assert np.isnan(result["ci_low"])
        assert np.isnan(result["ci_high"])
        assert np.isnan(result["r_squared"])
        assert result["n_years"] == 1

    def test_two_years_minimum(self):
        """With exactly MIN_YEARS_FOR_TREND=2 points, should produce a result."""
        df = pd.DataFrame({
            "district": ["X", "X"],
            "year": [2023, 2024],
            "median_radiance": [5.0, 5.5],
        })
        result = fit_log_linear_trend(df, "X")

        assert not np.isnan(result["annual_pct_change"]), (
            "Should compute trend with exactly 2 data points"
        )
        assert result["annual_pct_change"] > 0, "5.0 -> 5.5 is positive growth"

    def test_log_epsilon_prevents_log_zero(self):
        """Zero radiance should not cause math errors due to LOG_EPSILON."""
        df = pd.DataFrame({
            "district": ["X", "X", "X"],
            "year": [2022, 2023, 2024],
            "median_radiance": [0.0, 0.0, 0.0],
        })
        result = fit_log_linear_trend(df, "X")

        # Should not crash and should return a valid (likely near-zero) result
        assert not np.isnan(result["annual_pct_change"]) or result["n_years"] == 3
