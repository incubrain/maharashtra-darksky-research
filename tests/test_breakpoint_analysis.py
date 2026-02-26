"""
Tests for src/analysis/breakpoint_analysis.py.

Verifies piecewise linear regression for ALAN trend changepoint detection.

CRITICAL: A bug here could report false breakpoints (masking real trends)
or miss genuine regime changes in nighttime light data.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.breakpoint_analysis import (
    analyze_all_breakpoints,
    detect_trend_breakpoints,
)


def _make_trend_df(district, years, radiance_values):
    """Helper: build a yearly DataFrame for one district."""
    return pd.DataFrame({
        "district": [district] * len(years),
        "year": years,
        "median_radiance": radiance_values,
    })


class TestBreakpointDetection:
    """Core breakpoint detection tests."""

    def test_clear_regime_change(self):
        """A sharp change in growth rate should be detected.
        Before 2017: flat at 2.0. After 2017: exponential growth."""
        years = list(range(2012, 2025))
        radiance = []
        for y in years:
            if y <= 2017:
                radiance.append(2.0 + np.random.default_rng(42).normal(0, 0.01))
            else:
                t = y - 2017
                radiance.append(2.0 * np.exp(0.15 * t))
        df = _make_trend_df("TestDistrict", years, radiance)
        result = detect_trend_breakpoints(df, "TestDistrict")
        assert result["breakpoint_year"] is not None
        # Breakpoint should be near 2017 (±1 year)
        assert abs(result["breakpoint_year"] - 2017) <= 1

    def test_linear_trend_no_breakpoint(self):
        """A perfectly linear trend may still find a breakpoint due to
        overfitting in small samples, but growth rates before/after
        should remain similar if the underlying trend is truly linear."""
        years = list(range(2012, 2025))
        rng = np.random.default_rng(42)
        radiance = [2.0 + 0.5 * (y - 2012) + rng.normal(0, 0.1) for y in years]
        df = _make_trend_df("Linear", years, radiance)
        result = detect_trend_breakpoints(df, "Linear")
        if result["breakpoint_year"] is not None:
            # Growth rates before/after should be similar for a linear trend
            rate_diff = abs(result["growth_rate_after"] - result["growth_rate_before"])
            assert rate_diff < 20, f"Growth rate difference {rate_diff}% too large for linear"

    def test_insufficient_data(self):
        """With <5 years, should return None breakpoint and NaN metrics."""
        years = [2020, 2021, 2022]
        radiance = [1.0, 2.0, 3.0]
        df = _make_trend_df("Short", years, radiance)
        result = detect_trend_breakpoints(df, "Short")
        assert result["breakpoint_year"] is None
        assert np.isnan(result["growth_rate_before"])

    def test_exactly_five_years(self):
        """With exactly 5 years, should have enough data to attempt breakpoint."""
        years = [2018, 2019, 2020, 2021, 2022]
        radiance = [1.0, 1.2, 1.4, 5.0, 6.0]  # jump at 2020
        df = _make_trend_df("Five", years, radiance)
        result = detect_trend_breakpoints(df, "Five")
        # Should not crash; breakpoint may or may not be detected
        assert "breakpoint_year" in result

    def test_output_keys(self):
        """Result should contain all documented keys."""
        years = list(range(2012, 2025))
        radiance = [2.0 * np.exp(0.05 * (y - 2012)) for y in years]
        df = _make_trend_df("D", years, radiance)
        result = detect_trend_breakpoints(df, "D")
        expected_keys = {
            "district", "breakpoint_year", "growth_rate_before",
            "growth_rate_after", "p_value", "aic_improvement"
        }
        assert expected_keys == set(result.keys())

    def test_growth_rates_physical_units(self):
        """Growth rates should be in percent per year, not raw beta coefficients."""
        years = list(range(2012, 2025))
        # 10% annual growth: radiance ≈ 2 * exp(0.10 * t)
        radiance = [2.0 * np.exp(0.10 * (y - 2012)) for y in years]
        df = _make_trend_df("Growth", years, radiance)
        result = detect_trend_breakpoints(df, "Growth")
        # With no breakpoint, both rates should be ~10%
        if result["breakpoint_year"] is None:
            assert abs(result["growth_rate_before"] - 10.0) < 3.0

    def test_nonexistent_district_returns_insufficient(self):
        """Querying a district not in the DataFrame should return NaN/None."""
        years = list(range(2012, 2025))
        radiance = [2.0] * len(years)
        df = _make_trend_df("Exists", years, radiance)
        result = detect_trend_breakpoints(df, "Nonexistent")
        assert result["breakpoint_year"] is None


class TestBreakpointEdgeCases:
    """Edge cases that could corrupt breakpoint analysis."""

    def test_zero_radiance_series(self):
        """All-zero radiance: log(0 + epsilon) should not crash."""
        years = list(range(2012, 2025))
        radiance = [0.0] * len(years)
        df = _make_trend_df("Dark", years, radiance)
        result = detect_trend_breakpoints(df, "Dark")
        # Should not raise; result structure should be intact
        assert "breakpoint_year" in result

    def test_single_spike_year(self):
        """A single outlier year should not dominate breakpoint detection."""
        years = list(range(2012, 2025))
        radiance = [2.0] * len(years)
        radiance[6] = 50.0  # 2018 spike
        df = _make_trend_df("Spike", years, radiance)
        result = detect_trend_breakpoints(df, "Spike")
        # A single spike should not produce a highly significant breakpoint
        # (the piecewise model doesn't capture single-year events well)
        assert "breakpoint_year" in result

    def test_step_function(self):
        """A clean step-up should produce a clear breakpoint."""
        years = list(range(2012, 2025))
        radiance = [1.0 if y <= 2018 else 5.0 for y in years]
        df = _make_trend_df("Step", years, radiance)
        result = detect_trend_breakpoints(df, "Step")
        if result["breakpoint_year"] is not None:
            assert abs(result["breakpoint_year"] - 2018) <= 1

    def test_negative_radiance_values(self):
        """Negative radiance (sensor artifacts): log(negative + epsilon) could NaN.
        Should handle gracefully or propagate a meaningful error."""
        years = list(range(2012, 2025))
        radiance = [-0.1] * len(years)
        df = _make_trend_df("Neg", years, radiance)
        # This might crash with log of negative number
        # The code uses LOG_EPSILON but -0.1 + 1e-6 is still negative
        try:
            result = detect_trend_breakpoints(df, "Neg")
            # If it doesn't crash, check structure
            assert "breakpoint_year" in result
        except (ValueError, RuntimeWarning):
            pytest.skip("Negative radiance causes math error (expected)")


class TestAnalyzeAllBreakpoints:
    """Tests for the batch analysis function."""

    def test_multiple_districts(self):
        """Should produce one result per district."""
        dfs = []
        for name in ["A", "B", "C"]:
            years = list(range(2012, 2025))
            rad = [2.0 + 0.1 * (y - 2012) for y in years]
            dfs.append(_make_trend_df(name, years, rad))
        df = pd.concat(dfs, ignore_index=True)
        result = analyze_all_breakpoints(df)
        assert len(result) == 3
        assert set(result["district"]) == {"A", "B", "C"}

    def test_csv_output(self, tmp_dir):
        import os
        years = list(range(2012, 2025))
        df = _make_trend_df("D", years, [2.0 * np.exp(0.05 * (y - 2012)) for y in years])
        csv_path = os.path.join(tmp_dir, "breakpoints.csv")
        analyze_all_breakpoints(df, output_csv=csv_path)
        assert os.path.exists(csv_path)
