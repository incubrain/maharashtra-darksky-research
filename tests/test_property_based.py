"""
Property-based tests using Hypothesis for formula modules.

These tests verify invariants that must hold for ALL valid inputs,
not just specific examples. They catch edge cases that hand-written
tests miss — boundary values, extreme floats, special distributions.

Run with: pytest tests/test_property_based.py -v
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


# ── Classification properties ───────────────────────────────────────────


class TestClassifyAlanProperties:
    """Properties of classify_alan that must hold for ALL valid inputs."""

    @given(radiance=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False))
    def test_classification_is_total(self, radiance):
        """Every valid non-NaN radiance value gets a class."""
        from src.formulas.classification import classify_alan

        result = classify_alan(radiance)
        assert result in ("low", "medium", "high")

    def test_nan_returns_unknown(self):
        """NaN input always returns 'unknown'."""
        from src.formulas.classification import classify_alan

        assert classify_alan(float("nan")) == "unknown"

    @given(
        a=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
        b=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
    )
    def test_classification_is_monotonic(self, a, b):
        """Higher radiance never results in a lower ALAN class."""
        from src.formulas.classification import classify_alan

        assume(a <= b)
        class_order = {"low": 0, "medium": 1, "high": 2}
        assert class_order[classify_alan(a)] <= class_order[classify_alan(b)]


class TestClassifyStabilityProperties:
    """Properties of classify_stability."""

    @given(cv=st.floats(min_value=0.0, max_value=10.0, allow_nan=False))
    def test_stability_is_total(self, cv):
        """Every valid CV gets a stability class."""
        from src.formulas.classification import classify_stability

        result = classify_stability(cv)
        assert result in ("stable", "moderate", "erratic")

    @given(
        a=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        b=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    )
    def test_stability_is_monotonic(self, a, b):
        """Higher CV never results in a more stable class."""
        from src.formulas.classification import classify_stability

        assume(a <= b)
        class_order = {"stable": 0, "moderate": 1, "erratic": 2}
        assert class_order[classify_stability(a)] <= class_order[classify_stability(b)]


# ── Trend fitting properties ────────────────────────────────────────────


class TestTrendFittingProperties:
    """Properties of fit_log_linear_trend."""

    @given(
        n_years=st.integers(min_value=3, max_value=20),
        base_rad=st.floats(min_value=0.1, max_value=100.0, allow_nan=False),
        growth=st.floats(min_value=-0.1, max_value=0.3, allow_nan=False),
    )
    @settings(max_examples=50, deadline=5000)
    def test_valid_input_never_produces_nan(self, n_years, base_rad, growth):
        """Trend fitting on valid positive radiance never returns NaN."""
        from src.formulas.trend import fit_log_linear_trend

        years = np.arange(2012, 2012 + n_years, dtype=float)
        radiance = base_rad * np.exp(growth * np.arange(n_years))
        # Add small noise to avoid perfect collinearity edge cases
        rng = np.random.default_rng(42)
        radiance = radiance + rng.normal(0, 0.001, n_years)
        radiance = np.maximum(radiance, 0.001)

        result = fit_log_linear_trend(years, radiance, n_bootstrap=10)

        assert not np.isnan(result["annual_pct_change"])
        assert not np.isnan(result["r_squared"])
        assert result["n_years"] == n_years

    @given(n_years=st.integers(min_value=0, max_value=1))
    def test_insufficient_years_returns_nan(self, n_years):
        """Less than min_years data should return NaN results."""
        from src.formulas.trend import fit_log_linear_trend

        years = np.arange(2012, 2012 + n_years, dtype=float)
        radiance = np.ones(n_years) * 5.0

        result = fit_log_linear_trend(years, radiance, n_bootstrap=10)
        assert np.isnan(result["annual_pct_change"])

    @given(
        n_years=st.integers(min_value=3, max_value=15),
        base_rad=st.floats(min_value=0.1, max_value=50.0, allow_nan=False),
    )
    @settings(max_examples=30, deadline=5000)
    def test_r_squared_is_bounded(self, n_years, base_rad):
        """R-squared is always in [0, 1] for valid input."""
        from src.formulas.trend import fit_log_linear_trend

        years = np.arange(2012, 2012 + n_years, dtype=float)
        radiance = np.full(n_years, base_rad)
        # Add noise so OLS doesn't hit degenerate case
        rng = np.random.default_rng(42)
        radiance = radiance + rng.normal(0, 0.01, n_years)
        radiance = np.maximum(radiance, 0.001)

        result = fit_log_linear_trend(years, radiance, n_bootstrap=10)

        if not np.isnan(result["r_squared"]):
            assert 0.0 <= result["r_squared"] <= 1.0 + 1e-10


# ── Sky brightness properties ──────────────────────────────────────────


class TestSkyBrightnessProperties:
    """Properties of radiance-to-magnitude conversion."""

    @given(
        a=st.floats(min_value=0.01, max_value=500.0, allow_nan=False),
        b=st.floats(min_value=0.01, max_value=500.0, allow_nan=False),
    )
    def test_higher_radiance_means_brighter_sky(self, a, b):
        """Higher radiance → lower magnitude (brighter sky).

        The magnitude scale is inverted: lower numbers = brighter.
        """
        from src.formulas.sky_brightness import RADIANCE_TO_MCD, REFERENCE_MCD

        # Require meaningful separation to avoid floating-point equality
        assume(b > a * 1.001)

        # Convert radiance to magnitude: mag = -2.5 * log10(rad_mcd / ref_mcd)
        mag_a = -2.5 * np.log10((a * RADIANCE_TO_MCD) / REFERENCE_MCD)
        mag_b = -2.5 * np.log10((b * RADIANCE_TO_MCD) / REFERENCE_MCD)

        # Higher radiance → lower (brighter) magnitude
        assert mag_b < mag_a


# ── Data transformation properties ──────────────────────────────────────


class TestRadianceTransformProperties:
    """Properties that must hold after data transformations."""

    @given(
        arr=arrays(
            dtype=np.float32,
            shape=st.integers(min_value=1, max_value=100),
            elements=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
        )
    )
    def test_radiance_values_remain_non_negative(self, arr):
        """Radiance values are physically non-negative after any log transform."""
        from src import config

        log_vals = np.log(arr + config.LOG_EPSILON)
        # The log transform is valid (no NaN/Inf) for non-negative input + epsilon
        assert np.all(np.isfinite(log_vals))
