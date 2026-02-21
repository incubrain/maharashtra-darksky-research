"""
Tests for ALAN classification logic (threshold-based and percentile-based).

Verifies that:
1. Low/medium/high thresholds are applied correctly
2. Edge cases at threshold boundaries work
3. NaN values are handled
4. Scalar and series classification produce consistent results
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.formulas.classification import classify_alan, classify_alan_series


class TestALANThresholdClassification:
    """Test the threshold classification using the actual production function."""

    def test_low_threshold(self):
        assert classify_alan(0.5) == "low"
        assert classify_alan(0.0) == "low"
        assert classify_alan(0.99) == "low"

    def test_medium_threshold(self):
        assert classify_alan(1.0) == "medium"
        assert classify_alan(3.0) == "medium"
        assert classify_alan(4.99) == "medium"

    def test_high_threshold(self):
        assert classify_alan(5.0) == "high"
        assert classify_alan(50.0) == "high"
        assert classify_alan(100.0) == "high"

    def test_boundary_low_medium(self):
        """Exactly at ALAN_LOW_THRESHOLD (1.0) should be medium, not low."""
        assert classify_alan(config.ALAN_LOW_THRESHOLD) == "medium"

    def test_boundary_medium_high(self):
        """Exactly at ALAN_MEDIUM_THRESHOLD (5.0) should be high, not medium."""
        assert classify_alan(config.ALAN_MEDIUM_THRESHOLD) == "high"

    def test_nan_returns_unknown(self):
        assert classify_alan(np.nan) == "unknown"
        assert classify_alan(None) == "unknown"

    def test_config_thresholds_are_correct(self):
        """Verify expected config values haven't been accidentally changed."""
        assert config.ALAN_LOW_THRESHOLD == 1.0
        assert config.ALAN_MEDIUM_THRESHOLD == 5.0


class TestSiteALANClassification:
    """Test the series-based classification using the actual production function."""

    def test_site_classification_values(self):
        """classify_alan_series uses [low, high) intervals matching the scalar method."""
        values = [0.5, 1.0, 3.0, 5.0, 10.0]
        result = classify_alan_series(pd.Series(values))

        assert result.iloc[0] == "low"     # 0.5 in [-inf, 1.0)
        assert result.iloc[1] == "medium"  # 1.0 in [1.0, 5.0)
        assert result.iloc[2] == "medium"  # 3.0 in [1.0, 5.0)
        assert result.iloc[3] == "high"    # 5.0 in [5.0, inf)
        assert result.iloc[4] == "high"    # 10.0 in [5.0, inf)

    def test_boundary_consistency_with_scalar_method(self):
        """Series (classify_alan_series) and scalar (classify_alan) should agree at all boundaries."""
        for value in [1.0, 5.0, 0.5, 3.0, 10.0]:
            scalar_result = classify_alan(value)
            series_result = classify_alan_series(pd.Series([value])).iloc[0]
            assert scalar_result == series_result, (
                f"At {value} nW: scalar says '{scalar_result}', "
                f"series says '{series_result}'"
            )
