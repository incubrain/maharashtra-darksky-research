"""
Tests for ALAN classification logic (threshold-based and percentile-based).

Verifies that:
1. Low/medium/high thresholds are applied correctly
2. Edge cases at threshold boundaries work
3. NaN values are handled
"""

import numpy as np
import pandas as pd
import pytest

from src import config


class TestALANThresholdClassification:
    """Test the threshold classification used in viirs_process.run_full_pipeline."""

    @staticmethod
    def classify(median_radiance):
        """Replicates the classification logic from viirs_process.py lines 582-589."""
        if pd.isna(median_radiance):
            return "unknown"
        elif median_radiance < config.ALAN_LOW_THRESHOLD:
            return "low"
        elif median_radiance < config.ALAN_MEDIUM_THRESHOLD:
            return "medium"
        else:
            return "high"

    def test_low_threshold(self):
        assert self.classify(0.5) == "low"
        assert self.classify(0.0) == "low"
        assert self.classify(0.99) == "low"

    def test_medium_threshold(self):
        assert self.classify(1.0) == "medium"
        assert self.classify(3.0) == "medium"
        assert self.classify(4.99) == "medium"

    def test_high_threshold(self):
        assert self.classify(5.0) == "high"
        assert self.classify(50.0) == "high"
        assert self.classify(100.0) == "high"

    def test_boundary_low_medium(self):
        """Exactly at ALAN_LOW_THRESHOLD (1.0) should be medium, not low."""
        assert self.classify(config.ALAN_LOW_THRESHOLD) == "medium"

    def test_boundary_medium_high(self):
        """Exactly at ALAN_MEDIUM_THRESHOLD (5.0) should be high, not medium."""
        assert self.classify(config.ALAN_MEDIUM_THRESHOLD) == "high"

    def test_nan_returns_unknown(self):
        assert self.classify(np.nan) == "unknown"
        assert self.classify(None) == "unknown"

    def test_config_thresholds_are_correct(self):
        """Verify expected config values haven't been accidentally changed."""
        assert config.ALAN_LOW_THRESHOLD == 1.0
        assert config.ALAN_MEDIUM_THRESHOLD == 5.0


class TestSiteALANClassification:
    """Test the pd.cut-based classification used in site_analysis.py."""

    @staticmethod
    def classify_series(radiance_values):
        """Replicates the pd.cut classification from site_analysis.py line 180."""
        return pd.cut(
            pd.Series(radiance_values),
            bins=[-np.inf, config.ALAN_LOW_THRESHOLD, config.ALAN_MEDIUM_THRESHOLD, np.inf],
            labels=["low", "medium", "high"],
            right=False,
        )

    def test_site_classification_values(self):
        """pd.cut with right=False uses [low, high) intervals matching the manual method."""
        values = [0.5, 1.0, 3.0, 5.0, 10.0]
        result = self.classify_series(values)

        assert result.iloc[0] == "low"     # 0.5 in [-inf, 1.0)
        assert result.iloc[1] == "medium"  # 1.0 in [1.0, 5.0)
        assert result.iloc[2] == "medium"  # 3.0 in [1.0, 5.0)
        assert result.iloc[3] == "high"    # 5.0 in [5.0, inf)
        assert result.iloc[4] == "high"    # 10.0 in [5.0, inf)

    def test_boundary_consistency_with_district_method(self):
        """Site (pd.cut) and district (if/elif) should now agree at all boundaries."""
        from tests.test_alan_classification import TestALANThresholdClassification

        for value in [1.0, 5.0, 0.5, 3.0, 10.0]:
            district_result = TestALANThresholdClassification.classify(value)
            site_result = self.classify_series([value]).iloc[0]
            assert district_result == site_result, (
                f"At {value} nW: district says '{district_result}', "
                f"site says '{site_result}'"
            )
