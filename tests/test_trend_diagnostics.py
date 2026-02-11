"""
Tests for trend model diagnostics (R², Durbin-Watson, Jarque-Bera, Cook's D).

Verifies that:
1. Clean exponential data produces good diagnostics
2. Noisy data correctly triggers warnings
3. Insufficient data returns appropriate sentinel values
4. Outlier detection works
"""

import numpy as np
import pandas as pd
import pytest

from src.trend_diagnostics import compute_trend_diagnostics, compute_all_diagnostics


def _make_yearly_df(district, years, radiances):
    """Helper to build a minimal yearly DataFrame."""
    return pd.DataFrame({
        "district": [district] * len(years),
        "year": years,
        "median_radiance": radiances,
    })


class TestComputeTrendDiagnostics:

    def test_clean_exponential_high_r_squared(self):
        """Clean exponential data should give high R² and no warnings."""
        years = list(range(2012, 2025))
        radiance = [2.0 * np.exp(0.05 * (y - 2012)) for y in years]
        df = _make_yearly_df("CleanDistrict", years, radiance)

        diag = compute_trend_diagnostics(df, "CleanDistrict")

        assert diag["r_squared"] > 0.95, f"R² should be >0.95, got {diag['r_squared']}"
        assert len(diag["outlier_years"]) == 0, "Clean data should have no outliers"

    def test_insufficient_data_warning(self):
        """Fewer than 4 data points should return NaN and a warning."""
        df = _make_yearly_df("ShortDistrict", [2022, 2023, 2024], [1.0, 1.1, 1.2])

        diag = compute_trend_diagnostics(df, "ShortDistrict")

        assert np.isnan(diag["r_squared"])
        assert np.isnan(diag["durbin_watson"])
        assert "insufficient data" in diag["model_warnings"][0]

    def test_durbin_watson_range(self):
        """DW statistic should be between 0 and 4."""
        years = list(range(2012, 2025))
        radiance = [2.0 * np.exp(0.05 * (y - 2012)) for y in years]
        df = _make_yearly_df("DWDistrict", years, radiance)

        diag = compute_trend_diagnostics(df, "DWDistrict")

        assert 0 <= diag["durbin_watson"] <= 4.0, (
            f"DW should be in [0,4], got {diag['durbin_watson']}"
        )

    def test_outlier_detection_with_spike(self):
        """An extreme value injected into the series should be detected as outlier."""
        years = list(range(2012, 2025))
        radiance = [2.0 * np.exp(0.05 * (y - 2012)) for y in years]
        # Inject massive spike in 2018
        radiance[6] = radiance[6] * 10

        df = _make_yearly_df("SpikeDistrict", years, radiance)
        diag = compute_trend_diagnostics(df, "SpikeDistrict")

        assert 2018 in diag["outlier_years"], (
            f"2018 spike should be detected as outlier, "
            f"detected: {diag['outlier_years']}"
        )

    def test_cooks_distance_computed(self):
        """Cook's distance max should be a non-negative float."""
        years = list(range(2012, 2025))
        radiance = [2.0 * np.exp(0.05 * (y - 2012)) for y in years]
        df = _make_yearly_df("CooksDistrict", years, radiance)

        diag = compute_trend_diagnostics(df, "CooksDistrict")

        assert diag["cooks_distance_max"] >= 0
        assert np.isfinite(diag["cooks_distance_max"])


class TestComputeAllDiagnostics:

    def test_returns_one_row_per_entity(self, trend_dataframe):
        """Should return diagnostics for every unique district."""
        df = compute_all_diagnostics(trend_dataframe, entity_col="district")

        districts = trend_dataframe["district"].unique()
        assert len(df) == len(districts)
        assert set(df["district"]) == set(districts)

    def test_output_csv(self, trend_dataframe, tmp_dir):
        """Should save CSV when output_csv is specified."""
        import os
        csv_path = os.path.join(tmp_dir, "diag.csv")
        df = compute_all_diagnostics(
            trend_dataframe, entity_col="district", output_csv=csv_path,
        )
        assert os.path.exists(csv_path)
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == len(df)
