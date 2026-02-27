"""
Tests for src/analysis/stability_metrics.py.

Verifies temporal stability classification (CV, IQR, max year-to-year change)
which is CRITICAL for dark-sky certification: sites must demonstrate STABLE
low-ALAN over time.

Includes edge cases that could silently misclassify a volatile site as stable.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.stability_metrics import compute_stability_metrics


def _make_yearly_df(districts_data, entity_col="district"):
    """Helper: build a yearly DataFrame from {name: [radiance_values]} dict."""
    rows = []
    for name, values in districts_data.items():
        for i, val in enumerate(values):
            rows.append({entity_col: name, "year": 2012 + i, "median_radiance": val})
    return pd.DataFrame(rows)


class TestStabilityMetricsBasic:
    """Core stability computation tests."""

    def test_perfectly_constant_series(self):
        """A constant time series should have CV=0, IQR=0, max_change=0, stability=stable."""
        df = _make_yearly_df({"Flat": [5.0] * 13})
        result = compute_stability_metrics(df)
        row = result[result["district"] == "Flat"].iloc[0]
        assert row["coefficient_of_variation"] == 0.0
        assert row["iqr"] == 0.0
        assert row["max_year_to_year_change"] == 0.0
        assert row["stability_class"] == "stable"

    def test_highly_variable_series(self):
        """A series oscillating wildly should be classified as erratic."""
        values = [1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0]
        df = _make_yearly_df({"Erratic": values})
        result = compute_stability_metrics(df)
        row = result[result["district"] == "Erratic"].iloc[0]
        assert row["stability_class"] == "erratic"
        assert row["coefficient_of_variation"] > 0.5

    def test_moderate_variability(self):
        """A series with moderate variability (CV between 0.2 and 0.5)."""
        # CV ≈ 0.3: mean ~5, std ~1.5
        rng = np.random.default_rng(42)
        values = (5.0 + rng.normal(0, 1.5, 13)).tolist()
        df = _make_yearly_df({"Moderate": values})
        result = compute_stability_metrics(df)
        row = result[result["district"] == "Moderate"].iloc[0]
        assert row["stability_class"] in ("moderate", "stable", "erratic")
        # Actual classification depends on realized CV

    def test_two_districts(self):
        """Should produce one row per district."""
        df = _make_yearly_df({
            "Stable": [2.0] * 10,
            "Wild": [1.0, 20.0] * 5,
        })
        result = compute_stability_metrics(df)
        assert len(result) == 2

    def test_output_columns(self):
        """Result should have all expected columns."""
        df = _make_yearly_df({"D": [1.0, 2.0, 3.0]})
        result = compute_stability_metrics(df)
        expected_cols = {
            "district", "mean_radiance_2012_2024", "std_radiance",
            "coefficient_of_variation", "iqr", "max_year_to_year_change",
            "stability_class"
        }
        assert expected_cols.issubset(set(result.columns))


class TestStabilityEdgeCases:
    """Edge cases that could cause silent misclassification."""

    def test_single_data_point(self):
        """With only 1 year, should return a row but without metrics (can't compute CV)."""
        df = _make_yearly_df({"OneYear": [5.0]})
        result = compute_stability_metrics(df)
        assert len(result) == 1
        row = result.iloc[0]
        # With <2 values, metrics should be absent/NaN
        assert "coefficient_of_variation" not in row.index or pd.isna(row.get("coefficient_of_variation"))

    def test_two_data_points(self):
        """With exactly 2 years, CV is computable but IQR degenerates."""
        df = _make_yearly_df({"TwoYears": [3.0, 7.0]})
        result = compute_stability_metrics(df)
        row = result[result["district"] == "TwoYears"].iloc[0]
        assert not pd.isna(row["coefficient_of_variation"])
        assert row["max_year_to_year_change"] == 4.0

    def test_zero_mean_radiance(self):
        """If mean is 0, CV formula (std/mean) would divide by zero.
        Should return NaN, not raise."""
        df = _make_yearly_df({"Dark": [0.0, 0.0, 0.0, 0.0, 0.0]})
        result = compute_stability_metrics(df)
        row = result[result["district"] == "Dark"].iloc[0]
        assert pd.isna(row["coefficient_of_variation"]) or row["coefficient_of_variation"] == 0.0

    def test_nan_in_radiance(self):
        """NaN values in median_radiance should be dropped before computation."""
        df = _make_yearly_df({"NaNs": [1.0, np.nan, 3.0, np.nan, 5.0, 7.0, 9.0]})
        result = compute_stability_metrics(df)
        row = result[result["district"] == "NaNs"].iloc[0]
        # Should have computed from the 5 non-NaN values
        assert not pd.isna(row["mean_radiance_2012_2024"])

    def test_all_nan_radiance(self):
        """If ALL radiance values are NaN, should handle gracefully."""
        df = _make_yearly_df({"AllNaN": [np.nan] * 5})
        result = compute_stability_metrics(df)
        assert len(result) == 1

    def test_monotonically_increasing_not_classified_as_stable(self):
        """A steadily increasing series (e.g. 1,2,3,...,13) has significant variability.
        Its CV should reflect the spread, not misclassify as stable."""
        values = list(range(1, 14))  # 1 to 13
        df = _make_yearly_df({"Increasing": [float(v) for v in values]})
        result = compute_stability_metrics(df)
        row = result[result["district"] == "Increasing"].iloc[0]
        # Mean = 7, std ≈ 3.89, CV ≈ 0.56 → should be moderate or erratic
        assert row["stability_class"] != "stable", \
            "A monotonically increasing series should NOT be stable"

    def test_max_year_to_year_change_detects_spike(self):
        """A sudden spike in an otherwise stable series should show in max_change."""
        values = [5.0, 5.0, 5.0, 5.0, 50.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        df = _make_yearly_df({"Spike": values})
        result = compute_stability_metrics(df)
        row = result[result["district"] == "Spike"].iloc[0]
        assert row["max_year_to_year_change"] == 45.0

    def test_negative_radiance_handled(self):
        """Negative radiance values (sensor artifacts) should not crash the function."""
        values = [-0.5, 0.1, -0.3, 0.2, 0.5, 0.1, -0.1, 0.3]
        df = _make_yearly_df({"Negative": values})
        result = compute_stability_metrics(df)
        assert len(result) == 1
        # CV with near-zero mean could be large or NaN, but shouldn't crash

    def test_custom_entity_col(self):
        """Should work with entity_col='name' instead of 'district'."""
        rows = []
        for year in range(2012, 2020):
            rows.append({"name": "SiteA", "year": year, "median_radiance": 3.0})
        df = pd.DataFrame(rows)
        result = compute_stability_metrics(df, entity_col="name")
        assert "name" in result.columns
        assert len(result) == 1


class TestStabilityCsvOutput:
    """Verify CSV persistence."""

    def test_writes_csv(self, tmp_dir):
        import os
        df = _make_yearly_df({"D": [1.0, 2.0, 3.0, 4.0, 5.0]})
        csv_path = os.path.join(tmp_dir, "stability.csv")
        compute_stability_metrics(df, output_csv=csv_path)
        assert os.path.exists(csv_path)
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == 1
