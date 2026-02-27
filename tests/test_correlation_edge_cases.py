"""
Tests for src/formulas/correlation.py edge cases and failure modes.

CRITICAL: These functions produce the correlation statistics cited in the
research paper. Silent NaN propagation or wrong p-values undermine all
cross-dataset findings (e.g. "radiance correlates with population").
"""

import numpy as np
import pandas as pd
import pytest

from src.formulas.correlation import (
    _clean_pair,
    compute_correlation_matrix,
    ols_regression,
    partial_correlation,
    pearson_correlation,
    spearman_correlation,
)


class TestPearsonEdgeCases:
    """Edge cases for Pearson correlation."""

    def test_perfect_positive_correlation(self):
        """r should be 1.0 for identical arrays."""
        x = [1, 2, 3, 4, 5]
        result = pearson_correlation(x, x)
        assert result["r"] == pytest.approx(1.0, abs=1e-10)
        assert result["n"] == 5

    def test_perfect_negative_correlation(self):
        """r should be -1.0 for perfectly inversely related arrays."""
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        result = pearson_correlation(x, y)
        assert result["r"] == pytest.approx(-1.0, abs=1e-10)

    def test_zero_correlation(self):
        """Orthogonal data should have r ≈ 0."""
        x = [1, -1, 1, -1, 1, -1]
        y = [1, 1, -1, -1, 1, 1]
        result = pearson_correlation(x, y)
        assert abs(result["r"]) < 0.3  # Not strictly 0 due to small n

    def test_fewer_than_three_points(self):
        """With n<3, should return NaN (can't compute meaningful correlation)."""
        result = pearson_correlation([1, 2], [3, 4])
        assert np.isnan(result["r"])
        assert result["n"] == 2

    def test_single_point(self):
        result = pearson_correlation([1], [2])
        assert np.isnan(result["r"])

    def test_empty_arrays(self):
        result = pearson_correlation([], [])
        assert np.isnan(result["r"])
        assert result["n"] == 0

    def test_nan_pairwise_dropping(self):
        """NaNs should be dropped pairwise, not listwise."""
        x = [1, 2, np.nan, 4, 5]
        y = [2, 4, 6, 8, 10]
        result = pearson_correlation(x, y)
        assert result["n"] == 4  # one pair dropped
        assert result["r"] == pytest.approx(1.0, abs=1e-10)

    def test_all_nan(self):
        """All-NaN arrays should return NaN."""
        result = pearson_correlation([np.nan, np.nan], [np.nan, np.nan])
        assert np.isnan(result["r"])

    def test_constant_array(self):
        """If one array is constant, correlation is undefined (0 variance).
        scipy.pearsonr returns NaN with a warning."""
        x = [5, 5, 5, 5, 5]
        y = [1, 2, 3, 4, 5]
        result = pearson_correlation(x, y)
        # r should be NaN (can't divide by zero std)
        assert np.isnan(result["r"]) or result["r"] == 0.0

    def test_ci_contains_r(self):
        """Confidence interval should contain the point estimate."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        y = 0.7 * x + rng.normal(0, 0.5, 50)
        result = pearson_correlation(x, y)
        assert result["ci_low"] <= result["r"] <= result["ci_high"]

    def test_ci_narrows_with_more_data(self):
        """CI width should decrease as n increases."""
        rng = np.random.default_rng(42)
        x_small = rng.normal(0, 1, 10)
        y_small = 0.5 * x_small + rng.normal(0, 1, 10)
        r_small = pearson_correlation(x_small, y_small)

        x_large = rng.normal(0, 1, 500)
        y_large = 0.5 * x_large + rng.normal(0, 1, 500)
        r_large = pearson_correlation(x_large, y_large)

        width_small = r_small["ci_high"] - r_small["ci_low"]
        width_large = r_large["ci_high"] - r_large["ci_low"]
        assert width_large < width_small

    def test_p_value_significant_for_strong_correlation(self):
        """Strong linear relationship with many points should have p < 0.05."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = 0.9 * x + rng.normal(0, 0.1, 100)
        result = pearson_correlation(x, y)
        assert result["p_value"] < 0.05


class TestSpearmanEdgeCases:
    """Edge cases for Spearman rank correlation."""

    def test_perfect_monotone(self):
        """Monotone increasing data should have rho=1."""
        x = [1, 2, 3, 4, 5]
        y = [10, 20, 30, 40, 50]
        result = spearman_correlation(x, y)
        assert result["rho"] == pytest.approx(1.0, abs=1e-10)

    def test_nonlinear_monotone(self):
        """Exponential relationship is monotone: rho should be near 1."""
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        y = [np.exp(v) for v in x]
        result = spearman_correlation(x, y)
        assert result["rho"] > 0.95

    def test_fewer_than_three_points(self):
        result = spearman_correlation([1, 2], [3, 4])
        assert np.isnan(result["rho"])

    def test_nan_handling(self):
        x = [1, np.nan, 3, 4, 5]
        y = [2, 4, np.nan, 8, 10]
        result = spearman_correlation(x, y)
        assert result["n"] == 3  # two pairs dropped


class TestPartialCorrelation:
    """Tests for partial correlation controlling for covariates."""

    def test_basic_partial(self):
        """Partial correlation between x and y controlling for z."""
        rng = np.random.default_rng(42)
        z = rng.normal(0, 1, 100)
        x = z + rng.normal(0, 0.5, 100)
        y = z + rng.normal(0, 0.5, 100)
        result = partial_correlation(x, y, z)
        # After removing z, residual correlation should be lower than raw
        raw = pearson_correlation(x, y)
        # Partial r might be near 0 since shared variance is from z
        assert abs(result["r"]) < abs(raw["r"]) + 0.1

    def test_insufficient_data(self):
        """With fewer points than covariates + 3, should return NaN."""
        x = [1, 2, 3]
        y = [4, 5, 6]
        cov = [[1, 2], [3, 4], [5, 6]]  # 2 covariates, need n >= 5
        result = partial_correlation(x, y, cov)
        assert np.isnan(result["r"])

    def test_single_covariate_1d(self):
        """Should handle a 1-D covariate array."""
        rng = np.random.default_rng(42)
        z = rng.normal(0, 1, 50)
        x = z + rng.normal(0, 1, 50)
        y = z + rng.normal(0, 1, 50)
        result = partial_correlation(x, y, z)
        assert not np.isnan(result["r"])
        assert result["n"] == 50

    def test_dataframe_covariates(self):
        """Should accept a DataFrame as covariates."""
        rng = np.random.default_rng(42)
        z1 = rng.normal(0, 1, 50)
        z2 = rng.normal(0, 1, 50)
        x = z1 + rng.normal(0, 0.5, 50)
        y = z2 + rng.normal(0, 0.5, 50)
        cov_df = pd.DataFrame({"z1": z1, "z2": z2})
        result = partial_correlation(x, y, cov_df)
        assert not np.isnan(result["r"])

    def test_nan_in_covariates_dropped(self):
        """Rows with NaN in covariates should be excluded."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        z = [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10]
        result = partial_correlation(x, y, z)
        assert result["n"] == 9


class TestOlsRegression:
    """Tests for the OLS regression function."""

    def test_perfect_linear_fit(self):
        """y = 2x + 3 should have R²=1.0 and correct coefficients."""
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = 2 * x + 3
        result = ols_regression(y, x)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-10)
        assert result["coefficients"]["intercept"] == pytest.approx(3.0, abs=1e-6)
        assert result["coefficients"]["x0"] == pytest.approx(2.0, abs=1e-6)

    def test_no_relationship(self):
        """Random x and y should have low R²."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        result = ols_regression(y, x)
        assert result["r_squared"] < 0.2

    def test_insufficient_data(self):
        """With n < k+2, should return empty coefficients and NaN R²."""
        result = ols_regression([1], [2])
        assert np.isnan(result["r_squared"])
        assert result["coefficients"] == {}

    def test_multi_feature(self):
        """Multiple features: y = 2*x1 + 3*x2 + 1."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 2))
        y = 2 * X[:, 0] + 3 * X[:, 1] + 1
        result = ols_regression(y, X, feature_names=["x1", "x2"])
        assert result["r_squared"] > 0.99
        assert result["coefficients"]["x1"] == pytest.approx(2.0, abs=0.1)
        assert result["coefficients"]["x2"] == pytest.approx(3.0, abs=0.1)

    def test_residuals_sum_to_zero(self):
        """OLS residuals should sum to approximately zero (OLS property)."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        y = 3 * x + rng.normal(0, 1, 50)
        result = ols_regression(y, x)
        assert abs(np.sum(result["residuals"])) < 1e-8

    def test_feature_names_from_dataframe(self):
        """When X is a DataFrame, feature names should be inferred from columns."""
        df = pd.DataFrame({"pop": [1, 2, 3, 4, 5], "area": [10, 20, 30, 40, 50]})
        y = [2, 4, 6, 8, 10]
        result = ols_regression(y, df)
        assert "pop" in result["feature_names"]
        assert "area" in result["feature_names"]

    def test_nan_dropped(self):
        """Rows with NaN should be dropped before fitting."""
        x = [1, 2, np.nan, 4, 5, 6, 7, 8]
        y = [2, 4, 6, 8, 10, 12, 14, 16]
        result = ols_regression(y, x)
        # Should fit on 7 points
        assert len(result["residuals"]) == 7


class TestCorrelationMatrix:
    """Tests for compute_correlation_matrix."""

    def test_basic_matrix(self):
        """Should compute pairwise correlations between column sets."""
        df = pd.DataFrame({
            "radiance": [1, 2, 3, 4, 5],
            "growth": [2, 4, 6, 8, 10],
            "population": [10, 20, 30, 40, 50],
        })
        result = compute_correlation_matrix(df, ["radiance"], ["population", "growth"])
        assert len(result) == 2  # 1 x_col × 2 y_cols

    def test_both_methods(self):
        """method='both' should produce pearson and spearman columns."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
        result = compute_correlation_matrix(df, ["x"], ["y"], method="both")
        assert "pearson_r" in result.columns
        assert "spearman_r" in result.columns

    def test_pearson_only(self):
        """method='pearson' should only produce pearson columns."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
        result = compute_correlation_matrix(df, ["x"], ["y"], method="pearson")
        assert "pearson_r" in result.columns
        assert "spearman_r" not in result.columns

    def test_spearman_only(self):
        """method='spearman' should only produce spearman columns."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
        result = compute_correlation_matrix(df, ["x"], ["y"], method="spearman")
        assert "spearman_r" in result.columns
        assert "pearson_r" not in result.columns

    def test_missing_column_skipped(self):
        """Columns not in the DataFrame should be silently skipped."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = compute_correlation_matrix(df, ["x", "nonexistent"], ["y"])
        assert len(result) == 1  # only x vs y

    def test_symmetric_results(self):
        """cor(x, y) should equal cor(y, x) in terms of r value."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [5, 3, 4, 2, 1]})
        r1 = compute_correlation_matrix(df, ["x"], ["y"], method="pearson")
        r2 = compute_correlation_matrix(df, ["y"], ["x"], method="pearson")
        assert r1.iloc[0]["pearson_r"] == pytest.approx(r2.iloc[0]["pearson_r"])


class TestCleanPair:
    """Tests for the internal _clean_pair helper."""

    def test_no_nans(self):
        x, y = _clean_pair([1, 2, 3], [4, 5, 6])
        assert len(x) == 3

    def test_pairwise_nan_drop(self):
        x, y = _clean_pair([1, np.nan, 3], [4, 5, np.nan])
        assert len(x) == 1  # only index 0 survives
        assert x[0] == 1.0
        assert y[0] == 4.0

    def test_preserves_order(self):
        x, y = _clean_pair([10, np.nan, 30, 40], [1, 2, 3, 4])
        np.testing.assert_array_equal(x, [10, 30, 40])
        np.testing.assert_array_equal(y, [1, 3, 4])
