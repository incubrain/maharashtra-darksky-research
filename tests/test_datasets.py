"""
Tests for the multi-dataset comparison system.

Covers:
- _base.py dataclasses
- _name_resolver.py district name normalization and fuzzy matching
- census_2011_pca.py load, process, validate
- correlation.py pure correlation functions
- dataset_aggregator.py merge logic
"""

import numpy as np
import pandas as pd
import pytest


# ── DatasetMeta / DatasetResult ────────────────────────────────────────


class TestDatasetBase:
    """Tests for _base.py dataclasses."""

    def test_dataset_meta_defaults(self):
        from src.datasets._base import DatasetMeta

        meta = DatasetMeta(
            name="test",
            short_label="t",
            description="Test dataset",
            temporal_type="snapshot",
        )
        assert meta.name == "test"
        assert meta.short_label == "t"
        assert meta.entity_col == "district"
        assert meta.reference_years == []

    def test_dataset_result_ok(self):
        from src.datasets._base import DatasetResult

        result = DatasetResult(dataset_name="test", status="success", rows=10)
        assert result.ok is True
        assert result.rows == 10

    def test_dataset_result_error(self):
        from src.datasets._base import DatasetResult

        result = DatasetResult(
            dataset_name="test", status="error", error="Something failed"
        )
        assert result.ok is False
        assert result.error == "Something failed"


# ── Name Resolver ──────────────────────────────────────────────────────


class TestNameResolver:
    """Tests for _name_resolver.py."""

    def test_normalize_name(self):
        from src.census.name_resolver import normalize_name

        assert normalize_name("  Mumbai  ") == "mumbai"
        assert normalize_name("PUNE") == "pune"
        assert normalize_name("Navi  Mumbai") == "navi mumbai"
        assert normalize_name("") == ""
        assert normalize_name(None) == ""

    def test_exact_match(self):
        from src.census.name_resolver import resolve_names

        vnl = ["Mumbai", "Pune", "Nagpur"]
        dataset = ["Mumbai", "Pune", "Nagpur"]
        mapping, unmatched = resolve_names(vnl, dataset)

        assert len(mapping) == 3
        assert len(unmatched) == 0
        assert mapping["Mumbai"] == "Mumbai"

    def test_case_insensitive_match(self):
        from src.census.name_resolver import resolve_names

        vnl = ["Mumbai", "Pune"]
        dataset = ["MUMBAI", "pune"]
        mapping, unmatched = resolve_names(vnl, dataset)

        assert len(mapping) == 2
        assert mapping["MUMBAI"] == "Mumbai"
        assert mapping["pune"] == "Pune"

    def test_override_match(self):
        from src.census.name_resolver import resolve_names

        vnl = ["Bid", "Gondiya", "Raigarh"]
        dataset = ["Beed", "Gondia", "Raigad"]
        mapping, unmatched = resolve_names(vnl, dataset)

        assert mapping.get("Beed") == "Bid"
        assert mapping.get("Gondia") == "Gondiya"
        assert mapping.get("Raigad") == "Raigarh"
        assert len(unmatched) == 0

    def test_fuzzy_match(self):
        from src.census.name_resolver import resolve_names

        vnl = ["Ahmadnagar", "Aurangabad"]
        dataset = ["Ahmednagar", "Aurangabad"]
        mapping, unmatched = resolve_names(vnl, dataset, fuzzy_threshold=0.7)

        assert mapping.get("Ahmednagar") == "Ahmadnagar"
        assert len(unmatched) == 0

    def test_unmatched_names(self):
        from src.census.name_resolver import resolve_names

        vnl = ["Mumbai", "Pune"]
        dataset = ["Mumbai", "Zzzzzzz"]
        mapping, unmatched = resolve_names(vnl, dataset)

        assert mapping.get("Mumbai") == "Mumbai"
        assert "Zzzzzzz" in unmatched


# ── Census 2011 PCA Module ──────────────────────────────────────────────


class TestCensus2011PCA:
    """Tests for census_2011_pca.py."""

    def test_get_meta(self):
        from src.census.legacy_pca import get_meta

        meta = get_meta()
        assert meta.name == "census_2011_pca"
        assert meta.short_label == "c2011"
        assert meta.temporal_type == "snapshot"
        assert 2011 in meta.reference_years

    def test_load_missing_dir(self, tmp_path):
        from src.census.legacy_pca import load_and_process

        result, df = load_and_process(str(tmp_path / "nonexistent"))
        assert not result.ok
        assert df is None

    def test_load_from_csv(self, tmp_path):
        """Test loading from a pre-processed CSV."""
        from src.census.legacy_pca import load_and_process

        # Create a simple census CSV
        data = {
            "district": ["Mumbai", "Pune", "Nagpur"],
            "TOT_P": [12400000, 9400000, 4650000],
            "No_HH": [3100000, 2350000, 1160000],
            "P_LIT": [10800000, 8200000, 4000000],
            "TOT_WORK_P": [5000000, 3800000, 1900000],
            "MAIN_OT_P": [4200000, 3000000, 1400000],
            "NON_WORK_P": [7400000, 5600000, 2750000],
            "P_06": [1200000, 900000, 450000],
            "P_SC": [1500000, 1100000, 600000],
            "P_ST": [200000, 300000, 500000],
            "P_ILL": [1600000, 1200000, 650000],
            "MAINWORK_P": [4500000, 3400000, 1700000],
            "MARGWORK_P": [500000, 400000, 200000],
            "MAIN_CL_P": [100000, 200000, 150000],
            "MAIN_AL_P": [200000, 250000, 200000],
            "MAIN_HH_P": [500000, 350000, 150000],
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "census_2011_pca.csv"
        df.to_csv(csv_path, index=False)

        result, out_df = load_and_process(str(tmp_path))
        assert result.ok
        assert out_df is not None
        assert len(out_df) == 3
        assert "c2011_TOT_P" in out_df.columns
        assert "c2011_literacy_rate" in out_df.columns

    def test_derived_ratios(self, tmp_path):
        """Test that derived ratios are computed correctly."""
        from src.census.legacy_pca import load_and_process

        data = {
            "district": ["TestDistrict"],
            "TOT_P": [1000],
            "No_HH": [250],
            "P_LIT": [800],
            "P_ILL": [200],
            "TOT_WORK_P": [500],
            "MAIN_OT_P": [300],
            "NON_WORK_P": [500],
            "P_06": [100],
            "P_SC": [150],
            "P_ST": [50],
            "MAINWORK_P": [450],
            "MARGWORK_P": [50],
            "MAIN_CL_P": [80],
            "MAIN_AL_P": [70],
            "MAIN_HH_P": [50],
        }
        pd.DataFrame(data).to_csv(tmp_path / "census_2011_pca.csv", index=False)

        result, df = load_and_process(str(tmp_path))
        assert result.ok

        # literacy_rate = P_LIT / TOT_P = 800/1000 = 0.8
        assert df["c2011_literacy_rate"].iloc[0] == pytest.approx(0.8)
        # household_size = TOT_P / No_HH = 1000/250 = 4.0
        assert df["c2011_household_size"].iloc[0] == pytest.approx(4.0)
        # workforce_rate = TOT_WORK_P / TOT_P = 500/1000 = 0.5
        assert df["c2011_workforce_rate"].iloc[0] == pytest.approx(0.5)

    def test_validate_good_data(self, tmp_path):
        from src.census.legacy_pca import load_and_process, validate

        # Create 36 districts
        districts = [f"District{i}" for i in range(36)]
        data = {
            "district": districts,
            "TOT_P": [1000000 + i * 100000 for i in range(36)],
            "No_HH": [250000 + i * 25000 for i in range(36)],
            "P_LIT": [800000 + i * 80000 for i in range(36)],
            "TOT_WORK_P": [500000 + i * 50000 for i in range(36)],
            "MAIN_OT_P": [300000 + i * 30000 for i in range(36)],
            "NON_WORK_P": [500000 + i * 50000 for i in range(36)],
            "P_06": [100000 + i * 10000 for i in range(36)],
            "P_SC": [100000 + i * 10000 for i in range(36)],
            "P_ST": [50000 + i * 5000 for i in range(36)],
            "P_ILL": [200000 + i * 20000 for i in range(36)],
            "MAINWORK_P": [450000 + i * 45000 for i in range(36)],
            "MARGWORK_P": [50000 + i * 5000 for i in range(36)],
            "MAIN_CL_P": [80000 + i * 8000 for i in range(36)],
            "MAIN_AL_P": [70000 + i * 7000 for i in range(36)],
            "MAIN_HH_P": [50000 + i * 5000 for i in range(36)],
        }
        pd.DataFrame(data).to_csv(tmp_path / "census_2011_pca.csv", index=False)

        result, df = load_and_process(str(tmp_path))
        warnings = validate(df)
        assert len(warnings) == 0


# ── Correlation Functions ───────────────────────────────────────────────


class TestCorrelation:
    """Tests for formulas/correlation.py."""

    def test_pearson_perfect_positive(self):
        from src.formulas.correlation import pearson_correlation

        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([2, 4, 6, 8, 10], dtype=float)
        result = pearson_correlation(x, y)

        assert result["r"] == pytest.approx(1.0)
        assert result["p_value"] < 0.01
        assert result["n"] == 5

    def test_pearson_perfect_negative(self):
        from src.formulas.correlation import pearson_correlation

        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([10, 8, 6, 4, 2], dtype=float)
        result = pearson_correlation(x, y)

        assert result["r"] == pytest.approx(-1.0)

    def test_pearson_handles_nan(self):
        from src.formulas.correlation import pearson_correlation

        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        result = pearson_correlation(x, y)

        assert result["n"] == 4
        assert not np.isnan(result["r"])

    def test_pearson_insufficient_data(self):
        from src.formulas.correlation import pearson_correlation

        x = np.array([1.0, 2.0])
        y = np.array([2.0, 4.0])
        result = pearson_correlation(x, y)

        assert np.isnan(result["r"])

    def test_spearman_monotonic(self):
        from src.formulas.correlation import spearman_correlation

        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([1, 4, 9, 16, 25], dtype=float)  # y = x^2, monotonic
        result = spearman_correlation(x, y)

        assert result["rho"] == pytest.approx(1.0)

    def test_partial_correlation(self):
        from src.formulas.correlation import partial_correlation

        np.random.seed(42)
        n = 50
        z = np.random.randn(n)  # confounder
        x = z + np.random.randn(n) * 0.3
        y = z + np.random.randn(n) * 0.3

        # x and y correlate, but controlling for z should reduce correlation
        result_raw = partial_correlation(x, y, np.zeros((n, 0)).reshape(n, -1) if False else np.ones((n, 1)))
        result_controlled = partial_correlation(x, y, z)

        # After controlling for z, correlation should be weaker
        assert abs(result_controlled["r"]) < 0.8

    def test_compute_correlation_matrix(self):
        from src.formulas.correlation import compute_correlation_matrix

        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [5, 4, 3, 2, 1],
        })
        result = compute_correlation_matrix(df, ["a"], ["b", "c"])

        assert len(result) == 2
        assert "pearson_r" in result.columns
        assert "spearman_r" in result.columns

        # a vs b should be +1.0
        ab = result[result["y_col"] == "b"].iloc[0]
        assert ab["pearson_r"] == pytest.approx(1.0)

        # a vs c should be -1.0
        ac = result[result["y_col"] == "c"].iloc[0]
        assert ac["pearson_r"] == pytest.approx(-1.0)

    def test_ols_regression(self):
        from src.formulas.correlation import ols_regression

        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        y = 3 * x + 1 + np.random.randn(n) * 0.1

        result = ols_regression(y, x, feature_names=["x"])
        assert result["r_squared"] > 0.95
        assert result["coefficients"]["x"] == pytest.approx(3.0, abs=0.5)
        assert result["coefficients"]["intercept"] == pytest.approx(1.0, abs=0.5)


# ── Dataset Aggregator ──────────────────────────────────────────────────


class TestDatasetAggregator:
    """Tests for dataset_aggregator.py."""

    def test_get_dataset_suffix_empty(self):
        from src.dataset_aggregator import get_dataset_suffix

        assert get_dataset_suffix([]) == ""

    def test_get_dataset_suffix_single(self):
        from src.dataset_aggregator import get_dataset_suffix

        result = get_dataset_suffix(["census_2011_pca"])
        assert result == "_x_c2011"

    def test_merge_yearly_snapshot(self):
        from src.dataset_aggregator import merge_yearly_with_datasets

        yearly_df = pd.DataFrame({
            "district": ["A", "A", "B", "B"],
            "year": [2020, 2021, 2020, 2021],
            "median_radiance": [5.0, 5.5, 2.0, 2.1],
        })
        datasets = {
            "census_2011_pca": pd.DataFrame({
                "district": ["A", "B"],
                "c2011_population": [1000000, 500000],
            }),
        }

        merged = merge_yearly_with_datasets(yearly_df, datasets)

        assert len(merged) == 4  # Same as yearly
        assert "c2011_population" in merged.columns
        # A's population should be broadcast to both years
        assert merged[merged["district"] == "A"]["c2011_population"].nunique() == 1

    def test_merge_trends(self):
        from src.dataset_aggregator import merge_trends_with_datasets

        trends_df = pd.DataFrame({
            "district": ["A", "B"],
            "annual_pct_change": [5.0, -1.0],
        })
        datasets = {
            "census_2011_pca": pd.DataFrame({
                "district": ["A", "B"],
                "c2011_literacy_rate": [0.8, 0.6],
            }),
        }

        merged = merge_trends_with_datasets(trends_df, datasets)
        assert len(merged) == 2
        assert "c2011_literacy_rate" in merged.columns

    def test_get_enabled_datasets_from_cli(self):
        from src.dataset_aggregator import get_enabled_datasets
        from types import SimpleNamespace

        args = SimpleNamespace(datasets="census_2011_pca")
        result = get_enabled_datasets(args)
        assert result == ["census_2011_pca"]

    def test_get_enabled_datasets_all(self):
        from src.dataset_aggregator import get_enabled_datasets
        from types import SimpleNamespace

        args = SimpleNamespace(datasets="all")
        result = get_enabled_datasets(args)
        assert "census_2011_pca" in result

    def test_get_enabled_datasets_none(self):
        from src.dataset_aggregator import get_enabled_datasets
        from types import SimpleNamespace

        args = SimpleNamespace(datasets=None)
        result = get_enabled_datasets(args)
        # Default config has enabled=False
        assert result == []
