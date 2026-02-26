"""
Tests for src/census/legacy_pca.py validation and edge cases.

CRITICAL: Census data provides the population metrics correlated with
ALAN radiance in the paper. Silent data loading failures or wrong
district name resolution corrupt the entire cross-dataset analysis.
"""

import pandas as pd
import pytest

from src.census.legacy_pca import validate, get_meta, _compute_derived_ratios, _prefix_columns


class TestGetMeta:
    """Validate dataset metadata."""

    def test_meta_fields_present(self):
        meta = get_meta()
        assert meta.name == "census_2011_pca"
        assert meta.short_label == "c2011"
        assert 2011 in meta.reference_years
        assert meta.entity_col == "district"

    def test_meta_has_citation(self):
        meta = get_meta()
        assert meta.citation and len(meta.citation) > 20

    def test_meta_has_source_url(self):
        meta = get_meta()
        assert meta.source_url.startswith("http")


class TestValidate:
    """Tests for the validate() function edge cases."""

    def test_none_dataframe(self):
        """Should return a warning, not crash."""
        warnings = validate(None)
        assert len(warnings) == 1
        assert "None" in warnings[0]

    def test_missing_entity_column(self):
        """DataFrame without 'district' column should warn."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        warnings = validate(df)
        assert any("Missing entity column" in w for w in warnings)

    def test_too_few_districts(self):
        """With < 30 districts, should warn (expects ~36 for Maharashtra)."""
        df = pd.DataFrame({
            "district": [f"D{i}" for i in range(10)],
            "c2011_TOT_P": range(10),
        })
        warnings = validate(df)
        assert any("Only 10 districts" in w for w in warnings)

    def test_sufficient_districts_no_warning(self):
        """With >= 30 districts and enough columns, should have no district count warning."""
        districts = [f"District_{i}" for i in range(36)]
        data = {"district": districts}
        for col in ["c2011_TOT_P", "c2011_TOT_M", "c2011_TOT_F",
                     "c2011_P_LIT", "c2011_No_HH", "c2011_P_06"]:
            data[col] = range(36)
        df = pd.DataFrame(data)
        warnings = validate(df)
        assert not any("districts found" in w for w in warnings)

    def test_too_few_metric_columns(self):
        """With < 5 prefixed metric columns, should warn."""
        df = pd.DataFrame({
            "district": [f"D{i}" for i in range(36)],
            "c2011_TOT_P": range(36),
            "c2011_TOT_M": range(36),
        })
        warnings = validate(df)
        assert any("metric columns" in w for w in warnings)

    def test_high_nan_column_warns(self):
        """Columns with > 20% NaN should trigger a warning."""
        import numpy as np
        districts = [f"D{i}" for i in range(36)]
        data = {"district": districts}
        for col in ["c2011_TOT_P", "c2011_TOT_M", "c2011_TOT_F",
                     "c2011_P_LIT", "c2011_No_HH"]:
            data[col] = range(36)
        # Make one column mostly NaN
        data["c2011_P_SC"] = [np.nan] * 30 + list(range(6))
        df = pd.DataFrame(data)
        warnings = validate(df)
        assert any("NaN" in w for w in warnings)


class TestComputeDerivedRatios:
    """Tests for derived ratio computation."""

    def test_literacy_rate(self):
        """literacy_rate = P_LIT / TOT_P."""
        df = pd.DataFrame({
            "district": ["A"],
            "P_LIT": [800],
            "TOT_P": [1000],
        })
        result = _compute_derived_ratios(df)
        assert "literacy_rate" in result.columns
        assert result["literacy_rate"].iloc[0] == pytest.approx(0.8)

    def test_compound_ratio_sc_st_share(self):
        """sc_st_share = (P_SC + P_ST) / TOT_P."""
        df = pd.DataFrame({
            "district": ["A"],
            "P_SC": [100],
            "P_ST": [50],
            "TOT_P": [1000],
        })
        result = _compute_derived_ratios(df)
        assert "sc_st_share" in result.columns
        assert result["sc_st_share"].iloc[0] == pytest.approx(0.15)

    def test_zero_denominator(self):
        """Division by zero should produce NaN, not crash."""
        df = pd.DataFrame({
            "district": ["A"],
            "P_LIT": [100],
            "TOT_P": [0],
        })
        result = _compute_derived_ratios(df)
        import numpy as np
        assert pd.isna(result["literacy_rate"].iloc[0])

    def test_missing_columns_skipped(self):
        """Ratios requiring missing columns should be silently skipped."""
        df = pd.DataFrame({
            "district": ["A"],
            "TOT_P": [1000],
            # No P_LIT column
        })
        result = _compute_derived_ratios(df)
        # literacy_rate should not be added since P_LIT is missing
        assert "literacy_rate" not in result.columns


class TestPrefixColumns:
    """Tests for column prefixing."""

    def test_prefixes_metric_columns(self):
        df = pd.DataFrame({
            "district": ["A"],
            "TOT_P": [1000],
            "P_LIT": [800],
        })
        result = _prefix_columns(df, "c2011", "district")
        assert "c2011_TOT_P" in result.columns
        assert "c2011_P_LIT" in result.columns

    def test_entity_col_not_prefixed(self):
        df = pd.DataFrame({"district": ["A"], "TOT_P": [1000]})
        result = _prefix_columns(df, "c2011", "district")
        assert "district" in result.columns
        assert "c2011_district" not in result.columns
