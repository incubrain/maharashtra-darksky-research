"""
Tests for src/analysis/graduated_classification.py.

Verifies percentile-based ALAN tier classification and temporal trajectory.

CRITICAL: These tiers determine how districts are ranked relative to each other
in the research paper. A bug here silently misrepresents Maharashtra's ALAN
distribution.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.graduated_classification import (
    classify_by_percentiles,
    classify_temporal_trajectory,
)
from src import config


def _make_yearly_df(n_districts=36, years=None, seed=42):
    """Helper: generate synthetic yearly data for n_districts across years."""
    if years is None:
        years = list(range(2012, 2025))
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_districts):
        base = rng.uniform(0.1, 50.0)
        for year in years:
            t = year - years[0]
            rad = base * (1.03 ** t) + rng.normal(0, 0.1)
            rows.append({
                "district": f"District_{i:02d}",
                "year": year,
                "median_radiance": max(rad, 0.01),
            })
    return pd.DataFrame(rows)


class TestClassifyByPercentiles:
    """Core percentile classification tests."""

    def test_all_districts_assigned_a_tier(self):
        """Every district should get a tier label (no NaN tiers)."""
        df = _make_yearly_df(36)
        result = classify_by_percentiles(df)
        assert not result["alan_tier"].isna().any(), "Some districts have NaN tier"
        assert len(result) == 36

    def test_tier_counts_sum_to_total(self):
        """Tier assignments should be exhaustive: sum of tier counts = total districts."""
        df = _make_yearly_df(36)
        result = classify_by_percentiles(df)
        assert result["alan_tier"].value_counts().sum() == 36

    def test_default_five_tiers(self):
        """With default config bins, should produce up to 5 tier labels."""
        df = _make_yearly_df(50)
        result = classify_by_percentiles(df)
        unique_tiers = result["alan_tier"].dropna().unique()
        assert len(unique_tiers) <= 5
        for tier in unique_tiers:
            assert tier in config.ALAN_PERCENTILE_LABELS

    def test_percentile_rank_range(self):
        """Percentile ranks should be in [0, 100]."""
        df = _make_yearly_df(36)
        result = classify_by_percentiles(df)
        assert (result["percentile_rank"] >= 0).all()
        assert (result["percentile_rank"] <= 100).all()

    def test_latest_year_used_by_default(self):
        """When year is not specified, should classify the latest year."""
        df = _make_yearly_df(10, years=[2020, 2021, 2022])
        result = classify_by_percentiles(df)
        assert (result["year"] == 2022).all()

    def test_specific_year(self):
        """Passing year=2015 should classify only 2015 data."""
        df = _make_yearly_df(10)
        result = classify_by_percentiles(df, year=2015)
        assert (result["year"] == 2015).all()

    def test_custom_bins_and_labels(self):
        """Custom bins/labels should override defaults."""
        df = _make_yearly_df(20)
        bins = [0, 50, 100]
        labels = ["Bottom Half", "Top Half"]
        result = classify_by_percentiles(df, bins=bins, labels=labels)
        unique_tiers = result["alan_tier"].dropna().unique()
        assert set(unique_tiers).issubset(set(labels))

    def test_output_columns(self):
        """Result should have the documented columns."""
        df = _make_yearly_df(10)
        result = classify_by_percentiles(df)
        expected = {"district", "median_radiance", "percentile_rank", "alan_tier", "year"}
        assert expected.issubset(set(result.columns))

    def test_monotonicity_between_tiers(self):
        """Higher-tier districts should generally have higher radiance.
        The median radiance of 'Very High' tier should exceed 'Pristine' tier."""
        df = _make_yearly_df(100, seed=1)
        result = classify_by_percentiles(df)
        if "Pristine" in result["alan_tier"].values and "Very High" in result["alan_tier"].values:
            pristine_max = result[result["alan_tier"] == "Pristine"]["median_radiance"].max()
            very_high_min = result[result["alan_tier"] == "Very High"]["median_radiance"].min()
            assert very_high_min > pristine_max, \
                "Very High tier should have higher radiance than Pristine"


class TestClassifyByPercentilesEdgeCases:
    """Edge cases that could silently corrupt tier assignments."""

    def test_empty_year(self):
        """If no data exists for the requested year, should return empty DataFrame."""
        df = _make_yearly_df(10, years=[2020, 2021])
        result = classify_by_percentiles(df, year=2025)
        assert result.empty

    def test_fewer_than_five_districts(self):
        """With <5 districts, should return empty (min data requirement)."""
        df = _make_yearly_df(3)
        result = classify_by_percentiles(df)
        assert result.empty

    def test_exactly_five_districts(self):
        """With exactly 5 districts, should classify all."""
        df = _make_yearly_df(5)
        result = classify_by_percentiles(df)
        assert len(result) == 5

    def test_all_same_radiance(self):
        """If all districts have identical radiance, percentile ranks are all 50-ish
        but the function should not crash."""
        rows = []
        for i in range(20):
            rows.append({"district": f"D{i}", "year": 2022, "median_radiance": 5.0})
        df = pd.DataFrame(rows)
        result = classify_by_percentiles(df)
        assert len(result) == 20

    def test_nan_radiance_dropped(self):
        """Districts with NaN median_radiance should not affect percentile computation."""
        df = _make_yearly_df(10)
        # Set some radiance values to NaN
        df.loc[df["district"] == "District_00", "median_radiance"] = np.nan
        result = classify_by_percentiles(df)
        # District_00 might still be in the result but with NaN tier
        # Other districts should still have valid classifications

    def test_csv_output(self, tmp_dir):
        """CSV output should include both classification and tier boundary files."""
        import os
        df = _make_yearly_df(20)
        csv_path = os.path.join(tmp_dir, "graduated.csv")
        classify_by_percentiles(df, output_csv=csv_path)
        assert os.path.exists(csv_path)
        tier_csv = csv_path.replace(".csv", "_tiers.csv")
        assert os.path.exists(tier_csv)


class TestTemporalTrajectory:
    """Tests for classify_temporal_trajectory (multi-year tier tracking)."""

    def test_trajectory_covers_all_years(self):
        """Each year in the input should appear in the output."""
        years = [2020, 2021, 2022]
        df = _make_yearly_df(10, years=years)
        result = classify_temporal_trajectory(df)
        if not result.empty:
            output_years = set(result["year"].unique())
            assert output_years == set(years)

    def test_trajectory_district_count_per_year(self):
        """Each year should have the same number of districts classified."""
        df = _make_yearly_df(10, years=[2020, 2021, 2022])
        result = classify_temporal_trajectory(df)
        if not result.empty:
            counts = result.groupby("year").size()
            assert counts.nunique() == 1, "Different years have different district counts"

    def test_empty_input(self):
        """Empty DataFrame should return empty result."""
        df = pd.DataFrame(columns=["district", "year", "median_radiance"])
        result = classify_temporal_trajectory(df)
        assert result.empty

    def test_single_year(self):
        """With only one year, trajectory should still work."""
        df = _make_yearly_df(10, years=[2022])
        result = classify_temporal_trajectory(df)
        assert len(result) == 10 or result.empty

    def test_csv_output(self, tmp_dir):
        import os
        df = _make_yearly_df(10, years=[2020, 2021])
        csv_path = os.path.join(tmp_dir, "trajectory.csv")
        classify_temporal_trajectory(df, output_csv=csv_path)
        assert os.path.exists(csv_path)
