"""
Tests for Pandera schema validation gates and pipeline error handling.

Verifies that:
- Schemas reject invalid data (missing columns, wrong types, out-of-range)
- validate_schema() returns warnings in lenient mode
- validate_schema() raises in strict mode
- Pipeline runner handles validation failures gracefully
"""

import numpy as np
import pandas as pd
import pytest

from src.schemas import (
    YearlyRadianceSchema,
    TrendsSchema,
    SiteYearlySchema,
    SiteTrendsSchema,
    StabilitySchema,
    validate_schema,
)


# ── YearlyRadianceSchema ────────────────────────────────────────────────


class TestYearlyRadianceSchema:
    """Tests for the yearly radiance DataFrame schema."""

    def _valid_df(self):
        return pd.DataFrame({
            "district": ["DistrictA", "DistrictB"],
            "year": [2020, 2020],
            "mean_radiance": [5.0, 10.0],
            "median_radiance": [4.5, 9.0],
            "pixel_count": [1000, 800],
        })

    def test_valid_data_passes(self):
        df = self._valid_df()
        YearlyRadianceSchema.validate(df)

    def test_missing_required_column_fails(self):
        df = self._valid_df().drop(columns=["median_radiance"])
        with pytest.raises(Exception):
            YearlyRadianceSchema.validate(df)

    def test_negative_radiance_fails(self):
        df = self._valid_df()
        df.loc[0, "mean_radiance"] = -1.0
        with pytest.raises(Exception):
            YearlyRadianceSchema.validate(df)

    def test_radiance_too_high_fails(self):
        df = self._valid_df()
        df.loc[0, "mean_radiance"] = 600.0
        with pytest.raises(Exception):
            YearlyRadianceSchema.validate(df)

    def test_year_too_low_fails(self):
        df = self._valid_df()
        df.loc[0, "year"] = 1990
        with pytest.raises(Exception):
            YearlyRadianceSchema.validate(df)

    def test_zero_pixel_count_fails(self):
        df = self._valid_df()
        df.loc[0, "pixel_count"] = 0
        with pytest.raises(Exception):
            YearlyRadianceSchema.validate(df)

    def test_extra_columns_allowed(self):
        df = self._valid_df()
        df["extra_col"] = "foo"
        YearlyRadianceSchema.validate(df)


# ── TrendsSchema ────────────────────────────────────────────────────────


class TestTrendsSchema:
    """Tests for the trends DataFrame schema."""

    def _valid_df(self):
        return pd.DataFrame({
            "district": ["DistrictA", "DistrictB"],
            "annual_pct_change": [3.5, -1.2],
            "r_squared": [0.95, 0.80],
        })

    def test_valid_data_passes(self):
        TrendsSchema.validate(self._valid_df())

    def test_r_squared_above_one_fails(self):
        df = self._valid_df()
        df.loc[0, "r_squared"] = 1.5
        with pytest.raises(Exception):
            TrendsSchema.validate(df)

    def test_r_squared_negative_fails(self):
        df = self._valid_df()
        df.loc[0, "r_squared"] = -0.1
        with pytest.raises(Exception):
            TrendsSchema.validate(df)

    def test_nan_r_squared_allowed(self):
        """NaN is allowed for r_squared (insufficient data case)."""
        df = self._valid_df()
        df.loc[0, "r_squared"] = np.nan
        TrendsSchema.validate(df)

    def test_extreme_pct_change_fails(self):
        df = self._valid_df()
        df.loc[0, "annual_pct_change"] = 200.0
        with pytest.raises(Exception):
            TrendsSchema.validate(df)


# ── SiteYearlySchema ───────────────────────────────────────────────────


class TestSiteYearlySchema:
    """Tests for the site yearly DataFrame schema."""

    def _valid_df(self):
        return pd.DataFrame({
            "name": ["TestCity", "TestSite"],
            "year": [2020, 2020],
            "median_radiance": [15.0, 0.5],
            "type": ["city", "site"],
        })

    def test_valid_data_passes(self):
        SiteYearlySchema.validate(self._valid_df())

    def test_invalid_type_fails(self):
        df = self._valid_df()
        df.loc[0, "type"] = "village"
        with pytest.raises(Exception):
            SiteYearlySchema.validate(df)


# ── validate_schema() function ──────────────────────────────────────────


class TestValidateSchemaFunction:
    """Tests for the validate_schema() convenience function."""

    def test_none_df_returns_warning(self):
        warnings = validate_schema(None, YearlyRadianceSchema, "test", strict=False)
        assert len(warnings) == 1
        assert "None" in warnings[0]

    def test_none_df_raises_in_strict_mode(self):
        with pytest.raises(ValueError, match="None"):
            validate_schema(None, YearlyRadianceSchema, "test", strict=True)

    def test_empty_df_returns_warning(self):
        df = pd.DataFrame(columns=["district", "year", "mean_radiance",
                                     "median_radiance", "pixel_count"])
        warnings = validate_schema(df, YearlyRadianceSchema, "test", strict=False)
        assert len(warnings) == 1
        assert "empty" in warnings[0].lower()

    def test_empty_df_raises_in_strict_mode(self):
        df = pd.DataFrame(columns=["district", "year", "mean_radiance",
                                     "median_radiance", "pixel_count"])
        with pytest.raises(ValueError, match="empty"):
            validate_schema(df, YearlyRadianceSchema, "test", strict=True)

    def test_invalid_data_returns_warnings_lenient(self):
        df = pd.DataFrame({
            "district": ["A"],
            "year": [2020],
            "mean_radiance": [-5.0],  # Invalid
            "median_radiance": [1.0],
            "pixel_count": [100],
        })
        warnings = validate_schema(df, YearlyRadianceSchema, "test", strict=False)
        assert len(warnings) > 0

    def test_invalid_data_raises_in_strict_mode(self):
        df = pd.DataFrame({
            "district": ["A"],
            "year": [2020],
            "mean_radiance": [-5.0],  # Invalid
            "median_radiance": [1.0],
            "pixel_count": [100],
        })
        with pytest.raises(ValueError):
            validate_schema(df, YearlyRadianceSchema, "test", strict=True)

    def test_valid_data_returns_no_warnings(self):
        df = pd.DataFrame({
            "district": ["A"],
            "year": [2020],
            "mean_radiance": [5.0],
            "median_radiance": [4.0],
            "pixel_count": [100],
        })
        warnings = validate_schema(df, YearlyRadianceSchema, "test", strict=False)
        assert len(warnings) == 0


# ── Pipeline types integration ──────────────────────────────────────────


class TestPipelineTypesDeserialization:
    """Test from_dict() deserialization on StepResult and PipelineRunResult."""

    def test_step_result_roundtrip(self):
        from src.pipeline_types import StepResult

        original = StepResult(
            step_name="test",
            status="success",
            input_summary={"records": 100},
            output_summary={"trends": 36},
            timing_seconds=1.5,
            warnings=["low coverage"],
            nan_summary={"median_radiance": 2},
        )
        d = original.to_dict()
        restored = StepResult.from_dict(d)

        assert restored.step_name == original.step_name
        assert restored.status == original.status
        assert restored.input_summary == original.input_summary
        assert restored.timing_seconds == original.timing_seconds
        assert restored.nan_summary == original.nan_summary

    def test_pipeline_run_result_roundtrip(self):
        from src.pipeline_types import PipelineRunResult, StepResult

        original = PipelineRunResult(
            run_dir="/tmp/test",
            entity_type="district",
            years_processed=[2020, 2021],
            total_time_seconds=10.0,
        )
        original.step_results.append(
            StepResult(step_name="s1", status="success", timing_seconds=5.0)
        )
        original.step_results.append(
            StepResult(step_name="s2", status="error", error="boom")
        )

        d = original.to_dict()
        restored = PipelineRunResult.from_dict(d)

        assert restored.entity_type == "district"
        assert len(restored.step_results) == 2
        assert restored.step_results[0].step_name == "s1"
        assert restored.step_results[1].error == "boom"
        assert not restored.all_ok


# ── NaN tracking ────────────────────────────────────────────────────────


class TestNanTracking:
    """Tests for the NaN propagation tracker in pipeline_runner."""

    def test_track_nan_counts_no_nans(self):
        from src.pipeline_runner import track_nan_counts

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = track_nan_counts(df, "test")
        assert result == {}

    def test_track_nan_counts_with_nans(self):
        from src.pipeline_runner import track_nan_counts

        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, np.nan, 6]})
        result = track_nan_counts(df, "test")
        assert result["a"] == 1
        assert result["b"] == 2

    def test_track_nan_counts_none_df(self):
        from src.pipeline_runner import track_nan_counts

        result = track_nan_counts(None, "test")
        assert result == {}

    def test_nan_propagation_warning(self, caplog):
        """NaN count increase from prev step should produce a warning."""
        from src.pipeline_runner import track_nan_counts
        import logging

        prev = {"a": 1}
        df = pd.DataFrame({"a": [np.nan, np.nan, 3.0]})

        with caplog.at_level(logging.WARNING):
            track_nan_counts(df, "test_step", prev_nan_counts=prev)

        assert any("NaN count increased" in msg for msg in caplog.messages)
