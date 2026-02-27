"""
Integration tests for pipeline stage-to-stage data flow.

Verifies that data flows correctly between pipeline steps:
  year processing → trend fitting → classification → output.

Uses synthetic data (2 districts, 3 years) to run quickly (<30s).
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.formulas.classification import classify_alan
from src.formulas.trend import fit_log_linear_trend
from src.pipeline_types import StepResult


# ── Stage-to-Stage Data Flow ─────────────────────────────────────────


class TestYearlyToTrendFlow:
    """Year-level aggregated data → trend fitting → classification."""

    @pytest.fixture
    def yearly_df(self, trend_dataframe):
        """Use the conftest trend_dataframe (2 districts, 13 years)."""
        return trend_dataframe

    def test_trend_fitting_consumes_yearly_data(self, yearly_df):
        """fit_log_linear_trend should work with yearly DataFrame columns."""
        districts = yearly_df["district"].unique()
        assert len(districts) == 2

        for d in districts:
            d_data = yearly_df[yearly_df["district"] == d].sort_values("year")
            result = fit_log_linear_trend(
                d_data["year"].values,
                d_data["median_radiance"].values,
            )
            assert not np.isnan(result["annual_pct_change"])
            assert result["n_years"] == len(d_data)

    def test_classification_follows_trends(self, yearly_df):
        """ALAN classification from latest year should be consistent."""
        for d in yearly_df["district"].unique():
            d_data = yearly_df[yearly_df["district"] == d].sort_values("year")
            latest_rad = d_data.iloc[-1]["median_radiance"]
            cls = classify_alan(latest_rad)
            assert cls in ("low", "medium", "high")

    def test_no_nan_introduction_in_trend_pipeline(self, yearly_df):
        """Trend fitting should not introduce NaN for valid input data."""
        for d in yearly_df["district"].unique():
            d_data = yearly_df[yearly_df["district"] == d].sort_values("year")
            result = fit_log_linear_trend(
                d_data["year"].values,
                d_data["median_radiance"].values,
            )
            for key in ["annual_pct_change", "ci_low", "ci_high", "r_squared", "p_value"]:
                assert not np.isnan(result[key]), f"NaN in {key} for {d}"


# ── Column Schema Validation ─────────────────────────────────────────


class TestColumnSchemas:
    """Verify expected columns exist at each pipeline stage."""

    YEARLY_REQUIRED_COLS = [
        "district", "year", "mean_radiance", "median_radiance", "pixel_count",
    ]

    TREND_REQUIRED_COLS = [
        "annual_pct_change", "ci_low", "ci_high", "r_squared", "p_value", "n_years",
    ]

    def test_yearly_df_has_required_columns(self, trend_dataframe):
        for col in self.YEARLY_REQUIRED_COLS:
            assert col in trend_dataframe.columns, f"Missing column: {col}"

    def test_yearly_df_types(self, trend_dataframe):
        assert trend_dataframe["year"].dtype in (np.int64, np.int32, int)
        assert np.issubdtype(trend_dataframe["median_radiance"].dtype, np.floating)

    def test_trend_result_has_required_keys(self, trend_dataframe):
        d = trend_dataframe["district"].iloc[0]
        d_data = trend_dataframe[trend_dataframe["district"] == d].sort_values("year")
        result = fit_log_linear_trend(
            d_data["year"].values,
            d_data["median_radiance"].values,
        )
        for key in self.TREND_REQUIRED_COLS:
            assert key in result, f"Missing key: {key}"


# ── StepResult Validation ────────────────────────────────────────────


class TestStepResultIntegration:
    """StepResult dataclass integrates correctly with pipeline steps."""

    def test_success_step_result(self):
        sr = StepResult(
            step_name="test_step",
            status="success",
            input_summary={"records": 100},
            output_summary={"trends": 36},
            timing_seconds=1.5,
        )
        assert sr.ok
        assert sr.to_dict()["status"] == "success"

    def test_error_step_result(self):
        sr = StepResult(
            step_name="test_step",
            status="error",
            error="Something went wrong",
            timing_seconds=0.1,
        )
        assert not sr.ok
        assert sr.to_dict()["error"] == "Something went wrong"

    def test_step_result_to_dict_is_serializable(self):
        """to_dict() should produce JSON-serializable output."""
        import json

        sr = StepResult(
            step_name="test",
            status="success",
            input_summary={"years": [2020, 2021]},
            output_summary={"count": 10},
            timing_seconds=2.3,
            warnings=["low coverage in 2020"],
        )
        # Should not raise
        json_str = json.dumps(sr.to_dict())
        assert "test" in json_str


# ── Data Preservation Checks ─────────────────────────────────────────


class TestDataPreservation:
    """Verify no silent data loss or corruption between stages."""

    def test_all_districts_preserved_through_trend_fitting(self, trend_dataframe):
        """Every district in yearly data should get a trend result."""
        districts = trend_dataframe["district"].unique()
        results = []
        for d in districts:
            d_data = trend_dataframe[trend_dataframe["district"] == d].sort_values("year")
            result = fit_log_linear_trend(
                d_data["year"].values,
                d_data["median_radiance"].values,
            )
            result["district"] = d
            results.append(result)

        trends_df = pd.DataFrame(results)
        assert set(trends_df["district"]) == set(districts)

    def test_year_count_matches_input(self, trend_dataframe):
        """n_years in trend result should match input data."""
        for d in trend_dataframe["district"].unique():
            d_data = trend_dataframe[trend_dataframe["district"] == d]
            result = fit_log_linear_trend(
                d_data["year"].values,
                d_data["median_radiance"].values,
            )
            assert result["n_years"] == len(d_data)

    def test_no_extra_districts_appear(self, trend_dataframe):
        """Trend fitting should not create districts that don't exist."""
        input_districts = set(trend_dataframe["district"].unique())
        results = []
        for d in input_districts:
            d_data = trend_dataframe[trend_dataframe["district"] == d].sort_values("year")
            result = fit_log_linear_trend(
                d_data["year"].values,
                d_data["median_radiance"].values,
            )
            result["district"] = d
            results.append(result)

        output_districts = {r["district"] for r in results}
        assert output_districts == input_districts


# ── Output Directory Structure ───────────────────────────────────────


class TestOutputStructure:
    """Verify entity-based output directory creation."""

    def test_get_entity_dirs_creates_correct_structure(self, tmp_dir):
        from src.config import get_entity_dirs

        dirs = get_entity_dirs(tmp_dir, "district")
        assert dirs["csv"].endswith("district/csv")
        assert dirs["maps"].endswith("district/maps")
        assert dirs["reports"].endswith("district/reports")
        assert dirs["diagnostics"].endswith("district/diagnostics")

    def test_all_entity_types_supported(self, tmp_dir):
        from src.config import get_entity_dirs

        for entity in ("district", "city", "site"):
            dirs = get_entity_dirs(tmp_dir, entity)
            assert entity in dirs["csv"]


# ── Pipeline Types Integration ───────────────────────────────────────


class TestPipelineTypesIntegration:
    """PipelineRunResult tracks multi-step execution."""

    def test_pipeline_run_result_aggregation(self):
        from src.pipeline_types import PipelineRunResult

        run = PipelineRunResult(
            run_dir="/tmp/test_run",
            entity_type="district",
            years_processed=[2020, 2021, 2022],
        )

        # Add step results
        run.step_results.append(StepResult(step_name="step1", status="success", timing_seconds=1.0))
        run.step_results.append(StepResult(step_name="step2", status="success", timing_seconds=2.0))

        assert run.all_ok
        assert len(run.failed_steps) == 0

    def test_pipeline_run_result_detects_failure(self):
        from src.pipeline_types import PipelineRunResult

        run = PipelineRunResult(entity_type="district")
        run.step_results.append(StepResult(step_name="step1", status="success"))
        run.step_results.append(StepResult(step_name="step2", status="error", error="boom"))

        assert not run.all_ok
        assert len(run.failed_steps) == 1
        assert run.failed_steps[0].step_name == "step2"

    def test_pipeline_run_result_to_dict(self):
        import json
        from src.pipeline_types import PipelineRunResult

        run = PipelineRunResult(
            run_dir="/tmp/test",
            entity_type="district",
            years_processed=[2020],
            total_time_seconds=5.0,
        )
        run.step_results.append(StepResult(step_name="s1", status="success", timing_seconds=5.0))

        d = run.to_dict()
        json_str = json.dumps(d)
        assert "district" in json_str

    def test_pipeline_run_result_has_git_sha(self):
        from src.pipeline_types import PipelineRunResult

        run = PipelineRunResult(entity_type="district")
        d = run.to_dict()
        # git_sha may be None if not in a git repo, but the field must exist
        assert "git_sha" in d

    def test_pipeline_run_result_has_started_at(self):
        from src.pipeline_types import PipelineRunResult

        run = PipelineRunResult(entity_type="district")
        d = run.to_dict()
        assert "started_at" in d
        assert d["started_at"] != ""


# ── Logging Infrastructure ──────────────────────────────────────────────


class TestLoggingInfrastructure:
    """Logging configuration and reset behaviour."""

    def test_reset_logging_clears_handlers(self):
        import logging
        from src.logging_config import setup_logging, reset_logging

        reset_logging()
        setup_logging()
        root = logging.getLogger()
        assert len(root.handlers) > 0

        reset_logging()
        assert len(root.handlers) == 0

    def test_run_id_is_set(self):
        from src.logging_config import set_run_id, get_run_id

        rid = set_run_id("test123")
        assert get_run_id() == "test123"

    def test_run_id_auto_generates(self):
        from src.logging_config import reset_logging, get_run_id

        reset_logging()  # clears _run_id
        rid = get_run_id()
        assert len(rid) == 8  # uuid4 short form

    def test_log_level_env_var(self, monkeypatch):
        """LOG_LEVEL env var should control console handler level."""
        import logging
        from src.logging_config import reset_logging, setup_logging

        reset_logging()
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        setup_logging()
        root = logging.getLogger()
        console_handler = [h for h in root.handlers if isinstance(h, logging.StreamHandler)
                          and not isinstance(h, logging.FileHandler)]
        assert len(console_handler) > 0
        assert console_handler[0].level == logging.DEBUG
