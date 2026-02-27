"""
Tests for src/step_runner.py.

Verifies the generic step execution framework: timing, error handling,
StepResult construction, and the expected_exceptions pattern.

CRITICAL: Every pipeline step flows through run_step(). A bug here
silently swallows errors or misreports step status, making pipeline
failures invisible.
"""

import time

import pandas as pd
import pytest

from src.step_runner import run_step
from src.pipeline_types import StepResult


class TestRunStepSuccess:
    """Tests for successful step execution."""

    def test_basic_success(self):
        """A simple function should produce a success StepResult."""
        def simple_fn():
            return 42

        result, data = run_step("test_step", simple_fn)
        assert isinstance(result, StepResult)
        assert result.status == "success"
        assert result.step_name == "test_step"
        assert result.error is None
        assert data == 42

    def test_timing_recorded(self):
        """Elapsed time should be recorded and > 0."""
        def slow_fn():
            time.sleep(0.05)
            return "done"

        result, _ = run_step("timed_step", slow_fn)
        assert result.timing_seconds >= 0.04

    def test_args_and_kwargs_passed(self):
        """Arguments and keyword arguments should be forwarded to the function."""
        def adder(a, b, multiplier=1):
            return (a + b) * multiplier

        result, data = run_step("adder", adder, 3, 4, multiplier=2)
        assert data == 14

    def test_input_summary_recorded(self):
        """Input summary should be stored in the StepResult."""
        result, _ = run_step(
            "summarized",
            lambda: "ok",
            input_summary={"rows": 100, "cols": 5},
        )
        assert result.input_summary == {"rows": 100, "cols": 5}

    def test_output_summary_fn_called(self):
        """Output summary function should be invoked on the return value."""
        def produce():
            return [1, 2, 3]

        result, data = run_step(
            "with_summary",
            produce,
            output_summary_fn=lambda x: {"count": len(x)},
        )
        assert result.output_summary == {"count": 3}

    def test_output_summary_skipped_for_none(self):
        """If fn returns None, output_summary_fn should not be called."""
        def returns_none():
            return None

        called = []
        result, data = run_step(
            "none_result",
            returns_none,
            output_summary_fn=lambda x: called.append(True) or {"n": 0},
        )
        assert data is None
        assert len(called) == 0


class TestRunStepErrorHandling:
    """Tests for error handling in step execution."""

    def test_expected_exception_caught(self):
        """FileNotFoundError should be caught and produce an error StepResult."""
        def fails():
            raise FileNotFoundError("data.csv not found")

        result, data = run_step("failing_step", fails)
        assert result.status == "error"
        assert "data.csv not found" in result.error
        assert data is None

    def test_value_error_caught(self):
        """ValueError is in the default expected_exceptions."""
        def bad_value():
            raise ValueError("negative radiance")

        result, data = run_step("bad_value", bad_value)
        assert result.status == "error"
        assert data is None

    def test_key_error_caught(self):
        """KeyError is in the default expected_exceptions."""
        def missing_key():
            raise KeyError("district")

        result, data = run_step("missing_key", missing_key)
        assert result.status == "error"

    def test_empty_data_error_caught(self):
        """pd.errors.EmptyDataError is in the default expected_exceptions."""
        def empty_csv():
            raise pd.errors.EmptyDataError("No columns to parse")

        result, data = run_step("empty_csv", empty_csv)
        assert result.status == "error"

    def test_unexpected_exception_also_caught(self):
        """Exceptions NOT in expected_exceptions should still be caught
        (with different log level) but produce error status."""
        def unexpected():
            raise RuntimeError("unexpected crash")

        result, data = run_step("unexpected", unexpected)
        assert result.status == "error"
        assert "unexpected crash" in result.error

    def test_custom_expected_exceptions(self):
        """Custom expected_exceptions should override the defaults."""
        def type_error():
            raise TypeError("wrong type")

        result, data = run_step(
            "custom",
            type_error,
            expected_exceptions=(TypeError,),
        )
        assert result.status == "error"

    def test_error_timing_still_recorded(self):
        """Even on error, timing should be recorded."""
        def fails_slowly():
            time.sleep(0.05)
            raise ValueError("slow fail")

        result, _ = run_step("slow_fail", fails_slowly)
        assert result.timing_seconds >= 0.04


class TestRunStepEdgeCases:
    """Edge cases for step execution."""

    def test_function_returning_empty_dataframe(self):
        """An empty DataFrame is a valid (non-None) return value."""
        def empty_df():
            return pd.DataFrame()

        result, data = run_step(
            "empty_df",
            empty_df,
            output_summary_fn=lambda df: {"rows": len(df)},
        )
        assert result.status == "success"
        assert data is not None
        assert result.output_summary == {"rows": 0}

    def test_function_returning_false(self):
        """False is a valid return value, not None."""
        def returns_false():
            return False

        called = []
        result, data = run_step(
            "false_result",
            returns_false,
            output_summary_fn=lambda x: called.append(True) or {"val": x},
        )
        assert result.status == "success"
        assert data is False
        # output_summary_fn should NOT be called because result_data is False (falsy)
        # Actually, the code checks `result_data is not None`, so False should trigger it
        assert len(called) == 1

    def test_no_input_summary_defaults_to_empty(self):
        """When input_summary is not provided, should default to empty dict."""
        result, _ = run_step("no_summary", lambda: 1)
        assert result.input_summary == {}
