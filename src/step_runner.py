"""
Generic step executor for pipeline steps.

Eliminates the ~45-line boilerplate pattern (StepTimer, try/except,
log_step_summary, StepResult construction) that was duplicated across
every step function.  Each step now just provides the work function
and metadata; ``run_step()`` handles the rest.
"""

import traceback
from typing import TypeVar, Callable

import pandas as pd

from src.logging_config import StepTimer, get_pipeline_logger, log_step_summary
from src.pipeline_types import StepResult

T = TypeVar("T")

log = get_pipeline_logger(__name__)

_DEFAULT_EXPECTED = (
    FileNotFoundError,
    ValueError,
    KeyError,
    pd.errors.EmptyDataError,
    pd.errors.MergeError,
)


def run_step(
    step_name: str,
    fn: Callable[..., T],
    *args,
    input_summary: dict | None = None,
    output_summary_fn: Callable[[T], dict] | None = None,
    expected_exceptions: tuple[type[Exception], ...] = _DEFAULT_EXPECTED,
    **kwargs,
) -> tuple[StepResult, T | None]:
    """Execute a pipeline step with standardised error handling and timing.

    Parameters
    ----------
    step_name : str
        Human-readable name stored in StepResult for provenance.
    fn : Callable
        The work function.  Called as ``fn(*args, **kwargs)``.
    input_summary : dict, optional
        Metadata about inputs (logged in StepResult).
    output_summary_fn : callable, optional
        Receives *fn*'s return value and produces an output-summary dict.
        Skipped when *fn* raises or returns None.
    expected_exceptions : tuple
        Exception types that produce a "known error" log message.

    Returns
    -------
    tuple[StepResult, T | None]
    """
    result_data = None
    error_tb = None

    with StepTimer() as timer:
        try:
            result_data = fn(*args, **kwargs)
        except expected_exceptions as exc:
            error_tb = traceback.format_exc()
            log.error("%s failed: %s", step_name, exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("%s failed unexpectedly", step_name, exc_info=True)

    if error_tb:
        log_step_summary(log, step_name, "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name=step_name,
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    out_summary = {}
    if output_summary_fn is not None and result_data is not None:
        out_summary = output_summary_fn(result_data)

    log_step_summary(
        log, step_name, "success",
        input_summary=input_summary or {},
        output_summary=out_summary,
        timing_seconds=timer.elapsed,
    )
    return StepResult(
        step_name=step_name,
        status="success",
        input_summary=input_summary or {},
        output_summary=out_summary,
        timing_seconds=timer.elapsed,
    ), result_data
