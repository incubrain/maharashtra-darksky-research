"""
Centralized logging configuration for the VIIRS ALAN analysis pipeline.

Provides structured JSON Lines logging to files and human-readable
console output. All pipeline modules should use get_pipeline_logger()
instead of calling logging.basicConfig() directly.

Usage:
    from src.logging_config import get_pipeline_logger
    log = get_pipeline_logger(__name__)
"""

import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler


class JsonFormatter(logging.Formatter):
    """Format log records as JSON Lines for machine parsing."""

    def format(self, record):
        entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        # Include structured extra fields if present.
        for key in ("step_name", "input_summary", "output_summary",
                     "timing_seconds", "record_count", "warnings"):
            if hasattr(record, key):
                entry[key] = getattr(record, key)
        return json.dumps(entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable format matching the existing pipeline style."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# Track whether root logging has been configured to avoid duplicates.
_configured = False
_run_dir_handler = None


def setup_logging(run_dir=None, console_level=logging.INFO, file_level=logging.DEBUG):
    """Configure root logger with console and optional file handlers.

    Call once at pipeline entry point. Subsequent calls are no-ops unless
    run_dir changes.

    Parameters
    ----------
    run_dir : str, optional
        Directory for per-run log file. If provided, creates
        ``{run_dir}/pipeline.jsonl`` with JSON Lines at file_level.
    console_level : int
        Console handler log level. Default: INFO.
    file_level : int
        File handler log level. Default: DEBUG.
    """
    global _configured, _run_dir_handler

    root = logging.getLogger()

    if not _configured:
        root.setLevel(logging.DEBUG)

        # Console handler: human-readable at INFO.
        console = logging.StreamHandler()
        console.setLevel(console_level)
        console.setFormatter(ConsoleFormatter())
        root.addHandler(console)

        # Rotating file handler: pipeline.log (survives across runs).
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        rotating = RotatingFileHandler(
            os.path.join(log_dir, "pipeline.log"),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3,
        )
        rotating.setLevel(file_level)
        rotating.setFormatter(JsonFormatter())
        root.addHandler(rotating)

        _configured = True

    # Per-run JSON Lines log.
    if run_dir and _run_dir_handler is None:
        os.makedirs(run_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(run_dir, "pipeline.jsonl"))
        fh.setLevel(file_level)
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)
        _run_dir_handler = fh


def get_pipeline_logger(name, run_dir=None):
    """Get a logger for a pipeline module.

    If logging has not been set up yet, initialises with defaults.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__``).
    run_dir : str, optional
        Passed to setup_logging() if not yet configured.

    Returns
    -------
    logging.Logger
    """
    if not _configured:
        setup_logging(run_dir=run_dir)
    return logging.getLogger(name)


def log_step_summary(
    logger,
    step_name,
    status="success",
    input_summary=None,
    output_summary=None,
    timing_seconds=None,
    warnings_list=None,
):
    """Log a structured step summary at INFO level.

    Parameters
    ----------
    logger : logging.Logger
    step_name : str
    status : str
        "success", "skipped", or "error".
    input_summary : dict, optional
    output_summary : dict, optional
    timing_seconds : float, optional
    warnings_list : list[str], optional
    """
    parts = [f"[{step_name}] {status}"]
    if timing_seconds is not None:
        parts.append(f"({timing_seconds:.1f}s)")
    if output_summary:
        parts.append(f"output={output_summary}")

    extra = {"step_name": step_name}
    if input_summary:
        extra["input_summary"] = input_summary
    if output_summary:
        extra["output_summary"] = output_summary
    if timing_seconds is not None:
        extra["timing_seconds"] = timing_seconds
    if warnings_list:
        extra["warnings"] = warnings_list

    logger.info(" ".join(parts), extra=extra)


class StepTimer:
    """Context manager for timing pipeline steps.

    Usage:
        with StepTimer() as t:
            do_work()
        print(t.elapsed)
    """

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
