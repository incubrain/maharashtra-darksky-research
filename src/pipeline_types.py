"""
Typed result dataclasses for pipeline step tracking.

These types standardize what each pipeline step returns, enabling
structured logging, validation gates, and provenance tracking.
"""

import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class StepStatus(str, Enum):
    """Pipeline step outcome status."""
    SUCCESS = "success"
    SKIPPED = "skipped"
    ERROR = "error"


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _get_git_sha():
    """Return the short git SHA of the current HEAD, or None."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


@dataclass
class StepResult:
    """Result of a single pipeline step execution."""

    step_name: str
    status: str  # "success", "skipped", "error"
    input_summary: dict = field(default_factory=dict)
    output_summary: dict = field(default_factory=dict)
    timing_seconds: float = 0.0
    warnings: list = field(default_factory=list)
    error: Optional[str] = None
    started_at: str = field(default_factory=_now_iso)
    completed_at: Optional[str] = None
    nan_summary: Optional[dict] = None

    @property
    def ok(self):
        return self.status == "success"

    def to_dict(self):
        return {
            "step_name": self.step_name,
            "status": self.status,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "timing_seconds": self.timing_seconds,
            "warnings": self.warnings,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "nan_summary": self.nan_summary,
        }

    @classmethod
    def from_dict(cls, d):
        """Reconstruct a StepResult from a serialized dict."""
        return cls(
            step_name=d["step_name"],
            status=d["status"],
            input_summary=d.get("input_summary", {}),
            output_summary=d.get("output_summary", {}),
            timing_seconds=d.get("timing_seconds", 0.0),
            warnings=d.get("warnings", []),
            error=d.get("error"),
            started_at=d.get("started_at", ""),
            completed_at=d.get("completed_at"),
            nan_summary=d.get("nan_summary"),
        )


@dataclass
class YearProcessingResult:
    """Result of processing a single year of VIIRS data."""

    year: int
    districts_processed: int = 0
    pixels_total: int = 0
    pixels_valid: int = 0
    layers_found: list = field(default_factory=list)
    step_result: Optional[StepResult] = None

    @property
    def pct_valid(self):
        if self.pixels_total == 0:
            return 0.0
        return (self.pixels_valid / self.pixels_total) * 100


@dataclass
class PipelineRunResult:
    """Result of a complete pipeline execution."""

    run_dir: str = ""
    entity_type: str = ""  # "district", "city", "site", "all"
    years_processed: list = field(default_factory=list)
    step_results: list = field(default_factory=list)
    total_time_seconds: float = 0.0
    output_files: list = field(default_factory=list)
    git_sha: Optional[str] = field(default_factory=_get_git_sha)
    started_at: str = field(default_factory=_now_iso)

    @property
    def all_ok(self):
        return all(s.ok for s in self.step_results)

    @property
    def failed_steps(self):
        return [s for s in self.step_results if not s.ok]

    def to_dict(self):
        return {
            "run_dir": self.run_dir,
            "entity_type": self.entity_type,
            "years_processed": self.years_processed,
            "steps": [s.to_dict() for s in self.step_results],
            "total_time_seconds": self.total_time_seconds,
            "output_files": self.output_files,
            "all_ok": self.all_ok,
            "git_sha": self.git_sha,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, d):
        """Reconstruct a PipelineRunResult from a serialized dict."""
        result = cls(
            run_dir=d.get("run_dir", ""),
            entity_type=d.get("entity_type", ""),
            years_processed=d.get("years_processed", []),
            total_time_seconds=d.get("total_time_seconds", 0.0),
            output_files=d.get("output_files", []),
            git_sha=d.get("git_sha"),
            started_at=d.get("started_at", ""),
        )
        result.step_results = [
            StepResult.from_dict(s) for s in d.get("steps", [])
        ]
        return result
