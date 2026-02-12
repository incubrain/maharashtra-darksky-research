"""
Typed result dataclasses for pipeline step tracking.

These types standardize what each pipeline step returns, enabling
structured logging, validation gates, and provenance tracking.
"""

from dataclasses import dataclass, field
from typing import Optional


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
        }


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
        }
