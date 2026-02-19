"""
Shared types for the dataset module system.

Every dataset module returns DatasetMeta (metadata) and DatasetResult
(processing outcome) to enable uniform aggregation and reporting.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetMeta:
    """Metadata describing an external dataset."""

    name: str  # e.g. "census_2011_pca"
    short_label: str  # e.g. "c2011" — used in file naming
    description: str
    temporal_type: str  # "snapshot" | "timeseries"
    entity_type: str = "district"  # "district" | "town"
    reference_years: list[int] = field(default_factory=list)
    entity_col: str = "district"
    source_url: str = ""
    citation: str = ""


@dataclass
class DatasetResult:
    """Outcome of loading and processing a dataset."""

    dataset_name: str
    status: str  # "success" | "error"
    districts_matched: int = 0
    districts_unmatched: list[str] = field(default_factory=list)
    columns_produced: list[str] = field(default_factory=list)
    rows: int = 0
    timing_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self):
        return self.status == "success"
