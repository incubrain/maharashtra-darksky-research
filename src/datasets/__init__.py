"""
Dataset module registry.

Each dataset is a self-contained module exporting get_meta(),
load_and_process(), and validate(). Register new datasets here.
"""

from src.datasets import (
    census_1991,
    census_2001,
    census_2011,
    census_2011_pca,  # legacy adapter kept for backward compat
    census_projected,
    census_1991_towns,
    census_2001_towns,
    census_2011_towns,
    census_towns_projected,
)

DATASET_REGISTRY: dict = {
    # District-level (one row per district, Total population)
    "census_1991": census_1991,
    "census_2001": census_2001,
    "census_2011": census_2011,
    "census_2011_pca": census_2011_pca,  # legacy
    "census_projected": census_projected,
    # Town-level (one row per town)
    "census_1991_towns": census_1991_towns,
    "census_2001_towns": census_2001_towns,
    "census_2011_towns": census_2011_towns,
    "census_towns_projected": census_towns_projected,
}
