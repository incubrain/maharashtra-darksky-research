"""
Dataset module registry.

Each dataset is a self-contained module exporting get_meta(),
load_and_process(), and validate(). Register new datasets here.

The registry is built lazily to avoid circular imports between
src.datasets and src.census modules.
"""

_DATASET_REGISTRY = None


def _build_registry() -> dict:
    from src.datasets import (
        census_1991,
        census_2001,
        census_2011,
        census_1991_towns,
        census_2001_towns,
        census_2011_towns,
    )
    from src.census import (
        legacy_pca as census_2011_pca,
        projected as census_projected,
        towns_projected as census_towns_projected,
    )

    return {
        # District-level (one row per district, Total population)
        "census_1991": census_1991,
        "census_2001": census_2001,
        "census_2011": census_2011,
        "census_2011_pca": census_2011_pca,
        "census_projected": census_projected,
        # Town-level (one row per town)
        "census_1991_towns": census_1991_towns,
        "census_2001_towns": census_2001_towns,
        "census_2011_towns": census_2011_towns,
        "census_towns_projected": census_towns_projected,
    }


def __getattr__(name):
    global _DATASET_REGISTRY
    if name == "DATASET_REGISTRY":
        if _DATASET_REGISTRY is None:
            _DATASET_REGISTRY = _build_registry()
        return _DATASET_REGISTRY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
