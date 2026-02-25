"""
Factory for creating census dataset modules from configuration.

Each thin census adapter follows an identical pattern: get_meta() returns
a DatasetMeta, load_and_process() delegates to the shared loader, and
validate() delegates to the shared validator.

This factory generates those three functions from a DatasetMeta instance.
"""

import pandas as pd

from src.datasets._base import DatasetMeta
from src.census.loader import load_census_csv, validate_census
from src.census.town_loader import load_census_towns_csv, validate_census_towns


def make_district_dataset(meta: DatasetMeta):
    """Create get_meta / load_and_process / validate for a district-level census dataset."""
    year = meta.reference_years[0]

    def get_meta() -> DatasetMeta:
        return meta

    def load_and_process(data_dir, entity_col="district", vnl_district_names=None):
        return load_census_csv(data_dir, year, meta, vnl_district_names, entity_col)

    def validate(df: pd.DataFrame, entity_col: str = "district") -> list[str]:
        return validate_census(df, meta.short_label, entity_col)

    return get_meta, load_and_process, validate


def make_town_dataset(meta: DatasetMeta):
    """Create get_meta / load_and_process / validate for a town-level census dataset."""
    year = meta.reference_years[0]

    def get_meta() -> DatasetMeta:
        return meta

    def load_and_process(data_dir, entity_col="district", vnl_district_names=None):
        return load_census_towns_csv(data_dir, year, meta, vnl_district_names, entity_col)

    def validate(df: pd.DataFrame, entity_col: str = "district") -> list[str]:
        return validate_census_towns(df, meta.short_label, entity_col)

    return get_meta, load_and_process, validate
