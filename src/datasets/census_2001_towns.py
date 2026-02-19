"""Census 2001 PCA dataset — town level."""

import pandas as pd

from src.datasets._base import DatasetMeta
from src.datasets._census_town_loader import load_census_towns_csv, validate_census_towns


def get_meta() -> DatasetMeta:
    return DatasetMeta(
        name="census_2001_towns",
        short_label="c2001t",
        description="Census of India 2001 PCA — town level (379 towns)",
        temporal_type="snapshot",
        entity_type="town",
        reference_years=[2001],
        entity_col="district",
        source_url="https://censusindia.gov.in/",
        citation="Office of the Registrar General & Census Commissioner, India. Census of India 2001.",
    )


def load_and_process(data_dir, entity_col="district", vnl_district_names=None):
    return load_census_towns_csv(data_dir, 2001, get_meta(), vnl_district_names, entity_col)


def validate(df: pd.DataFrame, entity_col: str = "district") -> list[str]:
    return validate_census_towns(df, get_meta().short_label, entity_col)
