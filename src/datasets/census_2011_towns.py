"""Census 2011 PCA dataset — town level."""

import pandas as pd

from src.datasets._base import DatasetMeta
from src.datasets._census_town_loader import load_census_towns_csv, validate_census_towns


def get_meta() -> DatasetMeta:
    return DatasetMeta(
        name="census_2011_towns",
        short_label="c2011t",
        description="Census of India 2011 PCA — town level (537 towns)",
        temporal_type="snapshot",
        entity_type="town",
        reference_years=[2011],
        entity_col="district",
        source_url="https://censusindia.gov.in/",
        citation="Office of the Registrar General & Census Commissioner, India. Census of India 2011.",
    )


def load_and_process(data_dir, entity_col="district", vnl_district_names=None):
    return load_census_towns_csv(data_dir, 2011, get_meta(), vnl_district_names, entity_col)


def validate(df: pd.DataFrame, entity_col: str = "district") -> list[str]:
    return validate_census_towns(df, get_meta().short_label, entity_col)
