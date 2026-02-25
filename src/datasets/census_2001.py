"""Census 2001 PCA dataset — urban district-level data."""

from src.datasets._base import DatasetMeta
from src.census.factory import make_district_dataset

get_meta, load_and_process, validate = make_district_dataset(DatasetMeta(
    name="census_2001",
    short_label="c2001",
    description="Census of India 2001 PCA (urban)",
    temporal_type="snapshot",
    reference_years=[2001],
    entity_col="district",
    source_url="https://censusindia.gov.in/",
    citation="Office of the Registrar General & Census Commissioner, India. Census of India 2001.",
))
