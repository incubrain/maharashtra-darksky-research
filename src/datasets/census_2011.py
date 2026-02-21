"""Census 2011 PCA dataset — urban district-level data."""

from src.datasets._base import DatasetMeta
from src.datasets._census_factory import make_district_dataset

get_meta, load_and_process, validate = make_district_dataset(DatasetMeta(
    name="census_2011",
    short_label="c2011",
    description="Census of India 2011 PCA (urban)",
    temporal_type="snapshot",
    reference_years=[2011],
    entity_col="district",
    source_url="https://censusindia.gov.in/",
    citation="Office of the Registrar General & Census Commissioner, India. Census of India 2011.",
))
