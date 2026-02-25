"""Census 1991 PCA dataset — town level."""

from src.datasets._base import DatasetMeta
from src.census.factory import make_town_dataset

get_meta, load_and_process, validate = make_town_dataset(DatasetMeta(
    name="census_1991_towns",
    short_label="c1991t",
    description="Census of India 1991 PCA — town level (336 towns)",
    temporal_type="snapshot",
    entity_type="town",
    reference_years=[1991],
    entity_col="district",
    source_url="https://censusindia.gov.in/",
    citation="Office of the Registrar General & Census Commissioner, India. Census of India 1991.",
))
