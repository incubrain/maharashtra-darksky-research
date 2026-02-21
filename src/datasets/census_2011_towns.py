"""Census 2011 PCA dataset — town level."""

from src.datasets._base import DatasetMeta
from src.datasets._census_factory import make_town_dataset

get_meta, load_and_process, validate = make_town_dataset(DatasetMeta(
    name="census_2011_towns",
    short_label="c2011t",
    description="Census of India 2011 PCA — town level (537 towns)",
    temporal_type="snapshot",
    entity_type="town",
    reference_years=[2011],
    entity_col="district",
    source_url="https://censusindia.gov.in/",
    citation="Office of the Registrar General & Census Commissioner, India. Census of India 2011.",
))
