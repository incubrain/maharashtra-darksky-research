"""Monument verification map generation."""

import os

from src.geocoding.verification import load_all_csvs, plot_verification_map, check_bbox
from src.monuments.constants import TYPE_COLORS, TYPE_MARKERS

DATA_DIR = "data/monuments"


def load_all_monuments(data_dir: str = DATA_DIR):
    """Load all monument CSVs into a single DataFrame."""
    return load_all_csvs(data_dir)


def verify_monuments(sites, sample, seed, output_dir=DATA_DIR):
    """Generate a verification map for monuments."""
    return plot_verification_map(
        sites, sample, seed,
        title_prefix="Maharashtra Monuments",
        category_col="monument_type",
        colors=TYPE_COLORS,
        markers=TYPE_MARKERS,
        output_path=os.path.join(output_dir, "verification_map.png"),
    )
