"""Protected area verification map generation."""

import os

from src.geocoding.verification import load_all_csvs, plot_verification_map, check_bbox
from src.protected_areas.constants import CATEGORY_COLORS, CATEGORY_MARKERS

DATA_DIR = "data/protected_areas"


def load_all_sites(data_dir: str = DATA_DIR):
    """Load all protected area CSVs into a single DataFrame."""
    return load_all_csvs(data_dir)


def verify_sites(sites, sample, seed, output_dir=DATA_DIR):
    """Generate a verification map for protected areas."""
    return plot_verification_map(
        sites, sample, seed,
        title_prefix="Maharashtra Protected Areas",
        category_col="category",
        colors=CATEGORY_COLORS,
        markers=CATEGORY_MARKERS,
        output_path=os.path.join(output_dir, "verification_map.png"),
    )
