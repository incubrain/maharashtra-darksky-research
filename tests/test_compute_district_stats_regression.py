"""Regression test for the MemoryFile-lifecycle bug in compute_district_stats.

Background
----------
The original implementation piped the filtered raster through a
``rasterio.io.MemoryFile`` and passed ``src.name`` (a ``/vsimem/…`` path) to
``rasterstats.zonal_stats``. In isolated calls this worked, but when exercised
end-to-end via ``step_process_years`` the resulting DataFrame came back with
every statistic set to ``None`` and ``pixel_count=0`` — the MemoryFile context
closed before the DataFrame materialised, leaving rasterstats' results
referencing a freed virtual file.

This test reproduces the failure mode: it calls ``step_process_years`` for
a single year against the on-disk subset rasters that the preprocessing step
produced, and asserts that real urban districts (Pune, Mumbai Suburban) have
non-zero pixel counts and finite mean radiance. With the buggy version, every
row was ``NaN``; with the fix (direct ndarray + affine) the expected numbers
come through.

Skipped if the 2024 subsets aren't present (e.g. fresh clone without VIIRS
data) so CI in that mode stays green.
"""

from __future__ import annotations

import os

import pytest

SUBSETS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "subsets", "2024"
)
REQUIRED_LAYERS = ["average", "cf_cvg", "lit_mask", "median"]


def _have_2024_subsets() -> bool:
    return all(
        os.path.exists(os.path.join(SUBSETS_DIR, f"maharashtra_{layer}_2024.tif"))
        for layer in REQUIRED_LAYERS
    )


@pytest.mark.skipif(not _have_2024_subsets(), reason="2024 VIIRS subsets not present")
def test_compute_district_stats_returns_real_numbers():
    """Directly call compute_district_stats on the 2024 median raster."""
    import numpy as np
    import geopandas as gpd
    from src.viirs_process import apply_quality_filters, compute_district_stats

    gdf = gpd.read_file(
        os.path.join(os.path.dirname(__file__), "..", "data", "shapefiles", "maharashtra_district.geojson")
    )
    filtered, _meta, transform = apply_quality_filters(
        median_path=os.path.join(SUBSETS_DIR, "maharashtra_median_2024.tif"),
        lit_mask_path=os.path.join(SUBSETS_DIR, "maharashtra_lit_mask_2024.tif"),
        cf_cvg_path=os.path.join(SUBSETS_DIR, "maharashtra_cf_cvg_2024.tif"),
        cf_threshold=5,
    )

    df = compute_district_stats(filtered, transform, gdf)

    assert len(df) == len(gdf), "one row per district expected"

    # No district may be empty — the quality filters don't wipe every pixel.
    empty = df[df["pixel_count"] == 0]
    assert empty.empty, f"{len(empty)} districts had pixel_count=0 — regression to MemoryFile bug?\n{empty}"

    # All stats finite for every district.
    for col in ["mean_radiance", "median_radiance", "min_radiance", "max_radiance", "std_radiance"]:
        assert df[col].notna().all(), f"{col} has NaN values — regression"
        assert np.isfinite(df[col]).all(), f"{col} has non-finite values"

    # Sanity: urban districts are bright, rural forested districts are dim.
    pune = df[df["district"] == "Pune"].iloc[0]
    mumbai_sub = df[df["district"] == "Mumbai Suburban"].iloc[0]
    gadchiroli = df[df["district"] == "Gadchiroli"].iloc[0]

    assert pune["pixel_count"] > 10_000, f"Pune pixel_count suspiciously low: {pune['pixel_count']}"
    assert 1.0 < pune["mean_radiance"] < 20.0, f"Pune mean out of plausible range: {pune['mean_radiance']}"
    assert mumbai_sub["mean_radiance"] > 10.0, f"Mumbai Suburban should be bright: {mumbai_sub['mean_radiance']}"
    assert gadchiroli["mean_radiance"] < 3.0, f"Gadchiroli should be dim: {gadchiroli['mean_radiance']}"


@pytest.mark.skipif(not _have_2024_subsets(), reason="2024 VIIRS subsets not present")
def test_step_process_years_returns_populated_dataframe():
    """End-to-end through the pipeline orchestration for a single year.

    The original bug only manifested when called via ``step_process_years``
    (not when ``process_single_year`` was invoked directly), so this test
    guards the orchestration path specifically.
    """
    import geopandas as gpd
    from src.pipeline_steps import step_process_years
    from src import config

    gdf = gpd.read_file(
        os.path.join(os.path.dirname(__file__), "..", "data", "shapefiles", "maharashtra_district.geojson")
    )

    result, yearly_df = step_process_years(
        years=[2024],
        viirs_dir=config.DEFAULT_VIIRS_DIR,
        gdf=gdf,
        output_dir=config.DEFAULT_OUTPUT_DIR,
        cf_threshold=5,
    )

    assert result.status == "success"
    assert yearly_df is not None
    assert len(yearly_df) == 36
    # The buggy version produced object-dtype columns full of None; the fix
    # produces float64 columns with real numbers.
    assert str(yearly_df["mean_radiance"].dtype).startswith("float"), \
        f"mean_radiance dtype is {yearly_df['mean_radiance'].dtype!r} — fix regressed"
    assert yearly_df["mean_radiance"].notna().all(), "mean_radiance has NaN after pipeline step"
    assert (yearly_df["pixel_count"] > 0).all(), "some districts have pixel_count=0"
