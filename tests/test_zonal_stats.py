"""
Tests for zonal statistics computation (per-district aggregation).

Verifies that compute_district_stats correctly:
1. Computes mean, median, count, min, max, std per district
2. Handles districts with all-NaN pixels (no valid data)
3. Returns one row per district
4. Produces correct values for uniform rasters
"""

import numpy as np
import pandas as pd
import pytest

from src.viirs_process import compute_district_stats
from tests.conftest import HEIGHT, WIDTH, TRANSFORM, MID_LON


class TestComputeDistrictStats:

    def test_uniform_raster_stats(self, two_district_gdf):
        """Uniform raster of 5.0 should give mean=median=min=max=5.0, std≈0."""
        data = np.full((HEIGHT, WIDTH), 5.0, dtype="float32")
        df = compute_district_stats(data, TRANSFORM, two_district_gdf)

        assert len(df) == 2, "Should return one row per district"
        assert set(df["district"]) == {"DistrictA", "DistrictB"}

        for _, row in df.iterrows():
            assert row["mean_radiance"] == pytest.approx(5.0, abs=0.01)
            assert row["median_radiance"] == pytest.approx(5.0, abs=0.01)
            assert row["min_radiance"] == pytest.approx(5.0, abs=0.01)
            assert row["max_radiance"] == pytest.approx(5.0, abs=0.01)
            assert row["std_radiance"] == pytest.approx(0.0, abs=0.01)
            assert row["pixel_count"] > 0

    def test_different_values_per_district(self, two_district_gdf):
        """West half=2.0, east half=8.0 should give distinct stats per district."""
        data = np.full((HEIGHT, WIDTH), 8.0, dtype="float32")
        data[:, :WIDTH // 2] = 2.0  # West half = DistrictA

        df = compute_district_stats(data, TRANSFORM, two_district_gdf)
        row_a = df[df["district"] == "DistrictA"].iloc[0]
        row_b = df[df["district"] == "DistrictB"].iloc[0]

        assert row_a["median_radiance"] == pytest.approx(2.0, abs=0.5)
        assert row_b["median_radiance"] == pytest.approx(8.0, abs=0.5)

    def test_all_nan_raster(self, two_district_gdf):
        """Completely NaN raster should return 0 count and NaN stats for all districts."""
        data = np.full((HEIGHT, WIDTH), np.nan, dtype="float32")

        df = compute_district_stats(data, TRANSFORM, two_district_gdf)

        for _, row in df.iterrows():
            assert row["pixel_count"] == 0 or pd.isna(row["median_radiance"]), (
                f"District {row['district']} should have no valid pixels"
            )

    def test_pixel_count_proportional_to_area(self, two_district_gdf):
        """Both districts cover ~equal area, so pixel counts should be similar."""
        data = np.full((HEIGHT, WIDTH), 5.0, dtype="float32")
        df = compute_district_stats(data, TRANSFORM, two_district_gdf)

        count_a = df[df["district"] == "DistrictA"]["pixel_count"].values[0]
        count_b = df[df["district"] == "DistrictB"]["pixel_count"].values[0]

        # Equal-area rectangles should have similar pixel counts (within 10%)
        assert count_a == pytest.approx(count_b, rel=0.1)
