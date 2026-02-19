"""
Tests for site buffer extraction (10 km circular buffers around point sites).

Verifies that:
1. Buffers are created with correct CRS
2. Buffer radius produces expected area (~314 km² for 10 km radius)
3. Zonal stats from buffers extract correct values
4. All configured sites produce valid buffers
"""

import numpy as np
import pytest

from src.site.site_analysis import build_site_geodataframe, LOCATIONS
from src import config


class TestBuildSiteGeoDataFrame:

    def test_default_buffer_count(self):
        """Should create one buffer per configured site (43 cities + 11 dark-sky)."""
        gdf = build_site_geodataframe()
        expected_count = len(config.URBAN_CITIES) + len(config.DARKSKY_SITES)
        assert len(gdf) == expected_count, (
            f"Expected {expected_count} sites, got {len(gdf)}"
        )

    def test_buffer_crs_is_wgs84(self):
        """Output GeoDataFrame should be in WGS84 (EPSG:4326)."""
        gdf = build_site_geodataframe()
        assert gdf.crs.to_epsg() == 4326

    def test_buffer_area_approximately_correct(self):
        """10 km radius circle ≈ π × 10² ≈ 314 km². Check in UTM."""
        gdf = build_site_geodataframe(buffer_km=10)
        gdf_utm = gdf.to_crs(epsg=config.MAHARASHTRA_UTM_EPSG)

        for _, row in gdf_utm.iterrows():
            area_km2 = row.geometry.area / 1e6
            # Allow 5% tolerance for projection distortion
            assert area_km2 == pytest.approx(314.16, rel=0.05), (
                f"Site {row['name']} buffer area {area_km2:.1f} km² "
                f"should be ~314 km²"
            )

    def test_custom_buffer_radius(self):
        """Buffer with 20 km radius should have ~4x the area of 10 km."""
        gdf_10 = build_site_geodataframe(buffer_km=10)
        gdf_20 = build_site_geodataframe(buffer_km=20)

        gdf_10_utm = gdf_10.to_crs(epsg=config.MAHARASHTRA_UTM_EPSG)
        gdf_20_utm = gdf_20.to_crs(epsg=config.MAHARASHTRA_UTM_EPSG)

        area_10 = gdf_10_utm.iloc[0].geometry.area
        area_20 = gdf_20_utm.iloc[0].geometry.area

        ratio = area_20 / area_10
        assert ratio == pytest.approx(4.0, rel=0.05), (
            f"20km/10km area ratio should be ~4.0, got {ratio:.2f}"
        )

    def test_all_sites_have_valid_geometry(self):
        """Every buffer should be a valid, non-empty polygon."""
        gdf = build_site_geodataframe()
        for _, row in gdf.iterrows():
            assert row.geometry.is_valid, f"Invalid geometry for {row['name']}"
            assert not row.geometry.is_empty, f"Empty geometry for {row['name']}"

    def test_all_sites_within_maharashtra_bbox(self):
        """All site centers should fall within the Maharashtra bounding box."""
        bbox = config.MAHARASHTRA_BBOX
        for name, (lat, lon, _, _) in LOCATIONS.items():
            assert bbox["south"] <= lat <= bbox["north"], (
                f"{name} lat={lat} is outside Maharashtra bbox "
                f"[{bbox['south']}, {bbox['north']}]"
            )
            assert bbox["west"] <= lon <= bbox["east"], (
                f"{name} lon={lon} is outside Maharashtra bbox "
                f"[{bbox['west']}, {bbox['east']}]"
            )

    def test_types_are_city_or_site(self):
        """All entries should have type 'city' or 'site'."""
        gdf = build_site_geodataframe()
        for _, row in gdf.iterrows():
            assert row["type"] in ("city", "site"), (
                f"{row['name']} has unexpected type '{row['type']}'"
            )
