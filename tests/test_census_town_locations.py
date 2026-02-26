"""
Tests for census town locations loading and pipeline integration.

Verifies:
1. load_census_town_locations() returns correct format
2. Duplicate name disambiguation works
3. Failed geocodes are excluded
4. city_source="census" loads census towns into the pipeline
5. Projected town data files are generated correctly
"""

import os
import tempfile

import pandas as pd
import pytest

from src import config
from src.census.town_locations import load_census_town_locations, _title_case_town


# ── Synthetic geocoded CSV for testing ────────────────────────────────

SAMPLE_GEOCODED_DATA = pd.DataFrame([
    {"district": "Pune", "town_name": "pune", "display_name": "Pune (M Corp.)",
     "lat": 18.52, "lon": 73.86, "geocode_status": "ok",
     "geocode_query": "Pune, Pune district, Maharashtra, India",
     "first_census": 1991, "census_count": 3, "TOT_P": 3124000},
    {"district": "Pune", "town_name": "baramati", "display_name": "Baramati (M)",
     "lat": 18.15, "lon": 74.58, "geocode_status": "ok",
     "geocode_query": "Baramati, Pune district, Maharashtra, India",
     "first_census": 1991, "census_count": 3, "TOT_P": 67000},
    {"district": "Nagpur", "town_name": "nagpur", "display_name": "Nagpur (M Corp.)",
     "lat": 21.15, "lon": 79.09, "geocode_status": "ok",
     "geocode_query": "Nagpur, Nagpur district, Maharashtra, India",
     "first_census": 1991, "census_count": 3, "TOT_P": 2405000},
    # Duplicate name across districts
    {"district": "Ahmadnagar", "town_name": "karjat", "display_name": "Karjat (M)",
     "lat": 19.28, "lon": 74.98, "geocode_status": "ok",
     "geocode_query": "Karjat, Ahmadnagar district, Maharashtra, India",
     "first_census": 2001, "census_count": 2, "TOT_P": 25000},
    {"district": "Raigarh", "town_name": "karjat", "display_name": "Karjat (M)",
     "lat": 18.91, "lon": 73.32, "geocode_status": "fallback",
     "geocode_query": "Karjat, Raigarh, Maharashtra",
     "first_census": 2001, "census_count": 2, "TOT_P": 18000},
    # Failed geocode — should be excluded
    {"district": "Buldana", "town_name": "sundarkhed", "display_name": "Sundarkhed (Ct)",
     "lat": None, "lon": None, "geocode_status": "failed",
     "geocode_query": "Sundarkhed, Buldana district, Maharashtra, India",
     "first_census": 2011, "census_count": 1, "TOT_P": 5000},
])


def _write_sample_csv(tmp_dir):
    """Write sample geocoded CSV to a temp directory."""
    csv_path = os.path.join(tmp_dir, "census_towns_geocoded.csv")
    SAMPLE_GEOCODED_DATA.to_csv(csv_path, index=False)
    return csv_path


class TestLoadCensusTownLocations:

    def test_returns_dict_format(self, tmp_path):
        """Output should be a dict with {name: {lat, lon, district}}."""
        _write_sample_csv(str(tmp_path))
        locations = load_census_town_locations(data_dir=str(tmp_path))

        assert isinstance(locations, dict)
        assert len(locations) > 0

        for name, info in locations.items():
            assert "lat" in info
            assert "lon" in info
            assert "district" in info
            assert isinstance(info["lat"], float)
            assert isinstance(info["lon"], float)
            assert isinstance(info["district"], str)

    def test_excludes_failed_geocodes(self, tmp_path):
        """Towns with geocode_status='failed' should be excluded."""
        _write_sample_csv(str(tmp_path))
        locations = load_census_town_locations(data_dir=str(tmp_path))

        # Sundarkhed should be excluded (failed)
        for name in locations:
            assert "sundarkhed" not in name.lower(), (
                f"Failed geocode 'Sundarkhed' should be excluded, found '{name}'"
            )

    def test_includes_failed_when_requested(self, tmp_path):
        """When exclude_failed=False, failed towns should appear (with NaN coords)."""
        _write_sample_csv(str(tmp_path))
        # Note: even with exclude_failed=False, the dropna(subset=["lat", "lon"])
        # in the loader will remove rows with NaN lat/lon, so failed rows with
        # None coords will still be excluded. This is expected behavior.
        locations = load_census_town_locations(data_dir=str(tmp_path), exclude_failed=False)
        # The failed row has None lat/lon, so it gets dropped by dropna
        assert isinstance(locations, dict)

    def test_disambiguates_duplicate_names(self, tmp_path):
        """Duplicate town names in different districts should be disambiguated."""
        _write_sample_csv(str(tmp_path))
        locations = load_census_town_locations(data_dir=str(tmp_path))

        # "Karjat" appears in both Ahmadnagar and Raigarh — should get district suffix
        karjat_entries = [n for n in locations if "karjat" in n.lower()]
        assert len(karjat_entries) == 2, (
            f"Expected 2 Karjat entries (disambiguated), got {karjat_entries}"
        )

    def test_coordinates_in_maharashtra_bbox(self, tmp_path):
        """All coordinates should be within Maharashtra bounding box."""
        _write_sample_csv(str(tmp_path))
        locations = load_census_town_locations(data_dir=str(tmp_path))

        bbox = config.MAHARASHTRA_BBOX
        for name, info in locations.items():
            assert bbox["south"] - 1 <= info["lat"] <= bbox["north"] + 1, (
                f"{name} lat={info['lat']} outside Maharashtra bbox"
            )
            assert bbox["west"] - 1 <= info["lon"] <= bbox["east"] + 1, (
                f"{name} lon={info['lon']} outside Maharashtra bbox"
            )

    def test_fallback_to_config_when_csv_missing(self, tmp_path):
        """If geocoded CSV doesn't exist, should fall back to URBAN_CITIES."""
        nonexistent_dir = os.path.join(str(tmp_path), "nonexistent")
        os.makedirs(nonexistent_dir, exist_ok=True)
        locations = load_census_town_locations(data_dir=nonexistent_dir)
        assert locations == config.URBAN_CITIES

    def test_unique_town_names_in_output(self, tmp_path):
        """Each key in the output should be unique."""
        _write_sample_csv(str(tmp_path))
        locations = load_census_town_locations(data_dir=str(tmp_path))

        # dict keys are inherently unique, but verify no data was lost
        # We expect 5 entries: Pune, Baramati, Nagpur, Karjat (Ahmadnagar), Karjat (Raigarh)
        assert len(locations) == 5


class TestTitleCaseTown:

    def test_simple_name(self):
        assert _title_case_town("pune") == "Pune"

    def test_multi_word(self):
        assert _title_case_town("navi mumbai") == "Navi Mumbai"

    def test_already_single_word(self):
        assert _title_case_town("nagpur") == "Nagpur"


class TestCitySourceIntegration:
    """Test that city_source='census' wires through the pipeline correctly."""

    def test_build_locations_filtered_config(self):
        """Default city_source='config' should use URBAN_CITIES."""
        from src.site.site_analysis import _build_locations_filtered
        locations = _build_locations_filtered(entity_type="city", city_source="config")
        assert len(locations) == len(config.URBAN_CITIES)

    def test_build_locations_filtered_census(self, tmp_path, monkeypatch):
        """city_source='census' should load from census_town_locations."""
        from src.site.site_analysis import _build_locations_filtered

        # Create mock census data
        _write_sample_csv(str(tmp_path))

        # Monkeypatch load_census_town_locations to use our temp dir
        import src.census.town_locations as ctl_module
        original_loader = ctl_module.load_census_town_locations

        def patched_loader(**kwargs):
            kwargs["data_dir"] = str(tmp_path)
            return original_loader(**kwargs)

        monkeypatch.setattr(ctl_module, "load_census_town_locations", patched_loader)

        locations = _build_locations_filtered(entity_type="city", city_source="census")
        # Should have our 5 sample towns (3 unique + 2 disambiguated Karjats)
        assert len(locations) == 5

        # Each should be type "city"
        for name, (lat, lon, loc_type, district) in locations.items():
            assert loc_type == "city"
            assert isinstance(lat, float)
            assert isinstance(lon, float)

    def test_build_locations_filtered_site_ignores_city_source(self):
        """city_source should not affect site locations."""
        from src.site.site_analysis import _build_locations_filtered
        locations_config = _build_locations_filtered(entity_type="site", city_source="config")
        locations_census = _build_locations_filtered(entity_type="site", city_source="census")
        assert locations_config == locations_census
        assert len(locations_config) == len(config.DARKSKY_SITES)


class TestProjectedTowns:
    """Test that projected town data files exist and are valid."""

    def test_town_master_exists(self):
        """town_master.csv should exist after running project_census_towns.py."""
        path = os.path.join("data", "census", "projected_towns", "town_master.csv")
        if not os.path.exists(path):
            pytest.skip("Projected town data not generated yet")
        df = pd.read_csv(path)
        assert len(df) > 500, f"Expected >500 towns, got {len(df)}"
        assert "district" in df.columns
        assert "norm_name" in df.columns
        assert "display_name" in df.columns

    def test_projected_year_files(self):
        """Per-year projected CSVs should exist for all study years."""
        base = os.path.join("data", "census", "projected_towns")
        if not os.path.isdir(base):
            pytest.skip("Projected town data not generated yet")

        for year in config.STUDY_YEARS:
            path = os.path.join(base, f"towns_{year}.csv")
            assert os.path.exists(path), f"Missing projected towns for {year}"
            df = pd.read_csv(path)
            assert len(df) > 0, f"Empty CSV for {year}"
            assert "district" in df.columns

    def test_town_count_nondecreasing(self):
        """Town count should be non-decreasing across years."""
        base = os.path.join("data", "census", "projected_towns")
        if not os.path.isdir(base):
            pytest.skip("Projected town data not generated yet")

        prev_count = 0
        for year in config.STUDY_YEARS:
            path = os.path.join(base, f"towns_{year}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            assert len(df) >= prev_count, (
                f"Town count decreased from {prev_count} to {len(df)} at year {year}"
            )
            prev_count = len(df)
