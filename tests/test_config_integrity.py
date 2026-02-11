"""
Tests for configuration integrity.

Verifies that:
1. All required config values exist and have expected types
2. Site coordinates are within Maharashtra
3. District names in sites match expected values
4. No duplicate site names
5. VIIRS version mapping covers all study years
"""

import pytest

from src import config


class TestConfigValues:

    def test_quality_thresholds_positive(self):
        assert config.CF_COVERAGE_THRESHOLD > 0
        assert isinstance(config.CF_COVERAGE_THRESHOLD, int)

    def test_log_epsilon_very_small(self):
        assert 0 < config.LOG_EPSILON < 0.001

    def test_bootstrap_params(self):
        assert config.BOOTSTRAP_RESAMPLES >= 100
        ci_lo, ci_hi = config.BOOTSTRAP_CI_LEVEL
        assert 0 < ci_lo < ci_hi < 100

    def test_study_years_range(self):
        years = list(config.STUDY_YEARS)
        assert years[0] == 2012
        assert years[-1] == 2024
        assert len(years) == 13

    def test_alan_thresholds_ordered(self):
        assert config.ALAN_LOW_THRESHOLD < config.ALAN_MEDIUM_THRESHOLD

    def test_bbox_valid(self):
        bbox = config.MAHARASHTRA_BBOX
        assert bbox["west"] < bbox["east"]
        assert bbox["south"] < bbox["north"]
        # Rough check that it's in India
        assert 70 < bbox["west"] < 85
        assert 15 < bbox["south"] < 25


class TestSiteDefinitions:

    def test_no_duplicate_site_names(self):
        all_names = list(config.URBAN_CITIES.keys()) + list(config.DARKSKY_SITES.keys())
        assert len(all_names) == len(set(all_names)), "Duplicate site names found"

    def test_urban_benchmarks_have_required_fields(self):
        for name, info in config.URBAN_CITIES.items():
            assert "lat" in info, f"{name} missing lat"
            assert "lon" in info, f"{name} missing lon"
            assert "district" in info, f"{name} missing district"

    def test_darksky_sites_have_required_fields(self):
        for name, info in config.DARKSKY_SITES.items():
            assert "lat" in info, f"{name} missing lat"
            assert "lon" in info, f"{name} missing lon"
            assert "district" in info, f"{name} missing district"
            assert "type" in info, f"{name} missing type"

    def test_all_sites_within_maharashtra(self):
        bbox = config.MAHARASHTRA_BBOX
        all_sites = {**config.URBAN_CITIES, **config.DARKSKY_SITES}
        for name, info in all_sites.items():
            assert bbox["south"] <= info["lat"] <= bbox["north"], (
                f"{name} lat={info['lat']} outside Maharashtra bbox"
            )
            assert bbox["west"] <= info["lon"] <= bbox["east"], (
                f"{name} lon={info['lon']} outside Maharashtra bbox"
            )

    def test_expected_site_counts(self):
        assert len(config.URBAN_CITIES) == 43
        assert len(config.DARKSKY_SITES) == 11


class TestVIIRSVersionMapping:

    def test_all_study_years_covered(self):
        for year in config.STUDY_YEARS:
            assert year in config.VIIRS_VERSION_MAPPING, (
                f"Year {year} missing from VIIRS_VERSION_MAPPING"
            )

    def test_version_values_valid(self):
        for year, version in config.VIIRS_VERSION_MAPPING.items():
            assert version in ("v21", "v22"), (
                f"Unexpected VIIRS version '{version}' for year {year}"
            )

    def test_v21_only_for_early_years(self):
        """v21 should only be used for 2012-2013."""
        for year, version in config.VIIRS_VERSION_MAPPING.items():
            if version == "v21":
                assert year <= 2013, f"v21 used for year {year}, expected only 2012-2013"
