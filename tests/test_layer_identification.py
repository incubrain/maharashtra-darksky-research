"""
Tests for layer identification from filenames (NOAA and test data naming).

Verifies that identify_layers correctly classifies:
1. NOAA-style filenames (dot-delimited)
2. Test data filenames (underscore-delimited)
3. Priority order (median_masked before average)
"""

import pytest

from src.viirs_process import identify_layers


class TestIdentifyLayers:

    def test_noaa_naming_convention(self):
        """NOAA-style filenames should be classified correctly."""
        paths = [
            "/data/VNL_v22_npp_2024.median_masked.dat.tif",
            "/data/VNL_v22_npp_2024.avg_rade9h.dat.tif",
            "/data/VNL_v22_npp_2024.cf_cvg.dat.tif",
            "/data/VNL_v22_npp_2024.lit_mask.dat.tif",
        ]
        layers = identify_layers(paths)

        assert "median" in layers
        assert "average" in layers
        assert "cf_cvg" in layers
        assert "lit_mask" in layers

    def test_test_data_naming_convention(self):
        """Test data filenames (underscore style) should be classified correctly."""
        paths = [
            "/viirs/2024/median_masked_2024.tif",
            "/viirs/2024/average_masked_2024.tif",
            "/viirs/2024/cf_cvg_2024.tif",
            "/viirs/2024/lit_mask_2024.tif",
        ]
        layers = identify_layers(paths)

        assert "median" in layers
        assert "average" in layers
        assert "cf_cvg" in layers
        assert "lit_mask" in layers

    def test_partial_layers(self):
        """Should work with only median + cf_cvg (minimum for pipeline)."""
        paths = [
            "/viirs/2024/median_masked_2024.tif",
            "/viirs/2024/cf_cvg_2024.tif",
        ]
        layers = identify_layers(paths)

        assert "median" in layers
        assert "cf_cvg" in layers
        assert "average" not in layers
        assert "lit_mask" not in layers

    def test_empty_input(self):
        """Empty list should return empty dict."""
        assert identify_layers([]) == {}

    def test_unknown_files_ignored(self):
        """Files that don't match any pattern should be skipped."""
        paths = ["/data/random_file.tif", "/data/some_other.dat.tif"]
        layers = identify_layers(paths)
        assert len(layers) == 0

    def test_gz_files_also_match(self):
        """Compressed .gz filenames should also be classified (used for gz_layers)."""
        paths = [
            "/viirs/2024/median_masked_2024.tif.gz",
            "/viirs/2024/cf_cvg_2024.tif.gz",
        ]
        layers = identify_layers(paths)

        assert "median" in layers
        assert "cf_cvg" in layers
