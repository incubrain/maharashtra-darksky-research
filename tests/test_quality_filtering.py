"""
Tests for quality filtering logic (lit_mask, cf_cvg, finite checks).

Verifies that apply_quality_filters correctly:
1. Passes all pixels when all masks are valid
2. Excludes pixels where lit_mask = 0
3. Excludes pixels where cf_cvg < threshold
4. Handles NaN/Inf in median data
5. Combines filters correctly (AND logic)
"""

import numpy as np
import pytest

from src.viirs_process import apply_quality_filters


class TestApplyQualityFilters:

    def test_all_valid_pixels_pass(self, uniform_rasters):
        """When all masks are valid, all finite pixels should pass."""
        filtered, meta, transform = apply_quality_filters(
            uniform_rasters["median"],
            lit_mask_path=uniform_rasters["lit_mask"],
            cf_cvg_path=uniform_rasters["cf_cvg"],
            cf_threshold=5,
        )
        valid_count = np.count_nonzero(np.isfinite(filtered))
        total_pixels = filtered.size
        # All pixels should pass (uniform 5.0 radiance, lit=1, cf=10)
        assert valid_count == total_pixels, (
            f"Expected all {total_pixels} pixels to pass, but only {valid_count} did"
        )

    def test_lit_mask_excludes_unlit(self, partial_mask_rasters):
        """Pixels where lit_mask=0 should become NaN in filtered output."""
        filtered, _, _ = apply_quality_filters(
            partial_mask_rasters["median"],
            lit_mask_path=partial_mask_rasters["lit_mask"],
            cf_cvg_path=partial_mask_rasters["cf_cvg"],
            cf_threshold=5,
        )
        valid_count = np.count_nonzero(np.isfinite(filtered))
        total_pixels = filtered.size
        # Only the east half (lit=1) should pass; ~50% of pixels
        assert valid_count < total_pixels
        assert valid_count == pytest.approx(total_pixels / 2, rel=0.05)

    def test_cf_threshold_excludes_low_coverage(self, tmp_dir):
        """Pixels with cf_cvg below threshold should be excluded."""
        from tests.conftest import _write_raster, HEIGHT, WIDTH
        import os

        median_data = np.full((HEIGHT, WIDTH), 5.0, dtype="float32")
        lit_data = np.ones((HEIGHT, WIDTH), dtype="float32")
        # cf_cvg = 3 everywhere, which is below default threshold of 5
        cf_data = np.full((HEIGHT, WIDTH), 3.0, dtype="float32")

        median_path = _write_raster(os.path.join(tmp_dir, "med.tif"), median_data)
        lit_path = _write_raster(os.path.join(tmp_dir, "lit.tif"), lit_data)
        cf_path = _write_raster(os.path.join(tmp_dir, "cf.tif"), cf_data)

        filtered, _, _ = apply_quality_filters(
            median_path, lit_mask_path=lit_path, cf_cvg_path=cf_path,
            cf_threshold=5,
        )
        valid_count = np.count_nonzero(np.isfinite(filtered))
        assert valid_count == 0, "All pixels should fail cf_cvg < 5 filter"

    def test_nan_in_median_excluded(self, tmp_dir):
        """NaN pixels in median raster should be excluded regardless of masks."""
        from tests.conftest import _write_raster, HEIGHT, WIDTH
        import os

        median_data = np.full((HEIGHT, WIDTH), 5.0, dtype="float32")
        # Inject NaN in first 100 rows
        median_data[:100, :] = np.nan
        lit_data = np.ones((HEIGHT, WIDTH), dtype="float32")
        cf_data = np.full((HEIGHT, WIDTH), 10.0, dtype="float32")

        median_path = _write_raster(os.path.join(tmp_dir, "med.tif"), median_data)
        lit_path = _write_raster(os.path.join(tmp_dir, "lit.tif"), lit_data)
        cf_path = _write_raster(os.path.join(tmp_dir, "cf.tif"), cf_data)

        filtered, _, _ = apply_quality_filters(
            median_path, lit_mask_path=lit_path, cf_cvg_path=cf_path,
            cf_threshold=5,
        )
        nan_count = np.count_nonzero(np.isnan(filtered[:100, :]))
        assert nan_count == 100 * WIDTH, "NaN rows should remain NaN after filtering"

    def test_no_mask_files_only_finite_check(self, tmp_dir):
        """When no lit_mask or cf_cvg paths given, only finite check applies."""
        from tests.conftest import _write_raster, HEIGHT, WIDTH
        import os

        median_data = np.full((HEIGHT, WIDTH), 5.0, dtype="float32")
        median_path = _write_raster(os.path.join(tmp_dir, "med.tif"), median_data)

        filtered, _, _ = apply_quality_filters(
            median_path, lit_mask_path=None, cf_cvg_path=None,
        )
        valid_count = np.count_nonzero(np.isfinite(filtered))
        assert valid_count == HEIGHT * WIDTH

    def test_filtered_values_match_original(self, uniform_rasters):
        """Valid pixels should retain their original radiance values."""
        filtered, _, _ = apply_quality_filters(
            uniform_rasters["median"],
            lit_mask_path=uniform_rasters["lit_mask"],
            cf_cvg_path=uniform_rasters["cf_cvg"],
        )
        valid_pixels = filtered[np.isfinite(filtered)]
        assert np.allclose(valid_pixels, 5.0), (
            "Filtered values should match original radiance (5.0)"
        )
