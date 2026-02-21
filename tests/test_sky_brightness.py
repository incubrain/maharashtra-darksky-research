"""
Tests for sky brightness conversion (radiance → mag/arcsec² → Bortle scale).

Verifies that:
1. Known radiance values produce expected sky brightness
2. Zero radiance returns natural sky background (~22.0 mag)
3. Higher radiance = lower mag (brighter sky)
4. Bortle classification boundaries are correct
5. The conversion is monotonic
"""

import numpy as np
import pytest

from src.analysis.sky_brightness_model import (
    radiance_to_sky_brightness,
    classify_bortle,
    NATURAL_SKY_BRIGHTNESS,
    RADIANCE_TO_MCD,
    REFERENCE_MCD,
)


class TestRadianceToSkyBrightness:

    def test_zero_radiance_returns_natural_sky(self):
        """Zero artificial radiance should return approximately natural sky brightness."""
        mag = radiance_to_sky_brightness(0.0)
        assert mag == pytest.approx(NATURAL_SKY_BRIGHTNESS, abs=0.1), (
            f"Zero radiance should give ~{NATURAL_SKY_BRIGHTNESS} mag, got {mag:.2f}"
        )

    def test_higher_radiance_means_brighter_sky(self):
        """More radiance = lower mag/arcsec² (brighter sky)."""
        mag_low = radiance_to_sky_brightness(0.5)
        mag_high = radiance_to_sky_brightness(50.0)
        assert mag_high < mag_low, (
            f"Higher radiance should give lower (brighter) mag: "
            f"0.5 nW→{mag_low:.2f}, 50 nW→{mag_high:.2f}"
        )

    def test_monotonic_decrease(self):
        """Sky brightness (mag) should decrease monotonically with increasing radiance."""
        radiances = np.array([0, 0.1, 0.5, 1, 5, 10, 50, 100])
        mags = radiance_to_sky_brightness(radiances)
        for i in range(1, len(mags)):
            assert mags[i] <= mags[i - 1], (
                f"Not monotonic at radiance={radiances[i]}: "
                f"mag={mags[i]:.2f} > prev={mags[i-1]:.2f}"
            )

    def test_array_input(self):
        """Should accept and return arrays."""
        radiances = np.array([0.0, 1.0, 10.0])
        mags = radiance_to_sky_brightness(radiances)
        assert len(mags) == 3
        assert all(np.isfinite(mags))

    def test_scalar_input(self):
        """Should accept scalar input."""
        mag = radiance_to_sky_brightness(5.0)
        assert np.isfinite(mag)

    def test_typical_dark_site_dimmer_than_city(self):
        """A dark site (0.5 nW) should be dimmer (higher mag) than a city (50 nW)."""
        mag_dark = radiance_to_sky_brightness(0.5)
        mag_city = radiance_to_sky_brightness(50.0)
        assert mag_dark > mag_city

    def test_very_low_radiance_near_natural(self):
        """Very low radiance (0.001 nW) should be close to natural sky brightness."""
        mag = radiance_to_sky_brightness(0.001)
        # Should be within 1 mag of natural sky
        assert abs(mag - NATURAL_SKY_BRIGHTNESS) < 1.0, (
            f"0.001 nW should be near natural sky ({NATURAL_SKY_BRIGHTNESS}), got {mag:.2f}"
        )

    def test_conversion_constants_produce_expected_range(self):
        """Verify the conversion produces physically reasonable results.

        Calibrated against Falchi et al. (2016) World Atlas + SQM ground truth:
        - 1.22 nW/cm²/sr → ~20.9 mag (rural/suburban transition)
        - 50 nW/cm²/sr → ~17-18 mag (city sky)
        """
        mag_zero = radiance_to_sky_brightness(0.0)
        mag_low = radiance_to_sky_brightness(0.5)
        mag_mid = radiance_to_sky_brightness(1.22)
        mag_city = radiance_to_sky_brightness(50.0)

        # Zero radiance → natural sky
        assert mag_zero == pytest.approx(22.0, abs=0.1)
        # Low radiance → still relatively dark (rural sky)
        assert 20.5 < mag_low < 22.0
        # ~1.2 nW → suburban transition (World Atlas/SQM calibration point)
        assert mag_mid == pytest.approx(20.9, abs=0.3)
        # City → bright sky, Bortle 8+
        assert 16.0 < mag_city < 18.5


class TestClassifyBortle:

    def test_dark_site_bortle_1_or_2(self):
        """mag > 21.5 should classify as Bortle 1 or 2."""
        bortle, desc = classify_bortle(21.8)
        assert bortle == 1, f"21.8 mag should be Bortle 1, got {bortle}"
        bortle, desc = classify_bortle(21.6)
        assert bortle == 2, f"21.6 mag should be Bortle 2, got {bortle}"

    def test_city_bortle_8_or_9(self):
        """mag < 18.0 should classify as Bortle 8 or 9."""
        bortle, desc = classify_bortle(16.5)
        assert bortle in (8, 9), f"16.5 mag should be Bortle 8-9, got {bortle}"

    def test_suburban_bortle_5(self):
        """mag ~20.0 should classify as Bortle 5 (suburban)."""
        bortle, desc = classify_bortle(20.0)
        assert bortle == 5, f"20.0 mag should be Bortle 5, got {bortle}"

    def test_all_bortle_classes_reachable(self):
        """Every Bortle class 1-9 should be reachable with appropriate mag values."""
        test_mags = [21.8, 21.6, 21.3, 21.0, 20.0, 19.0, 18.2, 17.5, 15.0]
        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        for mag, expected_bortle in zip(test_mags, expected):
            bortle, _ = classify_bortle(mag)
            assert bortle == expected_bortle, (
                f"mag={mag} expected Bortle {expected_bortle}, got {bortle}"
            )

    def test_bortle_has_description(self):
        """All Bortle classes should return a non-empty description."""
        for bortle_class in range(1, 10):
            mag_val = 22 - bortle_class  # crude mapping
            _, desc = classify_bortle(mag_val)
            assert len(desc) > 0


class TestEndToEnd:

    def test_radiance_to_bortle_pipeline(self):
        """Full pipeline: radiance → mag → Bortle should work end-to-end."""
        # Zero radiance → natural sky (22.0 mag) → Bortle 1
        mag_zero = radiance_to_sky_brightness(0.0)
        bortle_zero, _ = classify_bortle(mag_zero)
        assert bortle_zero == 1, (
            f"Zero radiance should be Bortle 1, got {bortle_zero} (mag={mag_zero:.2f})"
        )

        # Dark site (0.5 nW) → very dark sky, Bortle 2-3
        mag_dark = radiance_to_sky_brightness(0.5)
        bortle_dark, _ = classify_bortle(mag_dark)
        assert 2 <= bortle_dark <= 4, (
            f"0.5 nW should be Bortle 2-4, got {bortle_dark} (mag={mag_dark:.2f})"
        )

        # City radiance (50 nW) → Bortle 8+
        mag_city = radiance_to_sky_brightness(50.0)
        bortle_city, _ = classify_bortle(mag_city)
        assert bortle_city >= 8, (
            f"50 nW should be Bortle ≥8, got {bortle_city} (mag={mag_city:.2f})"
        )

    def test_pipeline_produces_valid_bortle_range(self):
        """All radiance values should produce Bortle classes in [1, 9]."""
        for radiance in [0, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            mag = radiance_to_sky_brightness(radiance)
            bortle, desc = classify_bortle(mag)
            assert 1 <= bortle <= 9
            assert len(desc) > 0
