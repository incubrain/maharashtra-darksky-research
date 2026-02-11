# Plan: Centralize all formulas into `src/formulas/` subpackage

## Architecture

Create `src/formulas/` — a domain-grouped subpackage for all scientific constants and shared computational functions. `config.py` stays unchanged (params, locations, paths). Each module that currently defines constants/formulas inline will import from `src/formulas/` instead.

## New files

```
src/formulas/
  __init__.py               # Re-exports all public names
  classification.py         # classify_alan(), classify_alan_series(), classify_stability(), TIER_COLORS
  trend.py                  # fit_log_linear_trend(years, radiance) — shared implementation
  sky_brightness.py         # NATURAL_SKY_BRIGHTNESS, RADIANCE_TO_MCD, REFERENCE_MCD, BORTLE_THRESHOLDS
  spatial.py                # EARTH_RADIUS_KM, DIRECTION_ANGLES
  ecology.py                # LAND_COVER_CLASSES, ECOLOGICAL_SENSITIVITY
  benchmarks.py             # PUBLISHED_BENCHMARKS, BENCHMARK_INTERPRETATION_THRESHOLD
  diagnostics_thresholds.py # OUTLIER_Z_THRESHOLD, DW_WARNING_BOUNDS, JB_P_THRESHOLD, COOKS_D_THRESHOLD, R_SQUARED_WARNING, CV thresholds
  fitting.py                # LIGHT_DOME_FIT_PARAMS, LIGHT_DOME_P0_DEFAULTS
  quality.py                # LIT_MASK_THRESHOLD, CF_CVG_RANGE
```

## Existing file migrations

### Eliminate duplicated functions (highest value):

1. **viirs_process.py** — Replace inline `fit_log_linear_trend()` (~60 lines) with thin wrapper calling `src.formulas.trend.fit_log_linear_trend()`. Replace ALAN if/elif with `classify_alan()`.

2. **site_analysis.py** — Replace duplicated trend fitting loop body with call to shared `fit_log_linear_trend()`. Replace all 3 ALAN classification instances with `classify_alan()` / `classify_alan_series()`.

3. **stability_metrics.py** — Replace inline CV classification with `classify_stability()`.

### Migrate constants (import swap):

4. **sky_brightness_model.py** — Remove 4 constants + BORTLE_THRESHOLDS dict, import from `formulas.sky_brightness`. Functions stay.
5. **proximity_analysis.py** — Replace local `R = 6371.0` with `EARTH_RADIUS_KM`.
6. **directional_analysis.py** — Replace inline directions dict with `DIRECTION_ANGLES`.
7. **ecological_overlay.py** — Replace inline land cover classes/weights.
8. **benchmark_comparison.py** — Replace inline published benchmarks dict + interpretation threshold.
9. **trend_diagnostics.py** — Replace hardcoded thresholds (|z|>2, DW bounds, JB p, Cook's D, R²).
10. **light_dome_modeling.py** — Replace hardcoded curve-fit params.
11. **graduated_classification.py** — Replace inline tier colors.
12. **download_viirs.py** — Replace hardcoded lit_mask threshold 0.5 and cf_cvg range 0-30.

### Tests:

13. **test_alan_classification.py** — Update to import/test shared `classify_alan()` and `classify_alan_series()`.
14. **test_trend_model.py** — Continues working via wrapper; add direct test of shared function.
15. **test_sky_brightness.py** — Update imports to use `src.formulas.sky_brightness` constants.
16. New **test_formulas.py** — Test shared classification and trend functions directly.

### Backwards compatibility:
- Each migrated module re-exports imported names at module level (e.g. `from src.formulas.sky_brightness import BORTLE_THRESHOLDS` in sky_brightness_model.py means existing `from src.sky_brightness_model import BORTLE_THRESHOLDS` still works).
- `URBAN_BENCHMARKS = URBAN_CITIES` alias pattern already established.

## Implementation order:
1. Create all `src/formulas/` files (pure additions, zero risk)
2. Migrate viirs_process.py and site_analysis.py (shared trend + classification)
3. Migrate remaining modules one at a time (constant import swaps)
4. Update tests
5. Run full test suite
