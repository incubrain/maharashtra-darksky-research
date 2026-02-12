# Plan: Centralize all formulas into `src/formulas/` subpackage

**Status: COMPLETE** — All migrations done. 231 tests passing.

## Architecture

Created `src/formulas/` — a domain-grouped subpackage for all scientific constants and shared computational functions. `config.py` stays unchanged (params, locations, paths). Each module that previously defined constants/formulas inline now imports from `src/formulas/`.

## Files created

```
src/formulas/
  __init__.py               # Re-exports all public names
  classification.py         # classify_alan(), classify_alan_series(), classify_stability(), TIER_COLORS
  trend.py                  # fit_log_linear_trend(years, radiance) — shared implementation
  sky_brightness.py         # NATURAL_SKY_BRIGHTNESS, RADIANCE_TO_MCD, REFERENCE_MCD, BORTLE_THRESHOLDS
  spatial.py                # EARTH_RADIUS_KM, DIRECTION_DEFINITIONS
  ecology.py                # LAND_COVER_CLASSES, ECOLOGICAL_SENSITIVITY
  benchmarks.py             # PUBLISHED_BENCHMARKS
  diagnostics_thresholds.py # DW_WARNING_BOUNDS, JB_P_THRESHOLD, COOKS_D_THRESHOLD, R_SQUARED_WARNING, CV thresholds
  fitting.py                # EXP_DECAY_BOUNDS, LIGHT_DOME params
  quality.py                # LIT_MASK_THRESHOLD, CF_CVG_VALID_RANGE
```

## Completed migrations

1. **viirs_process.py** — Replaced inline `fit_log_linear_trend()` with thin wrapper calling `src.formulas.trend.fit_log_linear_trend()`. Replaced ALAN if/elif with `classify_alan()`.
2. **site_analysis.py** — Replaced duplicated trend fitting loop body with call to shared `fit_log_linear_trend()`. Replaced all ALAN classification instances with `classify_alan()` / `classify_alan_series()`.
3. **stability_metrics.py** — Replaced inline CV classification with `classify_stability()`.
4. **sky_brightness_model.py** — Removed 4 constants + BORTLE_THRESHOLDS dict, imports from `formulas.sky_brightness`.
5. **proximity_analysis.py** — Replaced local `R = 6371.0` with `EARTH_RADIUS_KM`.
6. **directional_analysis.py** — Replaced inline directions dict with `DIRECTION_DEFINITIONS`.
7. **ecological_overlay.py** — Replaced inline land cover classes/weights.
8. **benchmark_comparison.py** — Replaced inline published benchmarks dict.
9. **trend_diagnostics.py** — Replaced hardcoded thresholds with formulas imports.
10. **light_dome_modeling.py** — Replaced hardcoded curve-fit params.
11. **graduated_classification.py** — Replaced inline tier colors.

## Completed test updates

- **test_alan_classification.py** — Updated to import/test shared `classify_alan()` and `classify_alan_series()`.
- **test_trend_model.py** — Tests shared function via wrapper + direct tests.
- **test_sky_brightness.py** — Updated imports to use `src.formulas.sky_brightness` constants.
- **test_formulas.py** — 67 tests covering classification, trend fitting, and all constants.
- **test_research_validation.py** — Every threshold validated with full research citation.
- **test_edge_cases.py** — Extreme inputs, NaN, empty data, type coercion.
- **test_integration.py** — Stage-to-stage pipeline flow, schema validation.
- **test_regression.py** — Golden file comparison tests.

Backwards compatibility maintained: each migrated module re-exports imported names at module level.
