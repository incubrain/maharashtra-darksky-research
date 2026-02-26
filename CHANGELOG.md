# Changelog

Scientific corrections and methodological improvements to the Maharashtra
VIIRS ALAN analysis pipeline.

## [Unreleased] - 2026-02-27

### Test Coverage Enhancement (DRE-64)

#### 180 New Tests Across 8 Previously Untested Modules
- **`test_proximity_analysis.py`** (31 tests): Haversine distance verified
  against known geodesic references (Mumbai–Pune, equator degree, antipodal
  points), bearing calculations, cardinal direction mapping, triangle
  inequality property, date-line wraparound, `compute_nearest_city_distances()`
  integration with real config sites.
- **`test_stability_metrics.py`** (14 tests): CV computation, zero-mean
  division safety, NaN propagation through dropna, monotonically increasing
  series correctly classified as non-stable, spike detection via
  max_year_to_year_change, custom entity_col support.
- **`test_breakpoint_analysis.py`** (13 tests): Piecewise regression detects
  clear regime changes (±1 year), insufficient data returns None/NaN, step
  functions detected, negative radiance log-domain handling, growth rates
  reported in physical units (%/yr).
- **`test_graduated_classification.py`** (16 tests): Percentile tier
  exhaustiveness (no NaN tiers), monotonicity (Very High > Pristine radiance),
  <5 district minimum enforced, identical radiance degeneracy, temporal
  trajectory year consistency.
- **`test_monuments_integrity.py`** (21 tests): Tuple structure (6 fields per
  entry), no empty names/districts, all types map to TYPE_COLORS/TYPE_MARKERS,
  no duplicate (name, district) pairs, district distribution across 20+
  districts, Maharashtra bbox verification for geocoded sites.
- **`test_correlation_edge_cases.py`** (27 tests): Perfect/zero/negative
  Pearson correlations, n<3 graceful degradation to NaN, NaN pairwise
  dropping, constant array handling, CI contains point estimate, CI narrows
  with sample size, OLS residuals sum to zero, partial correlation covariate
  handling (1-D, DataFrame, NaN rows), correlation matrix method filtering.
- **`test_census_validation.py`** (12 tests): None DataFrame, missing entity
  column, <30 district warning, >20% NaN column warning, zero denominator in
  derived ratios produces NaN, compound ratio (P_SC+P_ST), column prefixing
  preserves entity column.
- **`test_step_runner.py`** (14 tests): Success/error StepResult construction,
  timing recording, arg/kwarg forwarding, output_summary_fn invocation and
  skip-on-None, all 5 default exception types caught, unexpected exceptions
  also caught with traceback, edge cases (empty DataFrame, False return).

#### Brittle Test Removal / Hardening
- **Removed:** `test_expected_site_counts` — hardcoded `== 43` cities and
  `== 11` dark-sky sites; broke when any site was added. Replaced with
  `test_minimum_site_counts` using `>= 40` / `>= 5` floor checks.
- **Removed:** `test_monument_count` exact `== 384` assertion. Replaced with
  `>= 350` minimum ensuring the list hasn't been accidentally truncated.
- **Fixed:** `test_study_years_range` — removed hardcoded `== 2024` and
  `== 13` that would break when VIIRS 2025 data is added. Now checks
  `>= 2012` start, `>= 2024` minimum end, and no gaps.
- **Fixed:** `test_returns_residuals` — `len(residuals) == 13` replaced with
  `len(residuals) == len(years)` (relative to input, not hardcoded).
- **Fixed:** `test_trend_fitting_consumes_yearly_data` — `n_years == 13`
  replaced with `n_years == len(d_data)`.
- **Fixed:** `test_projected_year_files` / `test_town_count_nondecreasing` —
  hardcoded `range(2012, 2025)` replaced with `config.STUDY_YEARS`.
- **Removed:** Duplicate `NATURAL_SKY_BRIGHTNESS == 22.0` assertion in
  `test_formulas.py` (already covered with full Falchi citation in
  `test_research_validation.py`).

## [Unreleased] - 2026-02-21

### Dead Code Removal & Structural Cleanup

- **site_analysis.py** (991 → 814 lines): Removed legacy CLI (`main()`,
  `parse_args()`, `if __name__` block), dead `generate_annual_frames()` (100
  lines, never called), redundant `_build_locations()` helper, unused
  `argparse`/`tempfile`/`viirs_utils` imports.
- **site_analysis.py**: Split 228-line `generate_site_maps()` into 3 focused
  helpers: `_plot_state_overlay()`, `_plot_radiance_chart()`,
  `_plot_radiance_raster()`. Public API unchanged.
- **pipeline_runner.py** (941 → 920 lines): Removed unused `sys`/`StepTimer`
  imports, dead `validate_site_yearly()`/`validate_site_trends()` validators.
  Extracted `parse_years()` helper (was duplicated in 2 places). Extracted
  `_validate_gate()` helper (was duplicated validation pattern). Replaced
  8-branch `elif` dispatch chain in `_run_single_district_step()` with a
  dispatch dict.
- **Census dataset factory**: New `_census_factory.py` with
  `make_district_dataset()` and `make_town_dataset()` factory functions.
  6 thin wrapper modules (census_1991, census_2001, census_2011 + town
  variants) now use the factory instead of duplicating boilerplate.

### Step Boilerplate Elimination

- **New:** `src/step_runner.py` with `run_step()` function — a generic step
  executor that encapsulates the ~45-line boilerplate pattern (StepTimer,
  try/except, log_step_summary, StepResult construction) that was duplicated
  across every pipeline step function.
- Refactored `pipeline_steps.py` (1018 → ~330 lines), `site_pipeline_steps.py`
  (624 → ~280 lines), `cross_dataset_steps.py` (727 → ~380 lines). Each step
  function now contains only the actual work logic.
- `generate_maps.py` and `generate_reports.py` now reuse the same step functions
  used by the pipeline, eliminating duplicated visualization/report logic.
- Net reduction: ~1,600 lines of boilerplate removed.

### Pipeline Consolidation

#### Single Entry Point
- **Breaking:** Removed `main()`, `parse_args()`, `_create_run_dir()` from
  `viirs_process.py`. The only entry point is now
  `python3 -m src.pipeline_runner`.
- `pipeline_runner.py` now creates timestamped run directories
  (`outputs/runs/<timestamp>/`) with `outputs/latest` symlink on every full run.
- Added `--download-shapefiles` flag (moved from the old `viirs_process` CLI).
- Added `--dryrun` flag: prints all planned pipeline steps and exits without
  executing, for easy flow auditing.

#### Dataset Group Aliases
- `--datasets census` now expands to all 8 census datasets (district + town).
- `--datasets census_district` → 4 district-level census datasets.
- `--datasets census_towns` → 4 town-level census datasets.
- Individual dataset names and `all` still work as before.

#### Log Noise Reduced
- DBS "Background Floor" message demoted from INFO to DEBUG (~200 fewer log
  lines per pipeline run).

#### Legacy Code Removed
- Removed `URBAN_BENCHMARKS` alias (use `URBAN_CITIES`).
- Removed `CENSUS_2011_DISTRICT_COLUMNS` / `CENSUS_2011_DERIVED_RATIOS` aliases
  (use `CENSUS_COMMON_COLUMNS` / `CENSUS_COMMON_DERIVED_RATIOS`).
- Removed `OUTPUT_DIRS` dict from config (use `get_entity_dirs()`).
- Removed duplicate `run_full_pipeline()` from `viirs_process.py` (identical
  logic already in `pipeline_runner.run_district_pipeline()`).

### Visualization Pipeline Integration

#### Animation Frames Integrated into District Pipeline
- **New step:** `animation_frames` generates sprawl, differential, and darkness
  frame sequences (one PNG per year) plus a pixel-wise trend map — all previously
  implemented in `src/outputs/visualizations.py` but never called from any pipeline.
- Frames output to `district/maps/frames/{sprawl,differential,darkness}/` for easy
  GIF creation with ImageMagick or ffmpeg.
- `alan_trend_map.png` shows per-pixel linear regression slope (coolwarm colormap).

#### Per-District Radiance Maps
- **New step:** `per_district_radiance_maps` generates a zoomed-in raster radiance
  map for each of the 36 districts, clipped to district boundaries.
- Uses log-scale magma colormap with district boundary overlay.
- Output to `district/maps/districts/<name>_radiance.png`.

#### City/Site Pipelines Integrated into Main Runner
- `--pipeline all` now actually runs city and site pipelines (previously just
  logged a delegation message and skipped execution).
- Added `--buffer-km` and `--city-source` CLI flags to the main pipeline runner.
- `_run_entity_pipeline()` now returns step results for proper provenance tracking.

#### Standalone Map Regeneration Expanded
- `python3 -m src.outputs.generate_maps --type district` now regenerates animation
  frames and per-district radiance maps in addition to existing visualizations.

#### Documentation
- Created `PIPELINE_GUIDE.md`: comprehensive reference for all pipelines, steps,
  outputs, visualization catalog, CLI flags, and source code map.

### Critical Bug Fixes

#### Silent Data Loss from GDAL MemoryFile Corruption
- **Fixed:** `compute_district_stats()` and `compute_site_metrics()` silently
  returned `None`/`0` for all zonal statistics, producing 100% NaN values
  across all 468 yearly radiance records (36 districts × 13 years).
- **Root cause:** `rasterstats.zonal_stats()` called with a `MemoryFile` path
  (`/vsimem/...`) silently failed after prior large raster operations.
  Decompressing and subsetting ~11 GB global VIIRS composites via
  `rasterio.mask.mask()` left GDAL's internal `GDALRasterIOExtraArg` struct
  in a corrupted state, causing subsequent MemoryFile reads to return empty
  results with no exception raised. Exacerbated by a GDAL version mismatch
  between rasterio (3.12.1) and fiona (3.9.2).
- **Fix:** Replaced MemoryFile approach with numpy array + affine transform
  passed directly to `zonal_stats()`, bypassing the GDAL RasterIO path.
- **Impact:** Pipeline now produces 100% valid data — all 468 records with
  non-null radiance values. Previously, every downstream step (trends,
  stability, visualizations) received all-NaN input, causing 3 step failures
  and meaningless outputs.
- **Files:** `src/viirs_process.py`, `src/site/site_analysis.py`

### Debugging & Diagnostics Infrastructure (DRE-62)

#### Pandera Schema Validation
- Added declarative DataFrame schemas (`src/schemas.py`) for validation at
  pipeline boundaries: `YearlyRadianceSchema`, `TrendsSchema`,
  `SiteYearlySchema`, `SiteTrendsSchema`, `StabilitySchema`.
- `validate_schema()` function supports strict (abort) and lenient (warn) modes.
- Pipeline runner validates DataFrames between critical steps.

#### NaN Propagation Tracking
- `track_nan_counts()` in the pipeline runner monitors NaN counts per column
  after each step. Warns when NaN count increases between consecutive steps,
  identifying the column and delta.

#### Structured Logging Enhancements
- Run-level correlation via `run_id` (UUID4 short form) in every log entry.
- `reset_logging()` for test isolation (autouse fixture in conftest.py).
- `LOG_LEVEL` environment variable controls console verbosity.
- Millisecond timestamps in JSON Lines output.

#### Pipeline Types Enhancements
- `StepResult`: added `started_at`, `completed_at`, `nan_summary` fields.
- `PipelineRunResult`: added `git_sha`, `started_at` fields.
- Both types support `from_dict()` deserialization for run comparison.
- `StepStatus` enum: `SUCCESS`, `SKIPPED`, `ERROR`.

#### Consolidated Data Audit Tool
- Replaced ad-hoc `debug_audit_all_years.py`, `debug_deep_dive.py`,
  `debug_stats.py` with formal `src/data_audit.py` module.
- CLI: `python -m src.data_audit --viirs-dir ./outputs --years 2012-2024`
- Outputs: raster statistics CSV, radiance histograms, threshold sensitivity.

#### HTML Diagnostic Report
- Auto-generated at `diagnostics/run_report.html` after each pipeline run.
- Sections: run summary cards, step results table, trend overview bar chart,
  NaN distribution heatmap, anomaly flags.

#### Pipeline Run Comparison Tool
- `src/run_diagnostics.py`: compare two pipeline runs side-by-side.
- CLI: `python -m src.run_diagnostics --run-a <dir1> --run-b <dir2>`
- Reports step status changes, timing regressions, CSV value diffs.

#### New Tests
- `test_property_based.py`: Hypothesis property-based tests for classification
  totality, monotonicity, trend fitting, sky brightness, radiance transforms.
- `test_validation_gates.py`: Pandera schema validation, error paths,
  `validate_schema()` strict/lenient modes, NaN tracking, pipeline types
  deserialization roundtrip.
- Updated `conftest.py` with `logging_reset` autouse fixture, `mock_args`,
  `all_nan_rasters`, `zero_pixel_rasters` fixtures.

#### Configuration
- Added `pyproject.toml` with pytest config (testpaths, custom markers),
  coverage config (branch=true, fail_under=15), pandera warning filters.
- Added `pandera==0.22.1` and `hypothesis==6.130.1` to `requirements.txt`.

## [Unreleased] - 2026-02-20

### Critical Scientific Fixes

#### Natural Sky Brightness Constant Corrected
- **Changed:** `NATURAL_SKY_BRIGHTNESS` from 21.6 to **22.0 mag/arcsec²**.
- **Source:** Falchi et al. (2016), Table S1 — zenith natural sky background
  in a moonless, cloudless night is 174 µcd/m² ≈ 22.0 mag/arcsec².
- **Impact:** All sky brightness conversions (mag/arcsec²), Bortle scale
  classifications, and derived dark-sky viability assessments are now
  calibrated to the correct physical reference. Previously, sites were
  classified ~0.4 mag brighter than reality, inflating Bortle classes.

#### Radial Profile Indentation Bug Fixed
- **Fixed:** `gradient_analysis.py` — the radial profile computation
  loop body (lines 71-112) was indented at the city-loop level rather
  than inside the radius-loop. Only the last radius (50 km) produced data;
  all inner radii were silently overwritten.
- **Impact:** Gradient analysis now correctly captures radiance at each
  concentric ring (5, 10, 20, 30, 50 km), enabling proper light-falloff
  characterization for each site.

#### DBS Removed from Site Metrics
- **Changed:** Dynamic Background Subtraction (DBS) removed from
  `compute_site_metrics()`. Site and city metrics now use **raw radiance**
  with quality filtering (lit_mask > 0, cf_cvg >= threshold).
- **Reason:** DBS computed a single P1.0 percentile floor from the
  entire state raster and applied it uniformly to all sites. This
  over-subtracted from dark sites (e.g., Tadoba: 0.3 nW floor applied
  to a 0.5 nW signal) and under-subtracted from bright cities.
- **DBS retained for:** Visualization paths only (gradient analysis
  annual map frames) where relative comparison is the goal.

### Methodological Improvements

#### Land Boundary Clipping for Coastal Sites
- Buffers for coastal sites (Mumbai, Ratnagiri) are now clipped to the
  Maharashtra land boundary (union of district polygons) to exclude
  ocean pixels. Mumbai's buffer reduced from 314 km² to 177 km² (56% land),
  preventing near-zero ocean pixels from diluting radiance statistics.
- VIIRS `lit_mask` already classifies water bodies (creeks, rivers, harbor)
  as `lit_mask=0`, so inland water is correctly excluded without additional
  masking.

#### Quality Gate for Low-Data Sites
- `fit_site_trends()` now flags sites with median quality_pct < 5% or
  median valid_pixels < 30 as `quality_flag='low_quality'`. Trend
  statistics are still computed but marked as unreliable.
- Prevents Tadoba-type artifacts where few valid pixels produce
  misleading trend coefficients.

#### Log-Scale Visualization for Multi-Year Maps
- Multi-year comparison grid now uses `LogNorm` color mapping instead of
  linear, revealing variation across the full dynamic range (dark rural
  districts were previously invisible against bright urban ones).

#### Adaptive Growth Classification Thresholds
- Growth classification map now uses quartile-based adaptive bins
  computed from the data distribution, instead of hardcoded fixed
  thresholds (0%, 2%, 5%). When all districts show positive growth
  (as in Maharashtra 2012-2024), fixed bins produced a map where
  every district was "Rapid" — the adaptive scheme always shows
  meaningful spatial variation.

### Code Quality

#### Tests Now Call Production Functions
- `test_alan_classification.py`: Removed reimplemented `classify()` and
  `classify_series()` methods that duplicated production logic. Tests now
  call `classify_alan()` and `classify_alan_series()` directly.
- `test_regression.py`: Sky brightness golden-file test now calls
  `radiance_to_sky_brightness()` instead of reimplementing the Falchi
  formula inline.
- `test_research_validation.py`: Calibration cross-check now calls
  `radiance_to_sky_brightness()` instead of using a simplified formula
  variant that produced different results from production code.

#### City/Site Pipeline Separation
- City and dark-sky site pipelines are now fully independent — no
  cross-type comparisons (boxplot, bar chart) are generated.
- City output files use `city_` prefix (e.g., `city_yearly_radiance.csv`,
  `city_trends.csv`); site files use `site_` prefix. Previously both
  used `site_` which was confusing.
- Removed `create_city_vs_site_boxplot()` and `step_site_visualizations()`.

### Known Behaviour (Not Bugs)

#### Rising DBS Background Floor
The P1.0 background floor increases over the study period (0.10 nW in
2012 to 0.49 nW in 2024). This is a characteristic of the VIIRS VNL
annual composite product, not a pipeline issue:
1. NOAA EOG background masking evolution — later composites retain more
   dim pixels.
2. Stray-light correction changes: vcmcfg (2012-2013) vs vcmslcfg (2014+).
3. NPP to NOAA-20 satellite transition (2018+).

#### Universal 2016 Breakpoint
34 of 36 districts show a structural breakpoint at 2016. This is
consistent across all independent analyses and attributable to the
vcmcfg-to-vcmslcfg composite transition, not a real change in ALAN.
