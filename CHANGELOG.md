# Changelog

Scientific corrections and methodological improvements to the Maharashtra
VIIRS ALAN analysis pipeline.

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
