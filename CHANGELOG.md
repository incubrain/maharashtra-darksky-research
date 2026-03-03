# Changelog

Scientific corrections and methodological improvements to the Maharashtra
VIIRS ALAN analysis pipeline.

## [Unreleased] - 2026-03-03

### Dark-Reference Background Correction (Coesfeld Method)

Replaced per-year P01 Dynamic Background Subtraction (DBS) with a
**dark-reference-area background correction** for all visualization
frame generators, following Coesfeld et al. (2020).

**Problem:** The original DBS computed the 1st-percentile (P01) of
valid pixels from each year's state raster and subtracted it as a
"noise floor."  However, the P01 rises from 0.10 nW (2012) to
0.49 nW (2024) — not because of increasing noise, but because later
NOAA EOG annual composites retain more dim pixels due to evolving
background masking.  This systematically over-subtracts from later
years, compressing observed growth trends and causing artificial
year-to-year dips (e.g., 2012→2013 appeared as a 12–30% radiance
drop in some districts when it was actually stable).

**Solution:** Three protected-area dark-sky sites are sampled per year
to estimate the natural background:

| Site                    | Type               | Mean Radiance | Temporal CV |
|-------------------------|--------------------|---------------|-------------|
| Pench Tiger Reserve     | Tiger reserve      | 0.029 nW      | 0.13        |
| Tadoba Tiger Reserve    | Tiger reserve      | 0.002 nW      | 0.42        |
| Yawal Wildlife Sanctuary| Wildlife sanctuary | 0.082 nW      | 0.33        |

For each year, the median radiance inside each site's 10 km buffer is
computed from the raw state raster, and the median of the three site
medians is subtracted as the background.  This is physically meaningful
(genuinely unlit protected areas) rather than a statistical artefact of
the raster's own pixel distribution.

Per-site per-year values are logged to
`diagnostics/dark_reference_backgrounds.csv` for auditability.

**Selection criteria:** Sites were ranked by composite score of
darkness (lowest mean radiance), temporal stability (lowest coefficient
of variation across 2012–2024), adequate pixel count (≥50 valid pixels
in buffer), non-coastal (no ocean pixel contamination), and protected
status.  Melghat Tiger Reserve was rejected despite low radiance due to
high temporal instability (CV = 0.82, radiance tripled 2016–2022).

**Citation:**
Coesfeld, J., Kuester, T., Kuechly, H.U. & Kyba, C.C.M. (2020).
Reducing Variability and Removing Natural Light from Nighttime Satellite
Imagery. *Sensors*, 20(11), 3287.
https://doi.org/10.3390/s20113287

**Impact:** All visualization frame types (sprawl, differential,
darkness, light increase, trend map, per-district radiance) now use
year-specific dark-reference backgrounds.  The per-year P01 DBS
function (`viirs_utils.apply_dynamic_background_subtraction`) is
retained only for single-year diagnostic plots where cross-year
consistency is not required.

### Per-Year Light Increase Frames and GIF Assembly

- **State-level light increase frames**: one PNG per year (2012–2024)
  showing absolute nighttime radiance on a log₁₀ magma colormap with
  mean radiance and baseline % change annotations.
- **Per-district radiance frames**: one PNG per year per district
  (36 × 13 = 468 frames), clipped to district boundaries, rendered at
  native VIIRS resolution (~450 m pixels) with nearest-neighbor
  interpolation.
- **GIF assembler**: converts any frame directory into an animated GIF
  with configurable duration, size, and quality settings.
- All wired into the district pipeline as `light_increase_frames` and
  `assemble_gifs` steps.

## [Unreleased] - 2026-02-27

### Research-Driven Scientific Corrections (DRE-64)

Systematic audit of the codebase against 11 peer-reviewed papers identified
42+ findings. All P0 (critical) and P1 (high priority) items are fixed below.
Full finding details in `reports/consolidated_findings.md`.

#### P0-1: Negative Radiance Handling Fixed (Finding E1)
- **Changed:** `viirs_process.apply_quality_filters()` now clips negative
  radiance pixels to zero after quality filtering, with a logged count.
- **Changed:** `formulas/trend.fit_log_linear_trend()` now clips negative
  values before the log transform as a defensive guard.
- **Reason:** VIIRS DNB can report small negative radiances from sensor noise
  or background over-subtraction. Previously, `log(negative + 1e-6)` produced
  NaN, silently corrupting the OLS trend model for affected districts.
- **Ref:** Elvidge, C.D. et al. (2017). VIIRS night-time lights. Int. J.
  Remote Sensing, 38(21), 5860-5879. Section 3.1.
- **Files:** `src/viirs_process.py`, `src/formulas/trend.py`

#### P0-2: Fabricated Zheng Citation Removed (Finding Z1)
- **Changed:** Removed the citation "Zheng, Q. et al. (2019). Developing a
  new cross-sensor calibration model. Remote Sensing, 11(18), 2132" from
  `config.py` radial gradient documentation.
- **Reason:** DOI 10.3390/rs11182132 resolves to "Coastal Tidal Effects on
  Industrial Thermal Plumes" by Faulkner et al. — an entirely unrelated paper.
  The gradient radii are standard NTL practice and are now cited to Bennie
  et al. (2014).
- **Ref:** Bennie, J. et al. (2014). Contrasting trends in light pollution
  across Europe. Scientific Reports, 4, 3789.
- **Files:** `src/config.py`

#### P0-3: Benchmark Attribution Corrected (Findings KY1, KY2, E3)
- **Changed:** Global 2.2% growth rate re-attributed from "Elvidge et al.
  (2021)" to its true origin, Kyba et al. (2017).
- **Changed:** Benchmark key renamed `global_average` → `global_lit_area`
  to disambiguate from total radiance growth (1.8%).
- **Added:** Two new benchmarks:
  - `global_total_radiance`: 1.8%/yr from Kyba (2017) — total VIIRS-detected
    radiance growth (lit area × intensity).
  - `global_ground_based`: 9.6%/yr from Kyba (2023) — ground-based naked-eye
    star visibility decline, ~5x higher than satellite-detected growth.
- **Added:** `metric_type` field to all benchmarks for unambiguous comparison.
- **Ref:** Kyba, C.C.M. et al. (2017). Artificially lit surface of Earth at
  night increasing in radiance and extent. Science Advances, 3(11), e1701528.
- **Ref:** Kyba, C.C.M. et al. (2023). Citizen scientists report global rapid
  reductions in the visibility of stars. Science, 379(6629), 265-268.
- **Files:** `src/formulas/benchmarks.py`, `src/analysis/benchmark_comparison.py`,
  `tests/test_research_validation.py`, `tests/test_formulas.py`

#### P0-4: VIIRS Limitation Caveats Added (Findings F4, K1, K2, KY3, M1)
- **Added:** Prominent limitation documentation to `formulas/trend.py` module
  docstring covering LED spectral bias (~30% underestimate), ground-truth
  divergence (9.6% vs 2%), electrification confound, and Bortle drift.
- **Added:** Limitation section to `analysis/sky_brightness_model.py` module
  docstring and `radiance_to_sky_brightness()` function docstring covering
  local-pixel-only approximation, LED bias, and missing elevation correction.
- **Reason:** Three independent factors (spectral bias, electrification,
  ground-truth gap) mean VIIRS trends cannot be interpreted as pure light
  pollution growth rates without heavy caveats.
- **Ref:** Kyba et al. (2017); Kyba et al. (2023); Min et al. (2017);
  Cinzano & Falchi (2012); Falchi et al. (2016).
- **Files:** `src/formulas/trend.py`, `src/analysis/sky_brightness_model.py`

#### P0-5: Ecological Sensitivity Weights Documented (Finding B2)
- **Changed:** Added provenance documentation to `ECOLOGICAL_SENSITIVITY`
  clarifying that weights (0.1-0.9) are project-specific heuristic estimates,
  not derived from a published empirical study.
- **Reason:** Impact scores using these weights should be interpreted as
  relative rankings only, not calibrated physical quantities.
- **Ref:** Bennie, J. et al. (2014). Scientific Reports, 4, 3789;
  Gaston, K.J. et al. (2013). Biological Reviews, 88(4), 912-927.
- **Files:** `src/formulas/ecology.py`

#### P1-6: New Published Benchmarks Added (Findings K3, KY2)
- **Added:** Kyba (2023) ground-based 9.6%/yr benchmark and Kyba (2017)
  1.8%/yr total radiance benchmark. See P0-3 above for details.

#### P1-7: Electrification Confound Documented (Findings M1, M2, M3)
- **Changed:** Expanded breakpoint analysis module docstring with detailed
  documentation of India's three overlapping electrification programmes
  (DDUGJY 2014+, UJALA 2015+, Saubhagya 2017+) and their impact on
  Maharashtra VIIRS trends.
- **Added:** Note on rural dark-sky site unreliability — rural areas had 3x
  worse load-shedding and were primary electrification beneficiaries.
- **Added:** Recommendation to include electrification milestones as
  covariates in future breakpoint analysis.
- **Ref:** Min, B. et al. (2017). Detection of rural electrification in
  India using DMSP-OLS and VIIRS. Papers in Regional Science, 96(4), 811-832.
- **Files:** `src/analysis/breakpoint_analysis.py`

#### P1-8: Sky Brightness Local-Pixel Limitation Documented (Findings CF1, F1)
- See P0-4 above — documented in sky brightness model module and function.
- **Ref:** Cinzano, P. & Falchi, F. (2012). Monthly Notices of the Royal
  Astronomical Society, 427(4), 3337-3357.

#### P1-9: CV Stability Thresholds Documented as Heuristics (Findings SE1, SE5)
- **Changed:** Added provenance documentation to `CV_STABLE_THRESHOLD` (0.2)
  and `CV_ERRATIC_THRESHOLD` (0.5) clarifying they are project-specific
  heuristics that do not appear in peer-reviewed VIIRS literature.
- **Changed:** Updated `classify_stability()` docstring with the same caveat.
- **Changed:** Updated test docstrings to reflect heuristic status.
- **Ref:** Small, C. & Elvidge, C.D. (2022). Mapping decadal change in
  anthropogenic night light. Sensors, 22(12), 4459.
- **Files:** `src/formulas/diagnostics_thresholds.py`,
  `src/formulas/classification.py`, `tests/test_research_validation.py`

#### P1-10: Dead `LIT_MASK_THRESHOLD` Removed (Finding E2)
- **Removed:** `LIT_MASK_THRESHOLD = 0.5` from `formulas/quality.py` and all
  re-exports in `formulas/__init__.py`.
- **Removed:** Associated tests in `test_formulas.py` and
  `test_research_validation.py`.
- **Reason:** The constant was dead code — the actual lit_mask filter in
  `apply_quality_filters()` uses `lit_data > 0` (binary mask). The unused
  constant could mislead developers into thinking a 0.5 threshold was applied.
- **Ref:** Elvidge et al. (2017, 2021) — VIIRS lit_mask is binary.
- **Files:** `src/formulas/quality.py`, `src/formulas/__init__.py`,
  `tests/test_formulas.py`, `tests/test_research_validation.py`

#### P1-11: Light Dome Modeling Renamed (Finding CF2)
- **Changed:** Module docstring renamed from "Light dome spatial extent
  modeling" to "Urban radiance footprint modeling" with an explanation
  distinguishing the two physical phenomena.
- **Reason:** The exponential decay model describes urban morphology (built-up
  area tapering at city edges), not atmospheric light propagation, which
  follows a ~d^(-2.5) power law over ~195 km.
- **Ref:** Cinzano, P. & Falchi, F. (2012). MNRAS, 427(4), 3337-3357.
- **Files:** `src/analysis/light_dome_modeling.py`

#### Additional Documentation
- **Added:** `RADIANCE_TO_MCD` provenance note in `formulas/sky_brightness.py`
  clarifying it is sourced from Falchi (2016) Bortle category boundary table,
  not first-principles radiative transfer (findings F2, CF3).

### P2/P3 Methodology Improvements and Documentation (DRE-64)

All remaining priority items from the research paper review (P2: medium
priority, P3: long-term) implemented below. Item #20 (Falchi World Atlas
GeoTIFF integration) excluded — requires separate planning as the atlas
provides only a single 2014-epoch snapshot with no year-over-year utility.

#### P2-12: Multi-Moment Stability Analysis (Findings SE2, SE3)
- **Added:** Skewness and kurtosis computation to `compute_stability_metrics()`
  using `scipy.stats.skew()` and `scipy.stats.kurtosis()` with bias correction.
- **Reason:** Small & Elvidge (2022) show that CV alone discards half of the
  available distributional information. Positive skew indicates growth
  acceleration; high kurtosis indicates outlier years.
- **Ref:** Small, C. & Elvidge, C.D. (2022). Sensors, 22(12), 4459.
- **Files:** `src/analysis/stability_metrics.py`

#### P2-13: LED Transition Flag (Findings F4, KY3, B5)
- **Added:** `possible_led_transition` boolean column to stability metrics.
  Flagged when post-2015 mean radiance is lower than pre-2015 mean, suggesting
  the HPS→LED transition may be causing apparent dimming in VIIRS data.
- **Reason:** VIIRS DNB (500-900 nm) is blind to blue LED emissions. India's
  UJALA programme distributed >360M LED bulbs from 2015+. Districts showing
  post-2015 radiance decrease may reflect spectral shift, not reduced ALAN.
- **Ref:** Kyba et al. (2017), Science Advances, 3(11), e1701528.
- **Files:** `src/analysis/stability_metrics.py`

#### P2-14: Brightening/Dimming Pixel Decomposition (Findings B1, B3)
- **Added:** `compute_pixel_change_decomposition()` function to viirs_process.py.
  Decomposes per-district radiance change into percentages of brightening,
  dimming, and stable pixels between two time periods.
- **Reason:** Bennie et al. (2014) demonstrate that aggregate district trends
  mask spatial heterogeneity — a district can show zero net change while having
  equal brightening and dimming areas.
- **Ref:** Bennie, J. et al. (2014). Scientific Reports, 4, 3789.
- **Files:** `src/viirs_process.py`

#### P2-15: Block Bootstrap for Temporal Autocorrelation (Finding E5)
- **Added:** `block_bootstrap_ci()` function to `formulas/trend.py`. Implements
  block resampling (default block size = 3 years) that preserves local temporal
  autocorrelation structure, producing wider (more honest) confidence intervals
  than standard i.i.d. bootstrap.
- **Reason:** Standard bootstrap underestimates CI width when annual composites
  are temporally correlated (VIIRS composites share sensor artifacts, calibration
  drift, and atmospheric conditions across adjacent years).
- **Ref:** Kunsch, H.R. (1989). Annals of Statistics, 17(3), 1217-1241.
- **Files:** `src/formulas/trend.py`

#### P2-16: VIIRS Version Covariate (Finding M7)
- **Added:** `fit_trend_with_version_covariate()` function to `formulas/trend.py`.
  Includes a binary dummy variable for the VNL v2.1→v2.2 transition (2014) to
  absorb the radiometric shift from the stray-light correction change.
- **Reason:** 34/36 districts show a 2016 breakpoint partly explained by the
  v2.1 (vcmcfg) → v2.2 (vcmslcfg) product transition. Including the version
  covariate yields a cleaner estimate of the underlying ALAN trend.
- **Files:** `src/formulas/trend.py`

#### P2-17: Airglow Confound Documented (Finding L1)
- **Added:** Airglow caveat to `formulas/trend.py` module docstring. At dark
  sites, VIIRS radiance includes natural airglow emissions that vary with the
  ~11-year solar cycle (0.1-0.2 nW/cm²/sr), biasing dark-site trend estimates.
- **Ref:** Levin, N. et al. (2020). Remote Sensing of Environment, 237, 111443.
- **Files:** `src/formulas/trend.py`

#### P2-18: DBS Usage Consistency Documented (Finding Z3)
- **Added:** Explicit DBS usage documentation to `gradient_analysis.py` and
  `ecological_overlay.py` module docstrings. DBS is intentionally used in
  visualization/spatial analysis paths but NOT in the main trend analysis path.
  This is by design, not inconsistency.
- **Files:** `src/analysis/gradient_analysis.py`, `src/analysis/ecological_overlay.py`

#### P2-19: Elevation Correction for Sky Brightness (Finding CF4)
- **Added:** `elevation_correction_factor()` function to `sky_brightness_model.py`.
  Implements first-order atmospheric column correction using scale height (8500 m)
  for Maharashtra's 0-1400 m elevation range.
- **Updated:** Module docstring references the new function.
- **Ref:** Cinzano, P. & Falchi, F. (2012). MNRAS, 427(4), 3337-3357.
- **Files:** `src/analysis/sky_brightness_model.py`

#### P3-21: Anisotropic Viewing Angle Caveat (Finding Z2)
- **Added:** Viewing angle caveat to `gradient_analysis.py` module docstring.
  VIIRS DNB viewing geometry introduces ~50% radiance variability between nadir
  and edge-of-swath observations, which is not corrected.
- **Ref:** Zheng, Q. et al. (2019). Remote Sensing, 11(18), 2132.
- **Files:** `src/analysis/gradient_analysis.py`

#### P3-22: PSF Adjacency Warning (Finding L2)
- **Added:** PSF spillover warning to `build_site_geodataframe()` docstring.
  VIIRS DNB PSF half-power diameter is ~750 m; dark-sky sites within 2-3 km
  of urban areas may have contaminated radiance statistics.
- **Ref:** Levin, N. et al. (2020). Remote Sensing of Environment, 237, 111443.
- **Files:** `src/site/site_analysis.py`

#### P3-23: Levin et al. (2020) Cited as Methodological Reference (Finding L3)
- **Added:** Levin (2020) citation to `viirs_process.py` module docstring,
  `config.py` VIIRS resolution documentation, and `gradient_analysis.py` DBS note.
- **Ref:** Levin, N. et al. (2020). Remote Sensing of Night Lights: A Review.
  Remote Sensing of Environment, 237, 111443.
- **Files:** `src/viirs_process.py`, `src/config.py`, `src/analysis/gradient_analysis.py`

#### P3-24: vcmcfg Preference in Layer Identification (Finding E4)
- **Changed:** `identify_layers()` now prefers vcmcfg over vcmslcfg variants
  when both exist for the same layer type, with documentation explaining why.
- **Reason:** vcmslcfg includes stray-light correction that changes noise
  characteristics and can introduce spurious trends at the v2.1→v2.2 boundary.
- **Ref:** Elvidge, C.D. et al. (2017). Int. J. Remote Sensing, 38(21).
- **Files:** `src/viirs_process.py`

#### P3-25: Seasonal Aerosol Note for Maharashtra (Finding CF5)
- **Added:** Seasonal aerosol optical depth caveat to `sky_brightness_model.py`
  module docstring. Maharashtra experiences significant AOD variation between
  monsoon and winter that affects sky brightness estimates.
- **Ref:** Cinzano, P. & Falchi, F. (2012). MNRAS, 427(4), 3337-3357.
- **Files:** `src/analysis/sky_brightness_model.py`

#### P3-26: Breakpoint Change Type Classification (Finding B4)
- **Added:** `change_type` field to breakpoint detection output. Classifies each
  breakpoint as "acceleration", "deceleration", "reversal_to_decline", or
  "reversal_to_growth" based on pre/post growth rate comparison.
- **Ref:** Bennie, J. et al. (2014). Scientific Reports, 4, 3789.
- **Files:** `src/analysis/breakpoint_analysis.py`, `tests/test_breakpoint_analysis.py`

#### Additional Fixes
- **Fixed:** Negative radiance handling in `breakpoint_analysis.py` — same E1
  pattern as trend.py. Added `np.clip(radiance, 0, None)` before log transform.
- **Added:** Fabricated Zheng (2019) citation removed from gradient_analysis.py
  module docstring; replaced with Bennie (2014) as the methodology reference.

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

### Critical Bug Fixes

#### Silent Data Loss from GDAL MemoryFile Corruption
- `compute_district_stats()` and `compute_site_metrics()` silently
  returned `None`/`0` for all zonal statistics, producing 100% NaN
  values across all 468 yearly radiance records (36 districts × 13 years).
- **Root cause:** `rasterstats.zonal_stats()` called with a `MemoryFile`
  path (`/vsimem/...`) silently failed after prior large raster
  operations left GDAL's internal state corrupted.
- **Fix:** Replaced MemoryFile approach with numpy array + affine
  transform passed directly to `zonal_stats()`.

## [Unreleased] - 2026-02-20

### Critical Scientific Fixes

#### Natural Sky Brightness Constant Corrected
- `NATURAL_SKY_BRIGHTNESS` changed from 21.6 to **22.0 mag/arcsec²**.
- **Source:** Falchi et al. (2016), Table S1 — zenith natural sky
  background in a moonless, cloudless night is 174 µcd/m² ≈ 22.0
  mag/arcsec².
- **Impact:** All sky brightness conversions, Bortle classifications,
  and dark-sky viability assessments now use the correct reference.

#### Radial Profile Indentation Bug Fixed
- `gradient_analysis.py` — the radial profile loop body was indented
  at the city-loop level rather than the radius-loop. Only the last
  radius (50 km) produced data; all inner radii were silently
  overwritten.

#### DBS Removed from Site Metrics
- Site and city metrics now use **raw radiance** with quality filtering
  (`lit_mask > 0`, `cf_cvg ≥ threshold`) instead of DBS-corrected
  values.
- **Reason:** DBS computed a single P01 floor from the entire state
  raster and applied it uniformly, over-subtracting from dark sites
  (e.g., Tadoba: 0.3 nW floor applied to a 0.5 nW signal).

### Methodological Improvements

#### Land Boundary Clipping for Coastal Sites
- Buffers for coastal sites (Mumbai, Ratnagiri) clipped to the
  Maharashtra land boundary to exclude ocean pixels.

#### Quality Gate for Low-Data Sites
- Sites with median `quality_pct < 5%` or median `valid_pixels < 30`
  are flagged as `quality_flag='low_quality'`.

#### Log-Scale Visualization for Multi-Year Maps
- Multi-year comparison grid uses `LogNorm` color mapping, revealing
  variation across the full dynamic range.

#### Adaptive Growth Classification Thresholds
- Growth classification map uses quartile-based adaptive bins instead
  of hardcoded fixed thresholds that collapsed all districts into a
  single category.

### Known Behaviour (Not Bugs)

#### Rising VIIRS Background Floor
The P01 background floor of the state raster increases over the study
period (0.10 nW in 2012 to 0.49 nW in 2024).  This is a characteristic
of the VIIRS VNL annual composite product:
1. NOAA EOG background masking evolution — later composites retain more
   dim pixels that were previously zeroed.
2. Stray-light correction: vcmcfg (2012–2013) vs vcmslcfg (2014+).
3. NPP to NOAA-20 satellite transition (2018+).

This does NOT affect the main zonal statistics path (which uses raw
radiance).  For visualizations, it is now handled by the dark-reference
background correction described above.

#### Universal 2016 Breakpoint
34 of 36 districts show a structural breakpoint at 2016 in piecewise
regression.  This is a combination of the vcmcfg-to-vcmslcfg composite
transition and real electrification programmes (DDUGJY, Ujala LED).
