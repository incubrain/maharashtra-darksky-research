# Research Paper Review: Codebase vs. Published Methodology

**Papers Reviewed:**
1. Elvidge, C.D. et al. (2017). "VIIRS night-time lights." *Int. J. Remote Sensing*, 38(21), 5860-5879.
2. Elvidge, C.D. et al. (2021). "Annual time series of global VIIRS nighttime lights derived from monthly averages: 2012 to 2019." *Remote Sensing*, 13(5), 922.

**Date:** 2026-02-27
**Scope:** Full pipeline audit — quality filtering, annual composites, trend modeling, benchmark comparisons, and lit-mask usage.

---

## Executive Summary

The codebase is broadly well-aligned with the two Elvidge papers, correctly implementing lit-mask filtering, cloud-free coverage thresholds, median-preferred aggregation, and log-linear trend modeling. However, this review identified **5 issues** (2 high-severity, 2 medium-severity, 1 low-severity) and **3 opportunities** for methodological improvement. The most critical finding is that **negative radiance values are not handled anywhere in the main pipeline**, which can silently produce NaN corruption in the log-linear trend model.

---

## Paper 1: Elvidge et al. (2017) — VIIRS Night-Time Lights

### What the paper establishes

This is the foundational paper describing the VIIRS Day/Night Band (DNB) nighttime lights composite products. Key methodological points:

- **Cloud-free coverage (cf_cvg):** The number of cloud-free observations per pixel per compositing period. Pixels with low cf_cvg are "more susceptible to ephemeral light contamination and temporal noise."
- **Background removal:** Uses a Data Range (DR) threshold calculated from 3x3 grid cells, indexed to cloud-cover levels. Areas with low cloud-free coverage get higher DR thresholds to avoid false positives.
- **Outlier removal:** The 12-month median is used to discard high radiance outliers (fires, flares) and low outliers (sensor noise, negative values), "filtering out most fires and isolating the background to a narrow range of radiances under 1 nW/cm²/sr."
- **Negative radiance:** The paper acknowledges that VIIRS DNB can produce negative radiance values due to sensor noise and dark-current subtraction. Background radiance "drops slightly below zero between 10°S and 50°S latitude" with overall range of -0.05 to 1.075 nW/cm²/sr. For their composites, they add a constant of 2 to all values before log-scaling to avoid taking logarithms of negative numbers.
- **Product variants:** `vcmcfg` (viirs cloud mask, config) excludes stray-light-affected data; `vcmslcfg` (stray light corrected) includes corrected stray-light data with more polar coverage but reduced quality.

### Cross-reference with codebase

| Paper Recommendation | Codebase Implementation | Status |
|---|---|---|
| CF_CVG threshold for quality filtering | `CF_COVERAGE_THRESHOLD = 5`, applied as `cf_data >= 5` | **Aligned** — threshold is cited correctly |
| Background removal via DR threshold | Background removal delegated to EOG's pre-computed `lit_mask` layer | **Aligned** — uses the product's own background mask rather than reimplementing DR |
| Exclude stray-light-affected data | Uses `vcmcfg` for 2012-2013, implied `vcmslcfg` for 2014+ via version mapping | **Potential issue** — see Finding #4 |
| Negative radiance handling | Not handled in main pipeline | **GAP** — see Finding #1 |
| 12-month median for outlier removal | Uses EOG's pre-computed `median_masked` layer | **Aligned** — leverages the product's built-in outlier removal |

---

## Paper 2: Elvidge et al. (2021) — Annual Time Series of Global VIIRS Nighttime Lights

### What the paper establishes

This paper describes the V2 annual composite methodology that the codebase uses directly:

- **Monthly-to-annual aggregation:** Rough composites are made monthly, then combined into annual composites. The 12-month median discards outliers (fires, aurora, sensor anomalies).
- **Lit mask:** Derived from the "masked median" — after background and outlier removal, a lit-pixel mask is created and applied to both the average and median annual composites. Background areas are "zeroed out."
- **Masked products:** `median_masked` and `average_masked` have background zeroed out. The lit-mask ensures that only genuinely lit pixels carry non-zero values.
- **Median preferred:** The paper implicitly favors the median as the primary radiance metric because it is robust to right-skewed distributions from urban cores and to high-radiance outliers.
- **Global growth statistics:** The paper reports global lit area increasing at 2.2% per year (2012-2019). This is a **lit area** growth rate (number of lit pixels), not a **radiance** growth rate.
- **Multiyear DR threshold:** A single DR threshold is set per grid cell using a multiyear maximum median and multiyear cloud-cover grid, ensuring consistent lit-pixel detection across all years in the time series.
- **Additional noise filtering:** Grid cells with low average radiance (<0.6 nW/cm²/sr) detected in only 1-2 years out of 8 are zeroed out.

### Cross-reference with codebase

| Paper Recommendation | Codebase Implementation | Status |
|---|---|---|
| Use `lit_mask` to exclude background | `lit_data > 0` filter applied | **Aligned** |
| Use `median_masked` for aggregation | Pipeline reads `median_masked` as primary layer | **Aligned** |
| Background areas zeroed out in product | Pipeline further applies `np.isfinite()` check | **Aligned** |
| Global growth = 2.2% (lit area) | Benchmark stored as `annual_growth_pct: 2.2` | **Potential misinterpretation** — see Finding #3 |
| Multiyear consistent DR threshold | Delegated to EOG product (not reimplemented) | **Aligned** |
| Additional filtering for low-detection pixels | Not implemented | **Minor gap** — see Opportunity #1 |

---

## Findings

### Finding #1 — CRITICAL: Negative Radiance Values Not Handled

**Severity:** High
**Location:** `src/viirs_process.py:apply_quality_filters()`, `src/formulas/trend.py:fit_log_linear_trend()`

**The problem:**

Elvidge et al. (2017) explicitly document that VIIRS DNB background radiance ranges from **-0.05 to 1.075 nW/cm²/sr**, with negative values occurring between 10°S and 50°S latitude. While Maharashtra (15.5°N - 22.1°N) is unlikely to see large negative values, the `median_masked` product can still contain small negative values from sensor noise in low-radiance pixels, especially in early VIIRS years (2012-2013) before stray-light corrections matured.

The quality filtering chain is:
```python
valid_mask = np.isfinite(median_data)        # only removes NaN/Inf
valid_mask &= (lit_data > 0)                 # background filter
valid_mask &= (cf_data >= cf_threshold)      # cloud-free filter
filtered = np.where(valid_mask, median_data, np.nan)
```

Negative finite values pass all three filters. They then flow into:
```python
log_rad = np.log(radiance + log_epsilon)     # log_epsilon = 1e-6
```

If `radiance = -0.05`, then `np.log(-0.05 + 1e-6) = np.log(-0.04999)` → **NaN** (with a numpy RuntimeWarning). This NaN silently propagates into the OLS fit, producing NaN for `annual_pct_change`, `r_squared`, and all downstream metrics.

Elvidge et al. (2017) solve this by adding a constant of 2 to all radiance values before log-scaling. Our `log_epsilon = 1e-6` is far too small to serve this purpose — it only prevents `log(0)`, not `log(negative)`.

**Impact:** For dark-sky sites and rural districts with median radiance near zero, even small negative noise values could silently corrupt trend results.

**Recommendation:**

Option A (minimal fix): Add a clipping step in quality filtering:
```python
filtered = np.where(valid_mask, np.maximum(0, median_data), np.nan)
```

Option B (paper-aligned): Increase `LOG_EPSILON` to a value that absorbs potential negatives (e.g., 0.1, which is the minimum VIIRS detectable radiance), or adopt the Elvidge approach of adding 2.0 before log transformation. Note that changing `LOG_EPSILON` would shift all trend estimates slightly, so this should be done carefully with documentation.

---

### Finding #2 — MEDIUM: `lit_mask` Threshold Inconsistency

**Severity:** Medium
**Location:** `src/formulas/quality.py`, `src/viirs_process.py`, `src/site/site_analysis.py`

**The problem:**

There is an unused constant `LIT_MASK_THRESHOLD = 0.5` defined in `src/formulas/quality.py`, but the actual lit_mask filtering in both `apply_quality_filters()` and `compute_site_metrics()` uses a hardcoded `lit_data > 0`.

The EOG `lit_mask` product is a binary integer raster where 0 = background and values ≥ 1 = lit. However, the threshold of 0.5 in `quality.py` suggests someone may have considered a fractional/probability mask at some point.

**Impact:** The `LIT_MASK_THRESHOLD` constant is dead code that could confuse future developers into thinking the lit_mask is applied with a 0.5 threshold when it actually uses `> 0`.

**Recommendation:** Either:
- Use the constant in the filtering code: `lit_data >= LIT_MASK_THRESHOLD` (if 0.5 is the intended semantic threshold for integer data, this is equivalent to `>= 1` after rounding), or
- Remove `LIT_MASK_THRESHOLD` from `quality.py` and document that lit_mask is binary.

---

### Finding #3 — MEDIUM: Benchmark Growth Rate May Be Misattributed

**Severity:** Medium
**Location:** `src/formulas/benchmarks.py`, `tests/test_research_validation.py`

**The problem:**

The codebase stores the "global_average" benchmark as:
```python
"global_average": {
    "source": "Elvidge et al. (2021)",
    "annual_growth_pct": 2.2,
    ...
}
```

The test docstring says: *"Table 3: Global lit area increased at 2.2% per year (2012-2019)."*

However, the 2.2% figure from the literature appears to originate from **Kyba et al. (2017)**, not Elvidge et al. (2021). Kyba et al. (2017) in *Science Advances* reported: "artificially lit outdoor areas increased by 2.2% annually from 2012 to 2016, with an annual radiance growth rate of 1.8%."

There is an important distinction:
- **2.2%** = lit *area* growth rate (number of newly lit pixels)
- **1.8%** = radiance growth rate (brightness of already-lit pixels)

The codebase pipeline computes **radiance trends** (median radiance per district over time), which should be compared against the **radiance growth rate (1.8%)**, not the lit-area growth rate (2.2%).

**Impact:** Comparing district-level radiance trends against a 2.2% lit-area benchmark overstates expected growth and may lead to incorrect interpretations (e.g., "this district is growing slower than the global average" when it's actually on par).

**Recommendation:**
1. Verify whether Elvidge et al. (2021) Table 3 reports this as a lit-area or radiance growth rate.
2. If it's a lit-area rate, consider adding a separate `radiance_growth_pct` field or renaming the existing field.
3. Add a note distinguishing lit-area growth from radiance growth, as they measure different phenomena.

---

### Finding #4 — LOW: VIIRS Version Mapping May Miss Stray Light Transition

**Severity:** Low
**Location:** `src/config.py:VIIRS_VERSION_MAPPING`, `src/config.py:LAYER_PATTERNS`

**The problem:**

The VIIRS product naming conventions differ between `vcmcfg` (excludes stray-light-affected orbits) and `vcmslcfg` (includes stray-light-corrected data). Elvidge et al. (2017) notes that `vcmslcfg` provides "more data coverage toward the poles, but will be of reduced quality."

The codebase has `LAYER_PATTERNS` that matches filenames containing `avg_rade9h`, `median_masked`, etc., but there is no explicit check or preference for `vcmcfg` vs `vcmslcfg` in the layer identification logic (`identify_layers()`). The version mapping is:
```python
{2012: "v21", 2013: "v21", 2014: "v22", ... 2024: "v22"}
```

This correctly maps the v2.1→v2.2 product transition, but does not address the `vcmcfg`/`vcmslcfg` configuration distinction within each version.

**Impact:** If both `vcmcfg` and `vcmslcfg` files exist in the same year directory, the layer identification may non-deterministically pick one or the other. For Maharashtra's latitude (15-22°N), stray-light contamination is minimal, so the practical impact is low.

**Recommendation:** Add a preference in `identify_layers()` to prefer `vcmcfg` files when both configurations exist, consistent with Elvidge et al. (2017)'s recommendation to exclude stray-light-affected data.

---

### Finding #5 — LOW: Bootstrap CI vs. Analytical Standard Error

**Severity:** Low
**Location:** `src/formulas/trend.py`

**The problem:**

The trend fitting uses bootstrap resampling (1000 iterations) to compute 95% confidence intervals for the annual percent change. While bootstrap CIs are valid, the paper's trend analysis is typically done with analytical (parametric) standard errors from the OLS regression, which are:
1. Much faster to compute (no resampling loop)
2. Exact for normally distributed residuals
3. Already available from statsmodels (`model.conf_int()`)

The bootstrap approach has merit when residual normality is questionable (which can happen with VIIRS data), but the current implementation resamples **year indices with replacement**, which is a **nonparametric bootstrap of the data**. This is appropriate for independent observations, but VIIRS annual composites may have temporal autocorrelation (Durbin-Watson < 2), which the naive bootstrap does not account for.

**Impact:** In the presence of positive temporal autocorrelation (common in light pollution time series), the bootstrap CIs will be **too narrow** because the resampled datasets don't preserve the autocorrelation structure. This could lead to overconfident trend assessments.

**Recommendation:** Consider one of:
- A **block bootstrap** (resample contiguous blocks of years) to preserve temporal structure
- A **residual bootstrap** (resample residuals from the fitted model and add to predicted values)
- Or simply use the OLS analytical standard errors with a Newey-West HAC correction for autocorrelation: `model = sm.OLS(log_rad, X).fit(cov_type='HAC', cov_kwds={'maxlags': 1})`

---

## Opportunities for Improvement

### Opportunity #1: Multiyear Detection Filtering

Elvidge et al. (2021) applies an additional noise filter: "zeroing out grid cells that have low average radiances (<0.6 nW/cm²/sr) and detection in only one or two years out of eight." The codebase does not implement this cross-year consistency check.

For a 13-year study (2012-2024), this could be adapted as: exclude pixels that appear lit in fewer than 3 years AND have average radiance below 0.6 nW/cm²/sr. This would reduce false-positive lit detections in rural areas.

### Opportunity #2: Mean vs. Median Composite Selection

The codebase defaults to `median_masked` as the primary radiance layer, which is well-justified for trend analysis (robust to outliers). However, the EOG product also provides `average_masked`, which Elvidge et al. (2021) treats as a complementary metric.

For spatial analysis (gradient analysis, light dome modeling), the average may be more informative because it captures the full magnitude of urban lighting rather than the central tendency. Consider using `average_masked` for spatial analyses and `median_masked` for temporal trends.

### Opportunity #3: Explicit Product Provenance Tracking

Both papers emphasize the importance of consistent product versions for time-series analysis. The codebase does snapshot `config_snapshot.json` at run time, but does not track:
- Which specific `.tif.gz` file was used for each year
- Whether the file was `vcmcfg` or `vcmslcfg`
- The NOAA EOG product creation timestamp (embedded in the filename like `c202205302300`)

Adding this to the `data_manifest.json` would strengthen reproducibility.

---

## Summary Table

| # | Finding | Severity | Category |
|---|---------|----------|----------|
| 1 | Negative radiance not handled → NaN in log-linear trend | **High** | Data quality |
| 2 | `LIT_MASK_THRESHOLD` constant unused, inconsistent with actual filter | **Medium** | Code hygiene |
| 3 | Global 2.2% benchmark may be lit-area, not radiance growth | **Medium** | Scientific accuracy |
| 4 | No vcmcfg/vcmslcfg preference in layer identification | **Low** | Data quality |
| 5 | Bootstrap CI doesn't account for temporal autocorrelation | **Low** | Statistical methodology |

| # | Opportunity | Category |
|---|-------------|----------|
| 1 | Multiyear detection consistency filter | Data quality |
| 2 | Use average_masked for spatial analyses | Methodology |
| 3 | Track exact product filenames in manifest | Reproducibility |

---

## References

- Elvidge, C.D. et al. (2017). VIIRS night-time lights. *Int. J. Remote Sensing*, 38(21), 5860-5879. DOI: [10.1080/01431161.2017.1342050](https://doi.org/10.1080/01431161.2017.1342050)
- Elvidge, C.D. et al. (2021). Annual time series of global VIIRS nighttime lights derived from monthly averages: 2012 to 2019. *Remote Sensing*, 13(5), 922. DOI: [10.3390/rs13050922](https://doi.org/10.3390/rs13050922)
- Kyba, C.C.M. et al. (2017). Artificially lit surface of Earth at night increasing in radiance and extent. *Science Advances*, 3(11), e1701528.
- Falchi, F. et al. (2016). The new world atlas of artificial night sky brightness. *Science Advances*, 2(6), e1600377.
- Li, X. et al. (2020). A harmonized global nighttime light dataset 1992-2018. *Scientific Data*, 7, 168.
- [NOAA EOG VNL Product Page](https://eogdata.mines.edu/products/vnl/)
- [Google Earth Engine VIIRS DNB V2.2 Catalog](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_ANNUAL_V22)
