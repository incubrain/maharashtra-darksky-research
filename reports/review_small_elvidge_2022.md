# Paper Review: Small & Elvidge (2022) vs. Codebase

**Paper:** Small, C. & Elvidge, C.D. (2022). "Statistical moments of VIIRS night-time lights." *Int. J. Remote Sensing*. DOI: [10.1080/01431161.2022.2161857](https://doi.org/10.1080/01431161.2022.2161857)
**Date:** 2026-02-27

---

## Paper Summary

Small & Elvidge (2022) argue that the standard VIIRS nighttime lights product -- which reports only mean radiance -- discards three-quarters of the available statistical information. By computing all four statistical moments (mean, variance, skewness, kurtosis) from nightly VIIRS Day/Night Band temporal profiles spanning 2012-2020, they demonstrate that the higher-order moments reveal fundamentally different patterns of nighttime illumination that the mean alone cannot distinguish.

### Key Methodology

1. **Multi-moment approach:** For each 15 arc-second grid cell, the authors compute four statistical moments from nightly temporal profiles, filtered to exclude cloudy and sunlit observations, with radiance adjustments to reduce view-angle and lunar illuminance effects.

2. **Variance-mean scattergram:** The central analytical tool is the scattergram of variance (second moment) versus mean (first moment). This scattergram reveals distinct data clusters that the authors use to define **five lighting zones**:

   - **Core lighting:** Moderate mean brightness, relatively low variance. These are established, stable urban lighting areas. Temporal profiles show steady mean over time with moderate variance.
   - **Bright and steady:** High mean brightness, low variance. The most stable illumination sources -- dense urban cores with consistent, high-quality lighting.
   - **Dark-erratics:** Low mean, high variance (relative to mean). The zone most affected by detection-limit effects. Boundary trajectory follows variance = 0.5 * sqrt(mean). These include areas at the edges of lit zones, fishing fleets, and ephemeral sources.
   - **Mid-erratics:** Intermediate mean and variance values, between core lighting and bright-erratics. Grid cells may pass through this zone during lighting transitions (e.g., developing areas gaining consistent illumination).
   - **Bright-erratics:** High mean (>70 nW/cm2/sr) and high variance (>560). Defined as grid cells with higher mean and variance relative to core lighting. The erratic brightness fluctuations may trace back to alternating-current flicker, common with outdoor high-pressure sodium and metal halide lamps. Boundary trajectory follows variance = 2.5 * sqrt(mean).

3. **Heteroskedasticity:** A central finding is the pervasive heteroskedasticity of VIIRS nighttime light data -- specifically, a monotonic decrease of temporal variability with increasing mean brightness. Brighter sources are inherently more stable; dimmer sources exhibit greater relative variability.

4. **Skewness and kurtosis:** The third and fourth moments provide additional discrimination. Grid cells can have nearly identical means yet exhibit sharply different skewness and kurtosis values, revealing different temporal behavior (e.g., intermittent lighting vs. steady sources with occasional anomalies).

5. **Physical interpretation of variance:** The VIIRS DNB pixel effective dwell time (2-3 ms) is short enough to capture alternating-current flicker in high-intensity discharge (HID) lamps. LED lighting, by contrast, is nearly flicker-free. Thus, variance patterns partially encode the lighting technology present in a grid cell. LED-conversion programs can shift grid cells from the bright-erratics zone toward bright-and-steady.

### Key Findings

- **Variance is an "equal partner" to the mean** for characterizing temporal behavior of nighttime lights.
- Grid cells with nearly the same mean can exhibit sharply different higher moments.
- The five-zone classification captures physically meaningful categories that a simple CV threshold cannot distinguish.
- Zone transitions track real-world changes (e.g., LED conversion, new development, lighting infrastructure changes).
- The multi-moment approach is recommended for any temporal analysis of VIIRS nighttime lights.

### Companion Work: Small (2021)

The related paper "Spatiotemporal Characterization of VIIRS Night Light" (Frontiers in Remote Sensing, 2021) provides additional context. Key findings include:
- Anthropogenic nighttime light is "remarkably stable" at subannual time scales.
- Almost 90% of spatiotemporal variance is associated with spatial variations in average brightness; only ~2% each for seasonal and interannual temporal variability.
- The variance-mean relationship follows a heteroskedastic pattern where temporal variability decreases monotonically with increasing brightness.
- The coefficient of variation is **not** explicitly recommended; the authors use raw mean vs. standard deviation (or variance) moment spaces instead.

---

## Cross-Reference: Paper Approach vs. Codebase Implementation

| Aspect | Paper (Small & Elvidge 2022) | Codebase Implementation | Gap |
|--------|------------------------------|------------------------|-----|
| **Moments computed** | All 4: mean, variance, skewness, kurtosis | Only mean, std, CV, IQR | Missing skewness and kurtosis |
| **Classification basis** | 2D variance-mean scattergram with 5 zones | 1D CV thresholds (0.2, 0.5) with 3 classes | Single-dimension vs. two-dimension |
| **Zone/class names** | Core lighting, dark-erratics, mid-erratics, bright-erratics, bright-and-steady | Stable, moderate, erratic | Different taxonomy |
| **Threshold derivation** | Empirical cluster boundaries on scattergram; trajectory equations (e.g., var = 0.5*sqrt(mean)) | CV < 0.2, 0.2-0.5, >= 0.5 -- no published citation | Published vs. ad-hoc |
| **Heteroskedasticity** | Explicitly modeled; brightness-dependent variability is a core finding | Not accounted for; CV normalizes by mean but ignores non-linear variance-mean relationship | Significant gap |
| **Physical interpretation** | Variance patterns encode lighting technology (HID flicker vs. LED stability) | No technology interpretation | Missing physical context |
| **Temporal resolution** | Nightly observations filtered for quality | Annual median composites | Different input granularity |
| **Scatter plot** | Variance vs. mean (or sigma vs. mu) | Mean vs. CV (Fig in `stability_metrics.py`) | Different axes, different insight |
| **Trend reliability** | Variance and higher moments flag unreliable trends | R-squared, Durbin-Watson, Cook's D, Jarque-Bera | Complementary but disjoint |

---

## Detailed Findings

### Finding SE1 -- HIGH: CV Thresholds (0.2, 0.5) Have No Published Justification

**Location:** `src/formulas/diagnostics_thresholds.py:35-36`, `src/formulas/classification.py:83-116`

**Problem:** The codebase classifies temporal stability using coefficient of variation thresholds: CV < 0.2 = "stable", 0.2 <= CV < 0.5 = "moderate", CV >= 0.5 = "erratic". These thresholds are stated without citation. An extensive search of the VIIRS nighttime lights literature -- including Small & Elvidge (2022), Small (2021), Elvidge et al. (2017, 2021) -- reveals no published study that defines these specific CV boundaries for nighttime light temporal classification. The thresholds appear to be ad-hoc choices.

Furthermore, Small & Elvidge (2022) do not use CV at all. They operate in variance-mean space, not CV space, because CV (a normalized metric) obscures the heteroskedastic relationship between variance and brightness that is central to their analysis. A dim source with CV=0.3 and a bright source with CV=0.3 have fundamentally different physical interpretations -- the dim source may be a dark-erratic near the detection limit, while the bright source may represent genuine infrastructure-level variability.

**Impact:** The classification labels lack scientific defensibility. Peer reviewers or collaborators may challenge the thresholds as arbitrary. The single-dimension CV approach conflates physically distinct phenomena that the paper shows require two dimensions (mean + variance) to properly separate.

**Recommendation:**
1. Add explicit citation or justification for the 0.2/0.5 CV thresholds in `diagnostics_thresholds.py`. If based on domain expertise, document the rationale.
2. Consider supplementing (not necessarily replacing) CV classification with a variance-mean scattergram approach following Small & Elvidge (2022).
3. At minimum, add a docstring note: "CV thresholds are project-specific heuristics; cf. Small & Elvidge (2022) for a multi-moment approach."

---

### Finding SE2 -- HIGH: Missing Higher-Order Moments (Skewness, Kurtosis)

**Location:** `src/analysis/stability_metrics.py:23-64`

**Problem:** The `compute_stability_metrics()` function computes mean, std, CV, IQR, and max year-to-year change. It does not compute skewness or kurtosis. Small & Elvidge (2022) demonstrate that these higher-order moments contain substantial information content that cannot be recovered from the first two moments alone. Specifically:

- **Skewness** (3rd moment): Reveals asymmetry in the temporal distribution. Positive skew indicates occasional bright outliers (e.g., fires, festivals, construction lighting); negative skew indicates occasional dips (e.g., power outages, cloud contamination residuals). A dark-sky site with high positive skewness may be experiencing periodic light intrusions that the mean does not capture.
- **Kurtosis** (4th moment): Reveals tail heaviness. High kurtosis indicates extreme events in the temporal profile -- critical for dark-sky assessment where even rare bright events can disqualify a site.

Both `scipy.stats.skew()` and `scipy.stats.kurtosis()` are trivial to add alongside the existing computations, and `scipy.stats` is already imported in `trend_diagnostics.py`.

**Impact:** The analysis discards roughly half the information content available from temporal profiles. For dark-sky site assessment specifically, skewness and kurtosis are arguably more important than CV because they capture the extreme events that matter most for astronomical observation quality.

**Recommendation:**
Add to `compute_stability_metrics()`:
```python
from scipy import stats as scipy_stats
skewness = scipy_stats.skew(values)
kurt = scipy_stats.kurtosis(values)  # excess kurtosis (Fisher definition)
```
Include these in the output DataFrame and stability reports. Add interpretation guidelines: positive skewness suggests episodic bright contamination; high kurtosis suggests extreme events requiring investigation.

---

### Finding SE3 -- MEDIUM: Stability Classification Uses Wrong Dimensionality

**Location:** `src/formulas/classification.py:83-116`, `src/analysis/stability_metrics.py:75-108`

**Problem:** The codebase classifies stability along a single axis (CV) and the scatter plot (`plot_stability_scatter`) shows mean vs. CV. Small & Elvidge (2022) demonstrate that the physically meaningful classification requires **two independent dimensions**: mean brightness and variance (not CV, which is variance/mean and thus collapses the two dimensions).

The five-zone classification in the paper captures phenomena that a CV-only scheme fundamentally cannot:
- **Dark-erratics** (low mean, high variance) appear as high-CV, but their variability is driven by detection-limit noise, not genuine instability.
- **Bright-erratics** (high mean, high variance) appear as moderate-CV due to normalization, masking that their absolute variance is enormous and physically significant (HID lamp flicker).
- **Core lighting** and **bright-and-steady** may have similar CV values but represent fundamentally different stability regimes.

**Impact:** A dark-sky candidate site near the detection limit could be classified as "erratic" (high CV) when it is actually a stable dark site with normal sensor-noise-driven variability. Conversely, a brightly lit area with HID lighting and large absolute variance could be classified as "moderate" due to CV normalization.

**Recommendation:**
1. Add a variance-mean scatter plot (variance on y-axis, mean on x-axis, both log-scaled) alongside the existing CV scatter plot in `plot_stability_scatter()`.
2. Consider implementing the five-zone classification as an additional classification scheme, using the published boundary trajectories (var = 0.5*sqrt(mean) for dark-erratic boundary, var = 2.5*sqrt(mean) for bright-erratic boundary).
3. At minimum, include mean radiance as a second classification dimension: a site with CV=0.6 but mean=0.1 nW/cm2/sr is very different from one with CV=0.6 and mean=50 nW/cm2/sr.

---

### Finding SE4 -- MEDIUM: Heteroskedasticity Not Acknowledged in Trend Diagnostics

**Location:** `src/analysis/trend_diagnostics.py:31-106`

**Problem:** The trend diagnostics module fits a log-linear model and checks standard regression diagnostics (R-squared, Durbin-Watson, Jarque-Bera, Cook's D). However, it does not account for the **pervasive heteroskedasticity** that Small & Elvidge (2022) and Small (2021) identify as a fundamental characteristic of VIIRS nighttime light data.

Specifically, the monotonic decrease of temporal variability with increasing mean brightness means that:
- Trend models for dim areas will have larger residual variance than models for bright areas -- this is expected, not pathological.
- Standard OLS confidence intervals will be inefficient (too wide for bright areas, too narrow for dim areas).
- The Jarque-Bera normality test and residual diagnostics may flag normal heteroskedastic behavior as problematic.

The log transform (`np.log(radiance + epsilon)`) partially stabilizes variance, but the paper shows the heteroskedasticity persists even in transformed data because the variance-mean relationship is not purely multiplicative.

**Impact:** Trend diagnostics may produce false warnings for dim areas (whose natural variability is higher) and false confidence for bright areas. Dark-sky sites -- which are by definition dim -- are most affected.

**Recommendation:**
1. Add a note in `diagnostics_thresholds.py` acknowledging brightness-dependent heteroskedasticity: "Residual variance naturally increases for dimmer areas (Small & Elvidge 2022). Diagnostic warnings for low-radiance sites should be interpreted with caution."
2. Consider using weighted least squares (WLS) or heteroskedasticity-consistent standard errors (e.g., `HC3` in statsmodels) for trend models, especially for dark-sky sites.
3. Stratify diagnostic thresholds by brightness level if feasible.

---

### Finding SE5 -- MEDIUM: "Stable/Moderate/Erratic" Labels Do Not Match Published Taxonomy

**Location:** `src/formulas/classification.py:83-116`, `src/formulas/diagnostics_thresholds.py:33-36`

**Problem:** The codebase uses a three-class scheme: "stable", "moderate", "erratic". The published literature (Small & Elvidge 2022) uses a five-class scheme: "core lighting", "bright and steady", "dark-erratics", "mid-erratics", "bright-erratics". These are not interchangeable:

- The paper's "stable" categories are split into "core lighting" (moderate brightness, moderate variance) and "bright and steady" (high brightness, low variance) -- these are physically distinct.
- The paper's "erratic" categories are split into three zones by brightness: dark-erratics (sensor noise dominated), mid-erratics (transitional), and bright-erratics (flicker dominated).
- The codebase's "moderate" class has no clear counterpart in the published taxonomy.

Using non-standard terminology may confuse readers familiar with the Elvidge/Small literature and makes cross-study comparison difficult.

**Impact:** The analysis outputs use terminology that does not align with the established literature, reducing comparability and potentially causing confusion in peer review.

**Recommendation:**
1. Consider renaming classes or adding a mapping to the published taxonomy in documentation.
2. At minimum, add a comment in `classification.py` referencing the Small & Elvidge (2022) five-zone scheme and explaining why the simplified three-class approach was chosen for this study.
3. Include in output reports: "Stability classification follows a simplified three-class CV scheme; see Small & Elvidge (2022) for the five-zone variance-mean approach."

---

### Finding SE6 -- LOW: Scatter Plot Axes Differ from Published Convention

**Location:** `src/analysis/stability_metrics.py:75-108`

**Problem:** The `plot_stability_scatter()` function plots mean radiance (x-axis, log scale) vs. coefficient of variation (y-axis). Small & Elvidge (2022) use variance (or standard deviation) vs. mean as their canonical scatter plot. The paper shows that plotting in variance-mean space reveals the five distinct clusters and their boundary trajectories, which are obscured when variance is normalized to CV.

The current plot is useful for its purpose but represents a different analytical perspective from the published methodology.

**Impact:** The scatter plot cannot be directly compared with published figures from the reference literature. The five-zone clustering pattern is not visible in CV space.

**Recommendation:**
Add an additional scatter plot function that plots variance (or std) vs. mean in log-log space, following the convention in Small & Elvidge (2022). This would complement (not replace) the existing CV plot and allow direct comparison with published results.

---

### Finding SE7 -- LOW: IQR Is Computed but Not Leveraged

**Location:** `src/analysis/stability_metrics.py:48-49`

**Problem:** The IQR (interquartile range) is computed and stored in the output CSV but is not used in any classification, diagnostic, or interpretive capacity. Small & Elvidge (2022) do not specifically recommend IQR, but it has value as a robust measure of dispersion that is resistant to the outliers that the paper shows are common in nighttime light temporal profiles.

**Impact:** Minor -- the metric is computed but essentially unused. However, IQR could serve as a valuable complement to CV for identifying outlier-driven variability vs. genuine temporal change.

**Recommendation:**
Consider using IQR as a robustness check against CV. If CV is high but IQR is low, the variability is driven by a few extreme observations (consistent with the paper's "erratic" categories driven by occasional events). If both are high, the variability is genuinely distributed across the temporal profile. This distinction matters for dark-sky site assessment.

---

### Finding SE8 -- LOW: No Reference to Published Zone Boundary Equations

**Location:** `src/formulas/diagnostics_thresholds.py` (general)

**Problem:** Small & Elvidge (2022) publish specific mathematical boundaries for their five zones on the variance-mean scattergram, including trajectory equations such as variance = 0.5 * sqrt(mean) (dark-erratic boundary) and variance = 2.5 * sqrt(mean) (bright-erratic boundary). These provide empirically-derived, published thresholds that could be implemented in the codebase as an alternative or complement to the ad-hoc CV thresholds.

**Impact:** The codebase misses an opportunity to use published, peer-reviewed classification boundaries.

**Recommendation:**
Add the published zone boundary equations as constants in `diagnostics_thresholds.py` with full citation, even if not immediately used for classification. This creates a reference for future enhancement:
```python
# Small & Elvidge (2022) variance-mean zone boundaries
# variance = coefficient * sqrt(mean_radiance)
ZONE_BOUNDARY_DARK_ERRATIC = 0.5   # var = 0.5 * sqrt(mean)
ZONE_BOUNDARY_BRIGHT_ERRATIC = 2.5  # var = 2.5 * sqrt(mean)
```

---

## Summary

| # | Finding | Severity | Category |
|---|---------|----------|----------|
| SE1 | CV thresholds (0.2, 0.5) have no published justification in VIIRS literature | HIGH | Scientific rigor |
| SE2 | Missing skewness and kurtosis -- half of available statistical information discarded | HIGH | Completeness |
| SE3 | Single-dimension (CV) classification cannot distinguish physically distinct zones | MEDIUM | Methodology |
| SE4 | Heteroskedasticity of VIIRS data not acknowledged in trend diagnostics | MEDIUM | Scientific accuracy |
| SE5 | "Stable/moderate/erratic" labels do not match published five-zone taxonomy | MEDIUM | Terminology |
| SE6 | Scatter plot axes (mean vs. CV) differ from published convention (mean vs. variance) | LOW | Visualization |
| SE7 | IQR computed but not used for classification or interpretation | LOW | Completeness |
| SE8 | Published zone boundary equations not referenced or implemented | LOW | Completeness |

---

## Key Takeaway

The most significant gap is methodological: the codebase uses a single-dimensional CV-based classification with unjustified thresholds, while the peer-reviewed literature demonstrates that temporal stability of VIIRS nighttime lights requires at minimum a two-dimensional analysis (mean + variance) and benefits substantially from all four statistical moments. The existing approach is not wrong per se -- CV is a reasonable first-order metric -- but it lacks published justification and discards information that the literature shows is physically meaningful. For a dark-sky research study that will undergo peer review, the CV thresholds should be either justified with citation or supplemented with the multi-moment approach.

---

## References

- Small, C. & Elvidge, C.D. (2022). Statistical moments of VIIRS night-time lights. *Int. J. Remote Sensing*, 45(21). DOI: [10.1080/01431161.2022.2161857](https://doi.org/10.1080/01431161.2022.2161857)
- Small, C. (2021). Spatiotemporal Characterization of VIIRS Night Light. *Frontiers in Remote Sensing*, 2, 775399. [Full text](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2021.775399/full)
- Elvidge, C.D. et al. (2022). The VIIRS Day/Night Band: A Flicker Meter in Space? *Remote Sensing*. [EOG flicker page](https://eogdata.mines.edu/products/vnl/flicker/)
- Elvidge, C.D. et al. (2021). Annual time series of global VIIRS nighttime lights. *Remote Sensing*, 13(5), 922. [Full text](https://www.mdpi.com/2072-4292/13/5/922)
- Elvidge, C.D. et al. (2017). VIIRS night-time lights. *Int. J. Remote Sensing*, 38(21), 5860-5879. [Full text](https://www.tandfonline.com/doi/full/10.1080/01431161.2017.1342050)
