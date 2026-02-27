# Paper Review: Kyba et al. (2017) vs. Codebase

**Paper:** Kyba, C.C.M. et al. (2017). "Artificially lit surface of Earth at night increasing in radiance and extent." *Science Advances*, 3(11), e1701528. DOI: [10.1126/sciadv.1701528](https://doi.org/10.1126/sciadv.1701528)

**Date:** 2026-02-27
**Scope:** Benchmark attribution, growth rate methodology, LED spectral bias, and trend comparison against the Maharashtra VIIRS dark-sky analysis pipeline.

---

## Paper Summary

### Overview

Kyba et al. (2017) is the first global analysis of nighttime light trends using VIIRS Day/Night Band (DNB) calibrated data. Using five years of cloud-free October monthly composites (2012--2016), the authors demonstrate that artificial light at night is increasing in both geographic extent and intensity. This paper is the original source of the widely cited "2.2% annual growth" figure for global nighttime light, predating Elvidge et al. (2021) by four years.

### Key Quantitative Findings

The paper reports three distinct global growth metrics, each measuring a different phenomenon:

| Metric | Annual Growth Rate | Definition |
|---|---|---|
| Lit area | 2.2% | Total area lit above 5 nW/cm^2/sr |
| Radiance of continuously lit areas | 2.2% | Brightness change in areas that were already lit |
| Total radiance (sum of lights) above 5 nW threshold | 1.8% | Combined effect of area expansion + brightening |

The coincidence of the lit-area rate and the continuously-lit-area radiance rate both being 2.2% is notable and is a source of confusion in downstream citations, including in our codebase.

### Methodology for Growth Rate Computation

The paper uses a **geometric mean ratio** approach, not regression:

    Annual rate = (Value_2016 / Value_2012)^(1/4)

This is equivalent to a compound annual growth rate (CAGR) computed from endpoint values over the 4-year period. This is distinct from the codebase's log-linear OLS regression approach, which fits all intermediate data points.

### Regional Findings

- Growth occurred "throughout South America, Africa, and Asia."
- The fastest growth rates were in developing regions "that didn't have a lot of light to start with."
- Only a few countries showed decreases: war-affected nations (Yemen, Syria) and some wealthy nations (Italy, Netherlands, Spain, USA) that may reflect LED transitions.
- "For the median country, the sum of total radiance grew by 15% from 2012 to 2016."
- No India-specific growth rate is reported in the main text, though India appears in the country-level supplementary figures as a rapidly growing country.

### LED Spectral Bias (Critical for Pipeline Interpretation)

The paper provides the most explicit discussion of VIIRS spectral limitations available at the time:

1. **Spectral band:** VIIRS DNB covers 500--900 nm, missing the 400--500 nm blue wavelength range entirely.
2. **LED underestimation:** "Decreases of approximately 30% could be due to complete lighting transition from high-pressure sodium to LED lamps rather than true decrease in visible light." This is the quantitative anchor for LED spectral bias.
3. **Milan ISS comparison:** Photographs from the International Space Station comparing Milan in 2012 (sodium vapor, yellow/orange) and 2015 (white LEDs) showed that VIIRS DNB recorded a decrease in radiance despite an actual increase in human-visible light output. This is direct observational evidence of VIIRS spectral blindness to LEDs.
4. **Lower bound interpretation:** All VIIRS-measured growth rates are described as "lower bounds on the increase of actual lighting in the human visual range."

### Aggregation Method

The paper explicitly uses **median** for country-level aggregation: "For the median country, the sum of total radiance grew by 15% from 2012 to 2016." This aligns with the codebase's use of median radiance.

### Study Period and Data

- **Years:** October cloud-free monthly composites from 2012, 2013, 2014, 2015, and 2016.
- **Sensor:** VIIRS DNB (500--900 nm).
- **Lit threshold:** 5 nW/cm^2/sr for defining "lit area."
- **Product:** VIIRS cloud-free composites (precursor to the Elvidge 2021 V2 annual series).

---

## Cross-Reference Table: Paper vs. Codebase

| Paper Recommendation / Finding | Codebase Implementation | Status |
|---|---|---|
| Lit area grows at 2.2%/yr (2012-2016) | `benchmarks.py`: `annual_growth_pct: 2.2` attributed to Elvidge (2021) | **Misattributed** -- see Finding 1 |
| Radiance of continuously lit areas grows at 2.2%/yr | Not distinguished from lit-area rate in benchmarks | **Missing distinction** -- see Finding 2 |
| Total radiance grows at 1.8%/yr | Not present in benchmarks | **Missing** -- see Finding 3 |
| Growth rates computed via geometric mean (CAGR) | `trend.py` uses log-linear OLS regression | **Different methodology** -- see Finding 4 |
| LED bias: ~30% underestimation for full HPS-to-LED transition | No LED bias quantification in pipeline | **Missing** -- see Finding 5 |
| VIIRS growth rates are lower bounds on actual growth | No caveat on trend outputs | **Missing** -- see Finding 6 |
| Lit area defined at 5 nW/cm^2/sr threshold | `config.py`: `SPRAWL_THRESHOLD_NW = 1.5` for urban area | **Different threshold** -- see Finding 7 |
| Median aggregation for country-level statistics | `benchmark_comparison.py` uses `growth_values.median()` | **Aligned** |
| October composites (single month per year) | Pipeline uses annual composites (12-month median) | **Different but acceptable** -- see Finding 8 |

---

## Findings

### Finding 1 -- HIGH: Benchmark 2.2% Is Misattributed to Elvidge (2021); Originates from Kyba (2017)

**Severity:** HIGH
**Location:** `src/formulas/benchmarks.py:12-19`, `tests/test_research_validation.py:334-346`

**The problem:**

The `global_average` benchmark is attributed to Elvidge et al. (2021):

```python
"global_average": {
    "source": "Elvidge et al. (2021)",
    "period": "2012-2019",
    "annual_growth_pct": 2.2,
    "ci_low": 1.8,
    "ci_high": 2.6,
}
```

The test docstring states: *"Table 3: Global lit area increased at 2.2% per year (2012-2019)."*

However, the 2.2% figure originates from **Kyba et al. (2017)**, who reported both 2.2% lit-area growth and 2.2% radiance growth over 2012--2016. Elvidge et al. (2021) is primarily a data product paper describing annual composite methodology; its growth rate reporting, if present, would cover a different period (2012--2019) and may use a different methodology.

This misattribution was previously flagged as finding E3 in the Elvidge review. Kyba (2017) now confirms:
- The 2.2% figure was first published by Kyba (2017), not Elvidge (2021).
- If Elvidge (2021) also reports 2.2%, it would be for a different time period and may reflect a coincidental convergence rather than an independent validation.

Furthermore, the confidence interval [1.8, 2.6] stored in the benchmark does not appear in Kyba (2017), which reports point estimates without formal confidence intervals for the global aggregate. This CI may be fabricated or sourced from an uncited third paper.

**Impact:** The codebase's primary global benchmark has an incorrect attribution, an unverified confidence interval, and an ambiguous period. Any published analysis referencing "Elvidge et al. (2021) reports 2.2% growth" could be factually incorrect.

**Recommendation:**
1. Change the attribution to `"Kyba et al. (2017)"` with period `"2012-2016"` if using the original 2.2% figure.
2. Separately verify whether Elvidge et al. (2021) independently reports a growth rate and, if so, add it as a distinct benchmark.
3. Remove or re-source the confidence interval [1.8, 2.6] -- it must be traceable to a specific table or statistical test in a published paper.
4. Update the test docstring to accurately reference the source.

---

### Finding 2 -- HIGH: Pipeline Compares Radiance Trends Against an Ambiguous Benchmark (Lit-Area vs. Radiance)

**Severity:** HIGH
**Location:** `src/formulas/benchmarks.py:16`, `src/analysis/benchmark_comparison.py:43`

**The problem:**

Kyba (2017) reports **three distinct** global growth rates, and the distinction matters:

| Metric | Rate | What it measures |
|---|---|---|
| Lit area growth | 2.2% | Spatial expansion of lighting above 5 nW threshold |
| Radiance growth (continuously lit) | 2.2% | Brightness increase of already-lit areas |
| Total radiance growth | 1.8% | Net effect of area + brightness combined |

The codebase's `benchmark_comparison.py` compares district-level `annual_pct_change` (computed from log-linear regression of median radiance time series) against the 2.2% benchmark. This pipeline metric is a **radiance trend** -- it measures how the brightness of a fixed geographic area changes over time.

The correct comparison benchmark depends on what the districts represent:
- If districts are **already urban/lit** throughout the study period, the correct benchmark is Kyba's radiance growth rate (2.2% for continuously lit areas, or 1.8% for total radiance).
- If districts include **newly lit pixels** (urban expansion into previously dark land), the metric conflates area growth with radiance growth.

Since the pipeline applies `lit_mask` filtering and computes median radiance across the district, it is primarily measuring radiance change in lit areas -- which should be compared against the 2.2% continuously-lit-area radiance rate or the 1.8% total radiance rate, not the lit-area expansion rate.

The field name `annual_growth_pct` does not indicate which type of growth it represents.

**Impact:** Without clarifying which of the three Kyba metrics the benchmark represents, the interpretation column in the benchmark comparison output ("faster"/"slower"/"similar") could be misleading. A district with 1.9% radiance growth would be classified as "similar" to the 2.2% benchmark (within the 1.0 pp threshold), but it would actually exceed the 1.8% total radiance rate. The ambiguity undermines the scientific validity of the comparison.

**Recommendation:**
1. Add a `metric_type` field to each benchmark entry: `"lit_area"`, `"radiance"`, or `"total_radiance"`.
2. Add a separate Kyba (2017) total-radiance benchmark at 1.8%.
3. In `benchmark_comparison.py`, match the pipeline metric type to the benchmark metric type, or at minimum log a warning when comparing radiance trends against a lit-area benchmark.

---

### Finding 3 -- MEDIUM: Total Radiance Growth Rate (1.8%) Missing from Benchmarks

**Severity:** MEDIUM
**Location:** `src/formulas/benchmarks.py`

**The problem:**

Kyba (2017) reports that "total radiance above 5 nW/cm^2/sr" grew at 1.8% per year globally. This is arguably the most holistic benchmark because it captures the combined effect of spatial expansion and brightness increase, which is what the sum-of-lights (SOL) metric measures.

The codebase has no benchmark for this rate. If district-level trends reflect total radiance changes (which they do, since new pixels enter the lit mask over time within a district boundary), 1.8% would be the appropriate global comparison, not 2.2%.

**Impact:** The absence of the 1.8% total radiance benchmark means the pipeline over-estimates the global baseline by 0.4 percentage points. For districts near the threshold, this could flip the interpretation from "faster than global" to "similar to global."

**Recommendation:**
Add a new benchmark entry:
```python
"global_total_radiance": {
    "source": "Kyba et al. (2017)",
    "region": "Global (all countries)",
    "period": "2012-2016",
    "metric_type": "total_radiance",
    "annual_growth_pct": 1.8,
    "ci_low": None,   # Not reported in paper
    "ci_high": None,
}
```

---

### Finding 4 -- MEDIUM: Trend Methodology Differs from Kyba's CAGR Approach

**Severity:** MEDIUM
**Location:** `src/formulas/trend.py:20-116`

**The problem:**

Kyba (2017) computes growth rates using a geometric mean ratio (CAGR):

    Annual rate = (Value_2016 / Value_2012)^(1/4) - 1

This uses only the endpoint values and assumes smooth exponential growth. It is simple but sensitive to noise in the endpoint years.

The codebase uses log-linear OLS regression:

```python
log_rad = np.log(radiance + log_epsilon)
X = sm.add_constant(years)
model = sm.OLS(log_rad, X).fit()
beta = model.params[1]
annual_pct = (np.exp(beta) - 1) * 100
```

This fits all intermediate years simultaneously, is more robust to endpoint noise, and provides statistical diagnostics (R-squared, p-value, confidence intervals). The log-linear approach is well-established in the ALAN literature and is cited to Elvidge et al. (2021) Section 3 and Li et al. (2020).

**Impact:** The methodological difference is not a bug -- the codebase's approach is arguably superior to Kyba's for multi-year time series (2012--2024). However, when comparing the pipeline's regression-derived growth rates against Kyba's CAGR-derived benchmarks, the two are not computed identically. In the presence of non-monotonic data (dips, spikes), the regression-derived rate can differ substantially from the CAGR.

This is a documented limitation, not a code defect, but it should be noted in the benchmark comparison output.

**Recommendation:**
1. Add a comment in `benchmark_comparison.py` noting the methodological difference: "Pipeline trends are log-linear OLS regression slopes; published benchmarks may use CAGR (endpoint ratio) methodology."
2. Optionally, compute and report the CAGR alongside the regression rate for transparency.

---

### Finding 5 -- HIGH: No LED Spectral Bias Quantification in Pipeline

**Severity:** HIGH
**Location:** Pipeline-wide; most relevant to `src/formulas/trend.py`, `src/analysis/benchmark_comparison.py`

**The problem:**

Kyba (2017) provides the single most important quantitative anchor for LED spectral bias in VIIRS data:

> "Decreases of approximately 30% could be due to complete lighting transition from high-pressure sodium to LED lamps rather than true decrease in visible light."

This means:
- A VIIRS-measured decrease of 30% in a city may actually represent **zero change** in human-visible light.
- A VIIRS-measured stability (0% change) may actually represent a **~43% increase** in human-visible light (since 1.0 / 0.7 = 1.43).
- All VIIRS growth rate estimates are lower bounds.

The Milan ISS comparison provides direct observational evidence: VIIRS recorded a radiance decrease for Milan between 2012 and 2015, while ISS photography showed the city had transitioned from sodium vapor (yellow/orange) to white LED lighting, which increased total human-visible light output.

Maharashtra is undergoing rapid LED adoption (India's UJALA LED distribution program has distributed hundreds of millions of LED bulbs since 2015). Any district-level trends from ~2015 onward are potentially biased downward by the LED spectral effect.

The pipeline currently has:
- No LED bias correction factor.
- No caveat on trend outputs about spectral limitations.
- No flagging of districts where LED adoption may have occurred.

This finding was previously identified as F4 (Falchi 2016 review) and K1 (Kyba 2023 review). Kyba (2017) provides the quantitative estimate (~30%) that those reviews lacked.

**Impact:** District trends for urbanizing areas of Maharashtra that have adopted LED street lighting may show artificially suppressed or negative growth rates that are interpreted as genuine ALAN reduction when they actually represent a measurement artifact. This directly affects the scientific conclusions of the analysis.

**Recommendation:**
1. Add a `LED_SPECTRAL_BIAS_FACTOR` constant in `config.py` with value `0.30` (Kyba 2017), representing the maximum fractional radiance reduction observable from full HPS-to-LED conversion.
2. Add a caveat string to all trend output files: "VIIRS DNB trends may underestimate actual radiance changes by up to 30% in areas transitioning to LED lighting (Kyba et al. 2017). All reported growth rates are lower bounds."
3. Consider flagging districts where the post-2015 trend shows a significant downward shift relative to the pre-2015 trend, as this may indicate LED adoption rather than genuine ALAN reduction.

---

### Finding 6 -- MEDIUM: Lit-Area Threshold Inconsistency (5 nW vs. 1.5 nW)

**Severity:** MEDIUM
**Location:** `src/config.py:75`

**The problem:**

Kyba (2017) defines "lit area" as pixels above **5 nW/cm^2/sr**. This is the threshold used for their 2.2% lit-area growth rate calculation.

The codebase uses `SPRAWL_THRESHOLD_NW = 1.5` nW/cm^2/sr as the threshold for "lit urban area in sprawl maps." There is also `ALAN_MEDIUM_THRESHOLD = 5.0` which matches Kyba's definition but is used for a different purpose (ALAN classification, not lit-area delineation).

If any part of the pipeline computes lit-area growth (as opposed to radiance growth) and compares it against the 2.2% benchmark, the threshold mismatch (1.5 vs. 5.0) would produce a different lit-area measurement.

**Impact:** Using a lower threshold (1.5 nW) captures more marginally-lit pixels, which tend to be more volatile (noise-prone) and more likely to cross in and out of the "lit" category between years. This would inflate apparent lit-area change rates relative to Kyba's 5 nW threshold. However, the pipeline does not currently compute a lit-area growth metric -- it computes radiance trends -- so this is primarily a documentation issue for now.

**Recommendation:**
1. If lit-area growth is ever computed, use 5 nW/cm^2/sr to match Kyba (2017).
2. Add a comment to `SPRAWL_THRESHOLD_NW` noting that it differs from the standard Kyba (2017) lit-area threshold of 5 nW.
3. Consider renaming or adding `LIT_AREA_THRESHOLD_NW = 5.0` with a Kyba citation for future use.

---

### Finding 7 -- LOW: Temporal Compositing Difference (October vs. Annual)

**Severity:** LOW
**Location:** `src/config.py:25`

**The problem:**

Kyba (2017) uses **October cloud-free monthly composites** for each year (a single month). The codebase uses **annual composites** (12-month median from Elvidge 2021 V2 product), as configured by `STUDY_YEARS = range(2012, 2025)`.

October composites have the advantage of avoiding seasonal lighting variations (holiday lights in December, agricultural burning periods) but are more susceptible to individual-month cloud and weather patterns. Annual composites smooth these effects via the 12-month median.

**Impact:** This is not a methodological flaw -- the codebase's use of annual composites is consistent with Elvidge et al. (2021) and is better suited for multi-year trend analysis. However, when comparing the pipeline's annual-composite-derived trends to Kyba's October-composite-derived benchmarks, the temporal basis differs. This is unlikely to cause more than minor discrepancies.

**Recommendation:** Document the compositing difference in the benchmark comparison output metadata.

---

### Finding 8 -- LOW: Kyba (2017) Not Cited Anywhere in Codebase

**Severity:** LOW
**Location:** `src/formulas/benchmarks.py`, `src/formulas/trend.py`, `src/config.py`

**The problem:**

Despite being the likely source of the 2.2% global benchmark and providing the key LED spectral bias estimate (~30%), Kyba et al. (2017) is not cited anywhere in the codebase's inline comments or docstrings. The only citations are to Elvidge (2017/2021) and Li (2020).

The paper is mentioned in the existing `research_paper_review.md` references section but not as a codebase citation.

**Impact:** Omitting the citation weakens provenance tracking and could lead reviewers to question the source of key parameter values.

**Recommendation:** Add Kyba et al. (2017) to the citations block in `benchmarks.py` and reference it specifically for the growth rate values and the LED spectral bias constant.

---

## Relationship to Previous Findings

This review directly resolves and extends several open findings from prior paper reviews:

| Prior Finding | Status After This Review |
|---|---|
| **E3** (Elvidge review): "Global 2.2% benchmark may be lit-area growth, not radiance" | **Confirmed and clarified.** Kyba (2017) reports 2.2% for BOTH lit-area AND continuously-lit radiance, but 1.8% for total radiance. The benchmark is misattributed to Elvidge. |
| **F4** (Falchi review): "No spectral bias warning for HPS-to-LED transition" | **Quantified.** Kyba (2017) provides the ~30% figure for full HPS-to-LED spectral bias. |
| **K1** (Kyba 2023 review): "VIIRS underestimates actual brightness change by ~5x" | **Corroborated.** The 30% LED bias plus horizontal emission blindness (Kyba 2023) together explain the 5x discrepancy between VIIRS (2%) and ground-based (9.6%) trends. |
| **K3** (Kyba 2023 review): "Published benchmarks missing ground-based growth rate" | **Additional context.** Kyba (2017) benchmarks are satellite-based, but the lower-bound interpretation supports adding the Kyba (2023) ground-based 9.6% rate. |

---

## Summary Table

| # | Finding | Severity | Category |
|---|---------|----------|----------|
| 1 | Benchmark 2.2% is misattributed to Elvidge (2021); originates from Kyba (2017) | **HIGH** | Attribution accuracy |
| 2 | Pipeline compares radiance trends against ambiguous benchmark (lit-area vs radiance vs total) | **HIGH** | Scientific accuracy |
| 3 | Total radiance growth rate (1.8%) missing from benchmarks | **MEDIUM** | Benchmark completeness |
| 4 | Trend methodology (OLS regression) differs from Kyba's CAGR approach | **MEDIUM** | Methodological documentation |
| 5 | No LED spectral bias quantification (~30%) in pipeline | **HIGH** | Scientific accuracy |
| 6 | Lit-area threshold inconsistency (1.5 nW vs Kyba's 5 nW) | **MEDIUM** | Parameter consistency |
| 7 | Temporal compositing difference (annual vs October) | **LOW** | Documentation |
| 8 | Kyba (2017) not cited in codebase despite being primary benchmark source | **LOW** | Citation integrity |

| Severity | Count |
|----------|-------|
| HIGH | 3 |
| MEDIUM | 3 |
| LOW | 2 |

---

## Recommended Consolidated Finding IDs

For integration with the consolidated findings report, the following IDs are proposed:

| New ID | Description | Merges With |
|---|---|---|
| **Ky1** | Benchmark 2.2% misattributed -- should cite Kyba (2017), not Elvidge (2021) | Extends E3 |
| **Ky2** | Lit-area vs radiance vs total-radiance distinction missing in benchmarks | Extends E3 |
| **Ky3** | LED spectral bias ~30% not quantified in pipeline | Extends F4, corroborates K1 |
| **Ky4** | Total radiance benchmark (1.8%) absent | New |
| **Ky5** | Methodology difference (OLS vs CAGR) undocumented | New |

---

## References

- Kyba, C.C.M. et al. (2017). Artificially lit surface of Earth at night increasing in radiance and extent. *Science Advances*, 3(11), e1701528. DOI: [10.1126/sciadv.1701528](https://doi.org/10.1126/sciadv.1701528)
- Elvidge, C.D. et al. (2021). Annual time series of global VIIRS nighttime lights derived from monthly averages: 2012 to 2019. *Remote Sensing*, 13(5), 922. DOI: [10.3390/rs13050922](https://doi.org/10.3390/rs13050922)
- Kyba, C.C.M. et al. (2023). Citizen scientists report global rapid reductions in the visibility of stars from 2011 to 2022. *Science*, 379(6639), 265-268. DOI: [10.1126/science.abq7781](https://doi.org/10.1126/science.abq7781)
- Falchi, F. et al. (2016). The new world atlas of artificial night sky brightness. *Science Advances*, 2(6), e1600377.
- [PMC Full Text (Open Access)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5699900/)
- [DarkSky International Press Release](https://darksky.org/news/five-years-of-satellite-images-show-global-light-pollution-increasing-at-a-rate-of-two-percent-per-year/)
- [Astronomy Now Summary](https://astronomynow.com/2017/11/26/a-brightening-world-study-shows-rise-in-global-light-pollution/)
