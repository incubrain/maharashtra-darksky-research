# Candidate Papers for Further Codebase Review

**Date:** 2026-02-27
**Purpose:** Independent literature search to identify papers that should be reviewed against the Maharashtra VIIRS dark-sky analysis codebase. These are papers we do NOT currently cite but whose methodology is directly relevant to what we implement.

---

## Already Reviewed (6 papers)

| Paper | Status |
|---|---|
| Elvidge et al. (2017) — VIIRS product quality | Reviewed |
| Elvidge et al. (2021) — Annual composites | Reviewed |
| Falchi et al. (2016) — Sky brightness atlas | Reviewed |
| Zheng et al. (2019) — Anisotropic viewing | Reviewed |
| Levin et al. (2020) — NTL review | Reviewed |
| Kyba et al. (2023) — Citizen science trends | Reviewed |

---

## Tier 1: High-Value Candidates (Strongly recommended for review)

### 1. Kyba et al. (2017) — "Artificially lit surface of Earth at night increasing in radiance and extent"

**Citation:** Kyba, C.C.M. et al. (2017). *Science Advances*, 3(11), e1701528. DOI: [10.1126/sciadv.1701528](https://doi.org/10.1126/sciadv.1701528)

**Why review:** This is the original source of the "2.2% annual growth" figure that our `benchmarks.py` attributes to Elvidge et al. (2021). Kyba (2017) reported **both** lit-area growth (2.2%) and radiance growth (2.2%) over 2012-2016, making it critical to verify our benchmark attribution. The paper also explicitly documents the LED spectral bias problem: "VIIRS sometimes records a dimming of some cities even though these cities are in fact the same in brightness or even more brightly lit" when LED replacement occurs.

**Codebase relevance:**
- `src/formulas/benchmarks.py` — benchmark growth rates and their attribution
- Finding E3 from previous review — may resolve the lit-area vs radiance ambiguity
- Finding F4 — LED spectral bias with quantitative ISS comparison data
- The 2012-2016 period aligns with our study's early years

**Review value:** HIGH — directly resolves an open finding and provides quantitative LED bias data

---

### 2. Small & Elvidge (2022) — "Statistical moments of VIIRS night-time lights"

**Citation:** Small, C. & Elvidge, C.D. (2022). Statistical moments of VIIRS night-time lights. *Int. J. Remote Sensing*, 45(21). DOI: [10.1080/01431161.2022.2161857](https://doi.org/10.1080/01431161.2022.2161857)

**Why review:** This paper introduces a framework for classifying VIIRS temporal behavior using four statistical moments (mean, variance, skew, kurtosis) rather than just coefficient of variation. They define 5 stability zones: "core lighting", "dark-erratics", "mid-erratics", "bright-erratics", and "bright and steady." Our pipeline uses only CV with hardcoded thresholds (0.2/0.5) that have no published citation.

**Codebase relevance:**
- `src/analysis/stability_metrics.py` — CV-based stability classification
- `src/formulas/diagnostics_thresholds.py` — `CV_STABLE_THRESHOLD=0.2`, `CV_ERRATIC_THRESHOLD=0.5`
- Could inform whether our simple stable/moderate/erratic classification is sufficient
- Their variance-vs-mean scattergram approach may reveal structure we're missing

**Review value:** HIGH — our stability classification has no published justification; this paper could provide one or expose gaps

---

### 3. Cinzano & Falchi (2012) — "The propagation of light pollution in the atmosphere"

**Citation:** Cinzano, P. & Falchi, F. (2012). *MNRAS*, 427(4), 3337-3357. DOI: [10.1111/j.1365-2966.2012.21884.x](https://doi.org/10.1111/j.1365-2966.2012.21884.x)

**Why review:** This is the atmospheric propagation model that underlies Falchi's (2016) World Atlas. Our pipeline converts VIIRS radiance to sky brightness using a simple linear multiplication (`radiance × 0.177`), bypassing the Cinzano-Falchi radiative transfer approach entirely. This paper would quantify how much error our simplified conversion introduces, especially for dark sites near urban light domes.

**Codebase relevance:**
- `src/analysis/sky_brightness_model.py` — radiance_to_sky_brightness()
- `src/analysis/light_dome_modeling.py` — exponential decay model (vs their propagation model)
- Finding F1 from Falchi review — provides the quantitative framework our simplified conversion omits
- Their 195 km integration radius vs our localized approach

**Review value:** HIGH — would quantify the error in our sky brightness estimates for the research paper

---

### 4. Bennie et al. (2014) — "Contrasting trends in light pollution across Europe based on satellite observed night time lights"

**Citation:** Bennie, J., Davies, T.W., Duffy, J.P. & Gaston, K.J. (2014). *Scientific Reports*, 4, 3789. DOI: [10.1038/srep03789](https://doi.org/10.1038/srep03789)

**Why review:** Demonstrates that light pollution trends are non-uniform — some areas brighten while others dim simultaneously. They apply spatial decomposition methods to understand where and why brightening/dimming occurs. Relevant to our finding that 34/36 districts show a 2016 breakpoint and to interpreting heterogeneous trends across Maharashtra.

**Codebase relevance:**
- `src/analysis/breakpoint_analysis.py` — interpreting structural breaks
- `src/analysis/graduated_classification.py` — district-level tier classification
- Their methodology for decomposing aggregate trends into spatial components
- Understanding LED-related apparent dimming in their European data

**Review value:** MEDIUM-HIGH — methodological framework for interpreting heterogeneous trend patterns

---

### 5. Min et al. (2017) — "Using VIIRS Day/Night Band to Measure Electricity Supply Reliability: Preliminary Results from Maharashtra, India"

**Citation:** Min, B. et al. (2017). *Remote Sensing*, 8(9), 711. DOI: [10.3390/rs8090711](https://doi.org/10.3390/rs8090711)

**Why review:** This is the only peer-reviewed study using VIIRS specifically for Maharashtra. It correlates VIIRS DNB early-morning radiance with feeder-line voltage data, finding r=0.85 between daytime outage frequency and VIIRS overpass-time brightness. Critical for understanding Maharashtra-specific VIIRS data characteristics and the impact of electricity reliability on radiance measurements.

**Codebase relevance:**
- Our entire pipeline processes Maharashtra VIIRS data
- Understanding whether radiance trends reflect actual lighting changes vs electricity supply reliability
- The paper uses early-morning VIIRS data (midnight-2am overpass) — relevant to our temporal assumptions
- Could inform interpretation of district-level trend heterogeneity (outage-prone vs reliable districts)

**Review value:** HIGH — the only Maharashtra-specific VIIRS study; directly relevant to our study area

---

## Tier 2: Moderate-Value Candidates

### 6. Gaston et al. (2012/2013) — "The ecological impacts of nighttime light pollution: a mechanistic appraisal"

**Citation:** Gaston, K.J. et al. (2013). *Biological Reviews*, 88(4), 912-927. DOI: [10.1111/brv.12036](https://doi.org/10.1111/brv.12036)

**Why review:** The foundational framework for understanding ecological impacts of ALAN. Our `src/formulas/ecology.py` defines `ECOLOGICAL_SENSITIVITY` weights (0.1-0.9) with no published citation. Gaston's framework distinguishes between light as a resource vs. light as information, and provides a mechanistic basis for sensitivity scoring.

**Codebase relevance:**
- `src/formulas/ecology.py` — uncited ecological sensitivity weights
- `src/analysis/ecological_overlay.py` — impact scoring methodology
- Would provide scientific grounding for the sensitivity weights

**Review value:** MEDIUM — ecological overlay is a secondary analysis, but the uncited weights are a gap

---

### 7. Hale et al. (2013) — "Mapping lightscapes: spatial patterning of artificial lighting in an urban landscape"

**Citation:** Hale, J.D. et al. (2013). *PLOS ONE*, 8(5), e61460. DOI: [10.1371/journal.pone.0061460](https://doi.org/10.1371/journal.pone.0061460)

**Why review:** Establishes a 20 km buffer distance for assessing light pollution impacts around ecologically sensitive sites, compared to our 10 km buffer. The CMS Light Pollution Guidelines (which build on this work) recommend 20 km as "a nominal distance at which artificial light impacts should be considered."

**Codebase relevance:**
- `src/config.py` — `SITE_BUFFER_RADIUS_KM=10`, `PROTECTED_AREA_BUFFER_KM=10`
- `src/site/site_analysis.py` — buffer construction methodology
- Currently cites Wang et al. (2022) for 10 km; Hale's work suggests this may be too small

**Review value:** MEDIUM — could challenge our buffer radius choice

---

### 8. Sánchez de Miguel et al. (2020) — "Reducing Variability and Removing Natural Light from Nighttime Satellite Imagery"

**Citation:** Sánchez de Miguel, A. et al. (2020). *Sensors*, 20(11), 3287. DOI: [10.3390/s20113287](https://doi.org/10.3390/s20113287)

**Why review:** Directly addresses airglow removal and natural light correction for VIIRS data — the exact gap identified in our Levin (2020) review (Finding L1). Provides a practical methodology for separating artificial light from natural airglow signal.

**Codebase relevance:**
- `src/formulas/trend.py` — airglow confound in dark-site trends
- `src/viirs_utils.py` — DBS approach vs their natural light removal
- Finding L1 from Levin review

**Review value:** MEDIUM — addresses a specific identified gap

---

### 9. Ghosh et al. (2021) — "Quantifying uncertainties in nighttime light retrievals from Suomi-NPP and NOAA-20 VIIRS Day/Night Band data"

**Citation:** Ghosh, T. et al. (2021). *Remote Sensing of Environment*, 263, 112557. DOI: [10.1016/j.rse.2021.112557](https://doi.org/10.1016/j.rse.2021.112557)

**Why review:** Quantifies pixel-level uncertainty in VIIRS DNB measurements — something our pipeline assumes is zero. Relevant to understanding confidence intervals in trend estimates and to the NPP→NOAA-20 satellite transition (2018+) that affects our study period.

**Codebase relevance:**
- `src/formulas/trend.py` — bootstrap CI doesn't account for measurement uncertainty
- Understanding the NPP→NOAA-20 transition impact on 2018+ data
- `src/analysis/sensitivity_analysis.py`

**Review value:** MEDIUM — would strengthen uncertainty quantification

---

## Tier 3: Lower Priority but Potentially Useful

### 10. Light pollution trends for International Dark Sky Places using VIIRS (2025)

**Citation:** Recent paper in *Progress in Earth and Planetary Science*. DOI: [10.1186/s40645-025-00739-x](https://doi.org/10.1186/s40645-025-00739-x)

**Why review:** The most recent paper specifically about dark-sky site monitoring with VIIRS. Could provide updated best practices and validation methodology for our dark-sky site analysis.

**Review value:** LOW-MEDIUM — very recent; methodology may align with or improve our approach

---

### 11. DDUGJY/Saubhagya Rural Electrification Impact

**Citation:** No single paper; the Deendayal Upadhyaya Gram Jyoti Yojana (2014-2018) and Saubhagya (2017-2019) programs are documented by the Government of India Ministry of Power.

**Why review:** Our breakpoint analysis shows 34/36 districts with a 2016 breakpoint, attributed partly to India's rural electrification programs. The Saubhagya scheme electrified 2.8 crore households by 2021, with Maharashtra being a key state. Understanding the temporal profile of electrification in Maharashtra would help distinguish genuine lighting growth from policy-driven step changes.

**Codebase relevance:**
- `src/analysis/breakpoint_analysis.py` — interpreting the ubiquitous 2016 breakpoint
- Understanding which districts had the largest electrification gains
- Not a traditional paper review but a data-driven validation

**Review value:** LOW-MEDIUM — contextual rather than methodological

---

## Summary: Priority Ranking

| Priority | Paper | Primary Value |
|---|---|---|
| 1 | Kyba et al. (2017) | Resolves benchmark attribution, quantifies LED bias |
| 2 | Small & Elvidge (2022) | Justifies/challenges our stability classification |
| 3 | Cinzano & Falchi (2012) | Quantifies sky brightness conversion error |
| 4 | Min et al. (2017) | Only Maharashtra-specific VIIRS study |
| 5 | Bennie et al. (2014) | Framework for heterogeneous trend interpretation |
| 6 | Gaston et al. (2013) | Ecological sensitivity framework |
| 7 | Hale et al. (2013) | 20 km buffer recommendation |
| 8 | Sánchez de Miguel (2020) | Airglow removal methodology |
| 9 | Ghosh et al. (2021) | Measurement uncertainty quantification |
| 10 | Dark Sky Places VIIRS (2025) | Latest dark-sky monitoring practices |
| 11 | DDUGJY/Saubhagya docs | Electrification timeline context |

---

## Selection Criteria Used

Papers were selected based on:
1. **Direct methodological overlap** — Does the paper describe a method we implement?
2. **Gap filling** — Does it address an identified finding from previous reviews?
3. **Study area relevance** — Is it specific to India or Maharashtra?
4. **Uncited constants** — Does it provide published justification for our hardcoded thresholds?
5. **Recency** — More recent papers may capture state-of-the-art best practices
6. **Cross-paper corroboration** — Does it independently confirm issues found in other reviews?

Papers about DMSP/OLS only, non-ALAN remote sensing, or purely social science applications were excluded as not directly relevant to our VIIRS processing pipeline.
