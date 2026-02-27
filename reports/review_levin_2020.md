# Paper Review: Levin et al. (2020) vs. Codebase

**Paper:** Levin, N., Kyba, C.C.M., Zhang, Q., de Miguel, A.S., Roman, M.O., Li, X. & Elvidge, C.D. (2020). "Remote sensing of night lights: A review and an outlook for the future." *Remote Sensing of Environment*, 237, 111443.
**Date:** 2026-02-27

---

## Paper Summary

Levin et al. (2020) is the most comprehensive review paper on nighttime light remote sensing, co-authored by the leading researchers in the field (including Kyba, Elvidge, and Li — whose other papers our pipeline cites directly). It covers the entire history from DMSP/OLS through VIIRS, catalogues applications, identifies methodological challenges, and recommends best practices.

### Key Recommendations and Findings

1. **VIIRS DNB spectral range (500-900 nm):** The DNB is panchromatic but excludes blue/violet (<500 nm). This means it cannot detect the full spectral impact of the HPS→LED transition
2. **LED transition creates measurement bias:** "The broad spectral bandpass enables it to capture changes in LED light emissions despite its spectral limitations," but the blue component is missed, leading to underestimation of total radiance increase
3. **Blooming and overestimation:** The review discusses how the DMSP/OLS suffered from significant blooming (pixels appearing lit beyond actual city boundaries). While VIIRS has greatly reduced this, some adjacency effects remain
4. **Airglow variability:** "Temporal variation of natural light sources such as airglow limits the ability of night light sensors to detect changes in small sources of artificial light"
5. **Multispectral future:** The review advocates for future sensors with higher spatial resolution and multispectral coverage (blue through NIR) to distinguish lighting technologies
6. **Zenith angle effects:** Cross-references the work by Zheng et al. on viewing angle dependencies
7. **Applications catalogue:** Maps the field into urban mapping, population estimation, GDP correlation, conflict monitoring, disaster response, ecological impact, and light pollution measurement

---

## Cross-Reference with Codebase

### Finding L1 — MEDIUM: No Airglow Correction Applied

**Location:** `src/viirs_process.py`, `src/formulas/trend.py`

Levin et al. (2020) highlights that natural airglow is a significant source of noise in VIIRS DNB data, particularly for low-radiance targets (like dark-sky sites). Airglow varies on timescales of minutes to years, correlated with the solar cycle.

The codebase uses annual composites, which partially average out short-term airglow variability. However, the solar-cycle component (11-year period) can introduce a systematic trend over the 2012-2024 study period. Solar minimum occurred around 2019-2020, meaning the first half of the study (2012-2014, near solar maximum) had higher airglow than the middle (2018-2020).

For dark-sky sites with median radiance near 0.1-0.5 nW/cm²/sr, airglow fluctuations of ~0.05-0.1 nW/cm²/sr represent 10-100% of the signal. This could significantly bias trend estimates for the darkest sites.

**Impact:** Trend results for dark-sky sites may be contaminated by the solar-cycle airglow signal, potentially showing artificial brightening during solar maximum years.

**Recommendation:** Document airglow as a known confound for low-radiance trend analysis. Consider adding the solar cycle phase as a covariate in the trend model for dark-sky sites, or at minimum noting it in trend diagnostics.

---

### Finding L2 — MEDIUM: No Adjacency/Blooming Correction for Urban Buffer Analysis

**Location:** `src/site/site_analysis.py`, `src/analysis/gradient_analysis.py`

Although VIIRS has much less blooming than DMSP/OLS, the review notes residual adjacency effects remain. For the 10 km buffer analysis around dark-sky sites near cities, light from adjacent bright pixels can "spill" into dark pixels through the point spread function (PSF) of the sensor.

The codebase clips site buffers to the Maharashtra land boundary but does not apply any PSF deconvolution or adjacency correction. For sites like Bhimashankar (~40 km from Pune) or Bhandardara (~60 km from Nashik), some of the measured radiance within the buffer may be optical spillover rather than actual ground-level lighting.

**Impact:** Dark-sky sites near urban areas may appear brighter than they actually are, leading to pessimistic dark-sky viability assessments.

**Recommendation:** Add a note about PSF effects in site analysis output. For sites within 50 km of major cities, consider flagging the adjacency risk. This is a known limitation that cannot be easily corrected without the sensor PSF model.

---

### Finding L3 — LOW: Levin et al. Not Referenced in Codebase

**Location:** General codebase

Despite being the most authoritative review paper in the field — co-authored by Kyba, Elvidge, Li (all cited elsewhere in the codebase) — Levin et al. (2020) is not cited anywhere in the pipeline code or documentation. The research paper's bibliography includes it, but the codebase does not reference it.

This is a missed opportunity: the review's recommendations could inform several methodological choices in the pipeline, and citing it would strengthen the scientific justification.

**Impact:** Documentation gap. No code impact.

**Recommendation:** Add Levin et al. (2020) as a reference in the methodology documentation, particularly for discussions of VIIRS limitations, spectral bias, and best practices.

---

### Finding L4 — LOW: Resolution Limitation Not Documented

**Location:** `src/config.py:VIIRS_RESOLUTION_DEG`

The codebase correctly records `VIIRS_RESOLUTION_DEG = 0.004166667` (~15 arc-seconds ≈ 450 m at the equator). However, Levin et al. (2020) notes the effective spatial resolution is actually ~750 m (the native DNB pixel size) and that the EOG composites are resampled to a 15 arc-second grid. At Maharashtra's latitude (~18°N), the actual ground resolution is approximately:
- E-W: 15" × cos(18°) × 111 km/° ≈ 440 m
- N-S: 15" × 111 km/° ≈ 460 m

This means each VIIRS pixel covers ~0.2 km², and the 10 km radius buffer contains approximately 440 pixels (as documented in the code). The resolution is sufficient for district-level and buffer-level aggregation but cannot resolve individual streets, buildings, or small light sources.

**Impact:** Minor. The codebase already uses appropriate aggregation methods (zonal statistics over many pixels) that are suitable for this resolution.

---

## Summary

| # | Finding | Severity | Category |
|---|---------|----------|----------|
| L1 | No airglow correction — solar-cycle signal may bias dark-site trends | Medium | Scientific accuracy |
| L2 | No PSF/adjacency correction for urban-proximate dark-sky sites | Medium | Methodology gap |
| L3 | Levin et al. (2020) not cited in codebase despite being the field's authoritative review | Low | Documentation |
| L4 | VIIRS effective resolution is 750 m (native) resampled to 450 m grid — not documented | Low | Documentation |

---

## References

- Levin, N. et al. (2020). Remote sensing of night lights: A review and an outlook for the future. *Remote Sensing of Environment*, 237, 111443. DOI: [10.1016/j.rse.2019.111443](https://doi.org/10.1016/j.rse.2019.111443)
- [Semantic Scholar entry](https://www.semanticscholar.org/paper/Remote-sensing-of-night-lights:-A-review-and-an-for-Levin-Kyba/a3f349034ced49c46f460a60f5253373607d300f)
