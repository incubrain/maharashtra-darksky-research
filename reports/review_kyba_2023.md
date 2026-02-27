# Paper Review: Kyba et al. (2023) vs. Codebase

**Paper:** Kyba, C.C.M. et al. (2023). "Citizen scientists report global rapid reductions in the visibility of stars from 2011 to 2022." *Science*, 379(6639), 265-268.
**Date:** 2026-02-27

---

## Paper Summary

Kyba et al. (2023) used 51,351 citizen scientist observations from the Globe at Night program (2011-2022) to measure changes in naked-eye stellar visibility across 19,262 locations worldwide. This is the first large-scale study to quantify the discrepancy between satellite-measured and human-perceived changes in light pollution.

### Key Findings

1. **Sky brightness increasing at 9.6% per year** globally (equivalent to a doubling in ~8 years), based on the rate of star visibility loss
2. **Regional variation:** Europe at 6.5% per year, North America at 10.4% per year
3. **Massive discrepancy with satellite data:** VIIRS satellites measure only ~2% per year increase globally — a **~5x underestimate**
4. **Three reasons for the discrepancy:**
   - **Spectral blindness:** VIIRS DNB (500-900 nm) cannot detect blue light (<500 nm). LED lighting emits strongly in the blue, and blue light scatters ~5.5x more efficiently in the atmosphere (Rayleigh scattering ∝ λ⁻⁴)
   - **Directional blindness:** Satellites see primarily upward-directed light. Horizontally emitted light (facades, windows, billboards, vehicle headlights) contributes heavily to ground-level skyglow but is invisible from orbit
   - **Resolution masking:** New lighting installations in previously unlit areas may be below VIIRS detection threshold individually but collectively increase sky brightness

5. **Methodology:** Volunteers compare their visible sky against standardized star charts showing progressively fainter stars. Observations are filtered for cloud-free, moonless nights. A global sky brightness model (based on 2014 VIIRS data) is used as a baseline to account for geographic variation.

---

## Cross-Reference with Codebase

### Finding K1 — HIGH: VIIRS-Based Trends Systematically Underestimate Actual Sky Brightness Changes

**Location:** `src/formulas/trend.py`, `src/formulas/benchmarks.py`, `src/analysis/sky_brightness_model.py`

This is the most consequential finding from any of the four papers reviewed. Kyba et al. (2023) demonstrates that VIIRS-based trend analysis underestimates the actual rate of sky brightness change by approximately **5x** (2% satellite vs 9.6% ground-based).

The codebase computes VIIRS-based radiance trends and converts them to sky brightness magnitudes without any acknowledgment of this systematic bias. When the pipeline reports "dark-sky site X is stable" based on flat VIIRS radiance, the actual sky brightness may be increasing at nearly 10% per year due to undetected LED spectral shifts and horizontal light emissions.

**Impact:** All trend conclusions and dark-sky viability assessments are likely optimistic. Sites classified as "stable" or "low ALAN" based on VIIRS trends may actually be experiencing rapid sky brightness degradation.

**Recommendation:**
1. Add a prominent caveat to all trend output: "VIIRS-based trends may underestimate actual sky brightness changes by ~5x (Kyba et al. 2023)"
2. Consider applying a correction factor to the VIIRS-derived sky brightness trends (e.g., multiply the annual percent change by 4.8 = 9.6/2.0 as a rough correction)
3. Add the Kyba et al. findings to the published benchmarks module for comparison

---

### Finding K2 — HIGH: Bortle Classification from VIIRS Is Increasingly Inaccurate

**Location:** `src/analysis/sky_brightness_model.py:classify_bortle()`

The Bortle scale is fundamentally based on **naked-eye stellar visibility** — exactly what Kyba et al. (2023) measures. The codebase derives Bortle classifications from VIIRS radiance via the linear magnitude conversion.

If VIIRS underestimates sky brightness changes by ~5x, then the Bortle classifications are increasingly stale over time. A site classified as Bortle 3 ("Rural sky") in 2012 based on VIIRS may actually have degraded to Bortle 5 ("Suburban sky") by 2024, while the VIIRS-derived Bortle remains at 3.

**Impact:** The Bortle class output gives false confidence in dark-sky site quality. For a 13-year study period, the cumulative error could be ~1-2 Bortle classes.

**Recommendation:**
1. Add a time-decay warning to Bortle classifications: "Bortle class derived from VIIRS data for [year]. Actual class may be 1-2 levels worse due to undetected spectral and directional light pollution (Kyba et al. 2023)"
2. Consider using Globe at Night data directly for sites where citizen science observations are available in Maharashtra

---

### Finding K3 — MEDIUM: Published Benchmarks Don't Include Ground-Based Growth Rate

**Location:** `src/formulas/benchmarks.py:PUBLISHED_BENCHMARKS`

The codebase benchmarks include:
- Global average: 2.2% per year (Elvidge 2021, satellite)
- Developing Asia: 4.1% per year (Elvidge 2021, satellite)
- India national: 5.3% per year (Li 2020, satellite)

But none of these capture the ground-truth finding: 9.6% per year globally (Kyba 2023, citizen science). Adding this would provide crucial context when interpreting district-level trends.

**Impact:** Users comparing Maharashtra trends only against satellite-derived benchmarks get an incomplete picture.

**Recommendation:** Add to `PUBLISHED_BENCHMARKS`:
```python
"global_ground_truth": {
    "source": "Kyba et al. (2023)",
    "region": "Global (citizen science)",
    "period": "2011-2022",
    "annual_growth_pct": 9.6,
    "ci_low": 6.5,    # Europe
    "ci_high": 10.4,   # North America
    "note": "Ground-based sky brightness, ~5x higher than satellite estimates"
}
```

---

### Finding K4 — MEDIUM: No Discussion of Horizontal vs Upward Light Emissions

**Location:** General pipeline methodology

Kyba et al. emphasizes that a major source of the VIIRS-ground discrepancy is **horizontal light emissions** — light from facades, windows, vehicle headlights, and signs that propagates horizontally and scatters into the sky dome but is invisible to overhead satellites.

The codebase's dark-sky site analysis is entirely based on vertically-detected VIIRS radiance. For sites near highways (Harihareshwar on the coast road) or near towns (Bhandardara near resort areas), horizontal light from vehicle headlights and local businesses may dominate the actual sky brightness while being undetected by VIIRS.

**Impact:** Dark-sky site viability assessments may miss the largest actual source of sky brightness — ground-level horizontal light.

**Recommendation:** Document this limitation. For the research paper, discuss the VIIRS-ground discrepancy when presenting dark-sky site recommendations.

---

## Summary

| # | Finding | Severity | Category |
|---|---------|----------|----------|
| K1 | VIIRS trends underestimate actual sky brightness changes by ~5x | High | Scientific accuracy |
| K2 | Bortle classification from VIIRS increasingly inaccurate over time | High | Scientific accuracy |
| K3 | Published benchmarks missing ground-based growth rate (9.6% global) | Medium | Completeness |
| K4 | Horizontal light emissions (major skyglow source) invisible to VIIRS | Medium | Methodology limitation |

---

## References

- Kyba, C.C.M. et al. (2023). Citizen scientists report global rapid reductions in the visibility of stars from 2011 to 2022. *Science*, 379(6639), 265-268. DOI: [10.1126/science.abq7781](https://doi.org/10.1126/science.abq7781)
- [Scientific American coverage](https://www.scientificamerican.com/article/light-pollution-is-dimming-our-view-of-the-sky-and-its-getting-worse/)
- [EurekAlert press release](https://www.eurekalert.org/news-releases/976947)
