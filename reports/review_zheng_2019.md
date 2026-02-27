# Paper Review: Zheng et al. (2019) vs. Codebase

**Paper:** Zheng, Q., Weng, Q. & Wang, K. (2019). "Anisotropic characteristic of artificial light at night — Systematic investigation with VIIRS DNB multi-temporal observations." *Remote Sensing of Environment*, 233, 111357.
**Date:** 2026-02-27

---

## Paper Summary

Zheng et al. (2019) investigates how VIIRS DNB radiance varies with satellite viewing angle (anisotropic effects). Using nightly DNB data from 30 urban points across 6 cities (Beijing, Houston, Los Angeles, Moscow, Quito, Sydney), they demonstrate that viewing zenith angle significantly affects observed radiance.

### Key Findings

1. **Viewing angle matters:** They propose a Zenith-Radiance Quadratic (ZRQ) model showing that radiance varies systematically with the satellite zenith angle (average R² = 0.50)
2. **Building height interaction:** Tall buildings create "cold spots" (reduced radiance at high zenith angles due to shadowing) and "hot spots" (enhanced radiance from facade illumination)
3. **Azimuth matters too:** The Zenith-Azimuth-Radiance Binary Quadratic (ZARBQ) model (R² = 0.53) shows radiance also depends on the satellite's azimuth relative to street grid orientation
4. **Time series implications:** Uncorrected anisotropic effects add ~50% variability to apparent radiance time series, potentially masking or exaggerating real trends

---

## Cross-Reference with Codebase

### Finding Z1 — HIGH: Incorrect Citation for Radial Gradient Methodology

**Location:** `src/config.py:46-51`

The codebase contains:
```python
# Following Zheng et al. (2019) anisotropic ALAN investigation:
# "Radial extraction at 1, 5, 10, 20, 50 km from city centers
# characterises the exponential decay of urban light domes."
# Citation: Zheng, Q. et al. (2019). Developing a new cross-sensor
#           calibration model. Remote Sensing, 11(18), 2132.
URBAN_GRADIENT_RADII_KM = [1, 5, 10, 20, 50]
```

**There are two problems:**

1. **Wrong paper:** The DOI 10.3390/rs11182132 resolves to "Coastal Tidal Effects on Industrial Thermal Plumes in Satellite Imagery" by Faulkner, Bulgin & Merchant (2019) — a completely unrelated paper about thermal plumes in coastal waters.

2. **Wrong methodology attribution:** Zheng et al. (2019) (DOI: 10.1016/j.rse.2019.111357) is about anisotropic viewing angle effects, NOT about radial profile extraction from city centers. The paper studies angle-radiance relationships, not distance-radiance gradients.

The actual paper in the user's bibliography is:
```
@article{Zheng2019Anisotropic,
  title   = {Anisotropic characteristic of artificial light at night...},
  journal = {Remote Sensing of Environment},
  doi     = {10.1016/j.rse.2019.111357}
}
```

**Impact:** Critical citation integrity issue. The radial gradient methodology (`[1, 5, 10, 20, 50] km`) has no verified published source in the codebase. The specific radii appear to be a reasonable set of distances for gradient analysis but are not traceable to a peer-reviewed paper.

**Recommendation:**
1. Remove or correct the Zheng citation for the radial gradient radii
2. Find the actual source for the [1, 5, 10, 20, 50] km radii set (common choices in urban NTL literature include Imhoff et al. 1997, Small et al. 2005, or simply operational/conventional choices)
3. If no specific source exists, document them as "conventional analysis distances" rather than attributing them to a paper

---

### Finding Z2 — MEDIUM: Anisotropic Viewing Angle Effects Not Accounted For

**Location:** Entire pipeline — `src/viirs_process.py`, `src/site/site_analysis.py`

The actual Zheng et al. (2019) paper demonstrates that VIIRS DNB radiance varies by up to ~50% depending on the satellite's viewing zenith angle for the same location. Since VIIRS has a wide swath (~3000 km) and Maharashtra (~800 km E-W extent) may be observed at different zenith angles on different passes, this introduces systematic noise into the annual composites.

The annual compositing process (median of monthly composites) partially mitigates this by averaging over many viewing geometries. However, for the **site-level buffer analysis** (10 km radius ≈ ~22 VIIRS pixels across), the small spatial extent means that all pixels within a buffer share a similar viewing geometry on any given night, so the annual median retains some anisotropic bias.

The codebase does not:
1. Check or record the viewing geometry metadata for each observation
2. Apply any anisotropic correction model
3. Flag sites or districts where viewing angle variation may affect trend reliability

**Impact:** For trend analysis, this introduces additional noise that inflates confidence intervals. For inter-site comparisons within a single year, sites at the edge of the VIIRS swath will have different systematic biases than nadir sites.

**Recommendation:** This is a known limitation of all VIIRS composite-based studies. Add a documentation note about viewing angle uncertainty. Consider mentioning the ~50% variability finding in trend diagnostics output to contextualize the bootstrap CI widths.

---

### Finding Z3 — LOW: Radial Gradient Analysis Uses DBS, Inconsistent with Main Pipeline

**Location:** `src/analysis/gradient_analysis.py`

The radial gradient extraction applies Dynamic Background Subtraction (DBS) before computing zonal statistics:
```python
data = viirs_utils.apply_dynamic_background_subtraction(data, year=year)
```

This is inconsistent with the main site metrics pipeline, which explicitly avoids DBS. For gradient analysis, DBS may be appropriate (to remove the state-wide noise floor), but it introduces a known time-varying bias (the P1.0 floor rises from 0.10 to 0.49 nW over 2012-2024).

If gradient profiles are compared across years (e.g., "did the light dome expand?"), the time-varying DBS floor could create artificial trends.

**Impact:** Inter-annual gradient comparisons may show systematic shifts unrelated to actual light dome changes.

**Recommendation:** Document whether gradient analysis is intended for within-year (single snapshot) or across-year (temporal trend) comparison. If across-year, consider using the un-DBS'd radiance or a fixed DBS floor.

---

## Summary

| # | Finding | Severity | Category |
|---|---------|----------|----------|
| Z1 | Fabricated/incorrect citation for radial gradient radii (DOI resolves to unrelated thermal plume paper) | High | Citation integrity |
| Z2 | Anisotropic viewing angle effects (~50% variability) not acknowledged or corrected | Medium | Methodology gap |
| Z3 | Gradient analysis uses DBS inconsistently with main pipeline | Low | Internal consistency |

---

## References

- Zheng, Q., Weng, Q. & Wang, K. (2019). Anisotropic characteristic of artificial light at night. *Remote Sensing of Environment*, 233, 111357. DOI: [10.1016/j.rse.2019.111357](https://doi.org/10.1016/j.rse.2019.111357)
- Faulkner, A., Bulgin, C.E. & Merchant, C.J. (2019). Coastal Tidal Effects on Industrial Thermal Plumes in Satellite Imagery. *Remote Sensing*, 11(18), 2132. DOI: [10.3390/rs11182132](https://doi.org/10.3390/rs11182132) — the paper the codebase actually cites
