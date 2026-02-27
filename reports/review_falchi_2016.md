# Paper Review: Falchi et al. (2016) vs. Codebase

**Paper:** Falchi, F. et al. (2016). "The new world atlas of artificial night sky brightness." *Science Advances*, 2(6), e1600377.
**Date:** 2026-02-27

---

## Paper Summary

Falchi et al. (2016) produced the definitive global atlas of artificial night sky brightness using VIIRS DNB data propagated through an atmospheric radiative transfer model. The atlas is calibrated against >35,000 ground-based Sky Quality Meter (SQM) observations and models zenith sky brightness at every point on Earth.

### Key Methodology

1. **Natural sky background:** 174 µcd/m² ≈ 22.0 mag/arcsec² (zenith, moonless, cloudless, solar minimum)
2. **Propagation model:** Light from each source is scattered through the atmosphere using Cinzano & Falchi (2012) radiative transfer code, integrating contributions from all sources within 195 km radius
3. **Upward emission function:** Three competing models tested — best fit uses a weighted combination with dominant Lambertian and low-angle components
4. **Calibration equation:** B = SN + (W_a·A + W_b·B + W_c·C)(1 + dh), where S=1.15 scaling factor, d=-4.5%/hour nighttime decay
5. **Spectral limitation warning:** VIIRS DNB is sensitive to 500-900 nm only, missing blue/violet light. The paper explicitly warns that LED transition "will not be detected by VIIRS DNB" and that "4000K white LED lighting is about 2.5 times more polluting for the scotopic band" than sodium lamps

---

## Cross-Reference with Codebase

### Finding F1 — MEDIUM: Simplified Radiance-to-Magnitude Conversion Skips Atmospheric Propagation

**Location:** `src/analysis/sky_brightness_model.py:radiance_to_sky_brightness()`

The codebase converts VIIRS radiance directly to sky brightness magnitude using:
```python
artificial_mcd = radiance_nw * RADIANCE_TO_MCD          # × 0.177
natural_mcd = REFERENCE_MCD * 10 ** (-0.4 * 22.0)       # ≈ 0.100 mcd/m²
total_mcd = artificial_mcd + natural_mcd
mag = -2.5 * np.log10(total_mcd / REFERENCE_MCD)
```

Falchi et al. (2016) does **not** provide a simple linear conversion factor from ground-level VIIRS radiance to sky brightness. The 0.177 mcd/m² per nW/cm²/sr factor appears in Table 1 as the lower threshold of "pristine sky" but is not documented as a general-purpose conversion formula.

The atlas instead uses a full atmospheric propagation model that accounts for:
- Distance and direction from light sources
- Rayleigh and Mie scattering
- Atmospheric absorption
- Site elevation
- Upward emission function geometry

The codebase's direct linear conversion (`radiance × 0.177`) treats VIIRS upward radiance as proportional to zenith sky brightness at the same location. This is a reasonable first-order approximation but systematically underestimates sky brightness in areas affected by distant light domes (where sky brightness comes from remote sources, not local upward emissions) and overestimates it in bright urban cores (where much of the detected upward light escapes to space rather than scattering back to the zenith).

**Impact:** The sky brightness and Bortle classifications are approximate rather than physically modeled. For dark-sky sites near urban areas (e.g., Bhimashankar 40 km from Pune), the actual sky brightness will be higher than what the linear conversion predicts because it ignores Pune's light dome propagating over the intervening distance.

**Recommendation:** Add a caveat/documentation note that the conversion is a first-order approximation. For more accurate results, consider using the Falchi et al. World Atlas data directly (freely available as GeoTIFF at the GFZ data service) rather than computing sky brightness from VIIRS radiance.

---

### Finding F2 — MEDIUM: RADIANCE_TO_MCD = 0.177 Source Is Unclear

**Location:** `src/formulas/sky_brightness.py:RADIANCE_TO_MCD`

The codebase cites Falchi et al. (2016) Table S1 as the source for `RADIANCE_TO_MCD = 0.177`. However, Falchi et al. Table 1 (the main paper's color-brightness mapping table) shows 0.176 mcd/m² as the lower pristine threshold — it is a **category boundary**, not a conversion factor.

The actual nW/cm²/sr → mcd/m² conversion depends on the spectral composition of the light source, the VIIRS DNB spectral response function, and the photometric weighting (scotopic vs photopic). Different lighting technologies (HPS, LED, metal halide) would produce different conversion factors.

**Impact:** Using a single conversion factor introduces a systematic bias that depends on the dominant lighting technology in each area. As India transitions from sodium to LED street lighting, the actual mcd/m² per nW/cm²/sr ratio changes.

**Recommendation:** Document that the 0.177 factor is an approximation valid primarily for the HPS-dominated lighting era. Consider adding a note about the spectral uncertainty range.

---

### Finding F3 — LOW: Bortle Scale Boundaries Use Crumey (2014) Not Original Bortle (2001)

**Location:** `src/formulas/sky_brightness.py:BORTLE_THRESHOLDS`

The codebase correctly attributes the Bortle boundary values to Crumey (2014) for the numerical thresholds. However, Falchi et al. (2016) does **not** use the Bortle scale directly — they define their own brightness categories based on astronomical observability (Milky Way visibility, zodiacal light, etc.) with different boundaries than Bortle.

This is not an error per se (Bortle is a well-established scale), but the codebase should note that Falchi's atlas uses a different categorization scheme and that the Bortle numerical boundaries (21.75, 21.50, etc.) are Crumey's numerical mapping of Bortle's originally qualitative descriptions.

**Impact:** Minor. The Bortle classification is internally consistent and well-documented.

---

### Finding F4 — HIGH: No Spectral Bias Warning for LED Transition

**Location:** Throughout the pipeline — no warning exists

Falchi et al. (2016) explicitly warns: *"The increase in the scotopic band and in the blue part of the spectrum will not be detected by VIIRS DNB because of its lack of sensitivity at wavelengths shorter than 500 nm."*

They predict that LED transition will make VIIRS-based brightness appear to **decrease** in areas switching from HPS to LED, even though actual sky brightness increases. Specifically, "4000K white LED lighting is about 2.5 times more polluting for the scotopic band."

The codebase has no mechanism to:
1. Flag districts where LED transition may bias VIIRS trends downward
2. Apply any spectral correction for the HPS→LED transition
3. Warn users that VIIRS-based sky brightness estimates become less reliable as LED adoption increases

**Impact:** Districts showing VIIRS radiance decreases during 2016-2024 may actually be experiencing increasing sky brightness due to LED transition. This could lead to false conclusions about successful light pollution reduction.

**Recommendation:** Add at minimum a documentation caveat in the sky brightness model. Ideally, add a flag in the trend output when a district shows radiance decrease during the known LED transition period (post-2015), noting that this may be a spectral artifact rather than genuine darkening.

---

## Summary

| # | Finding | Severity | Category |
|---|---------|----------|----------|
| F1 | Linear radiance→magnitude conversion skips atmospheric propagation | Medium | Scientific accuracy |
| F2 | RADIANCE_TO_MCD=0.177 source is a category boundary, not a derived conversion factor | Medium | Documentation |
| F3 | Bortle boundaries use Crumey (2014) numerical mapping, not Falchi's own categories | Low | Documentation |
| F4 | No spectral bias warning for HPS→LED transition affecting VIIRS trends | High | Scientific accuracy |

---

## References

- Falchi, F. et al. (2016). The new world atlas of artificial night sky brightness. *Science Advances*, 2(6), e1600377. DOI: [10.1126/sciadv.1600377](https://doi.org/10.1126/sciadv.1600377)
- Crumey, A. (2014). Human contrast threshold and astronomical visibility. *MNRAS*, 442(3), 2600-2619.
- Bortle, J.E. (2001). Introducing the Bortle Dark-Sky Scale. *Sky & Telescope*, 101(2), 126.
- [PMC Full Text](https://pmc.ncbi.nlm.nih.gov/articles/PMC4928945/)
