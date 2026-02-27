# Paper Review: Cinzano & Falchi (2012) vs. Codebase

**Paper:** Cinzano, P. & Falchi, F. (2012). "The propagation of light pollution in the atmosphere." *MNRAS*, 427(4), 3337-3357. DOI: 10.1111/j.1365-2966.2012.21884.x
**Date:** 2026-02-27

---

## Paper Summary

Cinzano & Falchi (2012) presents the **Extended Garstang Models (EGM)** and the **LPTRAN software package** -- a comprehensive numerical solution to the radiative transfer problem as applied to artificial light pollution propagation in the atmosphere. This paper is the theoretical foundation underlying the Falchi et al. (2016) World Atlas of Artificial Night Sky Brightness and represents the most complete treatment of how upward-emitted artificial light becomes zenith sky brightness at potentially distant observer locations.

### Key Methodology

1. **Radiative Transfer Approach:** The model computes the irradiance on each infinitesimal atmospheric volume element produced by ground sources, then integrates the scattered light along the line of sight toward the observer's zenith. This is fundamentally a ray-tracing approach through a layered atmosphere with both molecular (Rayleigh) and aerosol (Mie) scattering.

2. **Extended Garstang Models (EGM):** Improvements over the original Garstang (1986-2000) models include:
   - Multiple scattering (beyond the original double-scattering approximation)
   - Wavelength coverage from 250 nm to infrared
   - Earth curvature and horizon screening effects
   - Variable site and source elevation
   - Configurable atmospheric profiles (not just exponential decay)
   - Mix of boundary-layer and tropospheric aerosols, with up to 5 upper-atmosphere aerosol layers
   - Continuum and line gas absorption including ozone
   - Up to 5 cloud layers
   - Wavelength-dependent bidirectional ground surface reflectance from MODIS data
   - Geographically variable upward emission functions

3. **Sky Brightness as a Spatial Integral:** The zenith sky brightness B at an observer location is computed by integrating contributions from **all surrounding light sources** within a radius of up to 350 km (with 1 km resolution steps). In the Falchi (2016) atlas application, the operational integration radius was **195 km** (210 pixels at the equator). This means sky brightness at any point is **not** a property of the local pixel alone -- it is the sum of scattered light from potentially thousands of surrounding source pixels.

4. **Three-Parameter Upward Emission Function (UEF):** The model uses a three-parameter function describing how light is emitted upward as a function of angle from the zenith. Three intensity distributions were tested:
   - **Lambertian** (peak toward zenith)
   - **Low-angle** (peak near horizon, from poorly shielded luminaires)
   - **Intermediate** (peak at ~30 degrees above horizon)

   The Falchi (2016) atlas calibration found best-fit weights: W_a = 1.9 x 10^-3, W_b = 5.2 x 10^-4, W_c = 7.6 x 10^-5, revealing that significant low-angle emission exists beyond pure Lambertian reflection.

5. **Atmospheric Parameters:** The standard configuration uses a US62 atmosphere with:
   - Aerosol clarity parameter K = 1 (vertical extinction 0.33 mag in V-band at sea level)
   - Horizontal visibility 26 km
   - Optical depth tau = 0.31
   - Exponentially decreasing molecular density (scale height ~8 km)
   - Exponentially decreasing aerosol density (scale height ~1-2 km)

6. **Propagation Kernel / Point Spread Function:** The model effectively computes a propagation kernel that describes how a single source pixel's upward radiance contributes to sky brightness at any distance. This kernel falls off roughly as distance^(-2.5) -- steeper than inverse square due to atmospheric extinction, but much slower than exponential decay. The kernel is then convolved with the satellite radiance map to produce the sky brightness map.

### Key Findings

- **Double scattering** is sufficient for clean atmospheres, but multiple scattering becomes important for hazy/polluted atmospheres.
- **Earth curvature** screening becomes significant beyond ~100 km, limiting contributions from very distant sources.
- **Elevation** of both source and observer strongly affects propagation -- higher sites receive less scattered light because they are above much of the scattering atmosphere.
- The original exponential-density atmospheric model is "too simple compared with detailed models available in atmospheric physics."

---

## Cross-Reference: Paper Approach vs. Codebase Implementation

| Aspect | Cinzano & Falchi (2012) | Codebase Implementation |
|--------|------------------------|------------------------|
| **Sky brightness computation** | Integral over all sources within 195-350 km, using atmospheric radiative transfer | Direct linear multiplication of local pixel radiance by 0.177 |
| **Atmospheric scattering** | Rayleigh + Mie scattering with phase functions, extinction along paths | Not modeled -- assumes fixed proportionality |
| **Distance dependence** | Sky brightness depends on all sources within ~195 km; falls as ~d^(-2.5) | Sky brightness depends only on VIIRS radiance at local pixel |
| **Upward emission function** | Three-parameter model with Lambertian + low-angle + intermediate | Not modeled -- radiance used directly |
| **Elevation effects** | Source and observer elevation affect atmospheric path length | Not modeled |
| **Atmospheric conditions** | Parameterized by aerosol clarity K, optical depth tau, humidity | Not modeled -- assumes fixed conditions |
| **Light dome modeling** | Emerges naturally from integrating scattered light | Exponential decay fit: R(d) = peak * exp(-decay * d) + background |
| **Natural sky background** | 174 µcd/m² = 22.0 mag/arcsec² | 22.0 mag/arcsec² (consistent) |

---

## Findings

### Finding CF1 — HIGH: Sky Brightness Computed from Local Pixel Only, Ignoring 195 km Integration Radius

**Location:** `src/analysis/sky_brightness_model.py:radiance_to_sky_brightness()`, `src/formulas/sky_brightness.py:RADIANCE_TO_MCD`

**Problem:** The codebase computes sky brightness at each location using only the VIIRS radiance measured at that exact pixel:

```python
artificial_mcd = radiance_nw * RADIANCE_TO_MCD  # 0.177
total_mcd = artificial_mcd + natural_mcd
mag = -2.5 * np.log10(total_mcd / REFERENCE_MCD)
```

Cinzano & Falchi (2012) demonstrates that zenith sky brightness at any point is the **integral of scattered light from all sources within ~195 km**. The sky brightness at a dark-sky site 50 km from a major city is dominated not by the local pixel's upward radiance (which may be near zero), but by the light dome of the distant city whose upward emissions scatter off the atmosphere.

**Impact:** Distance-dependent and asymmetric errors:

| Scenario | Estimated Error |
|----------|----------------|
| Urban core (surrounded by bright area) | Underestimates by 0.2-0.5 mag |
| Suburban transition (near city) | Underestimates by 0.5-1.5 mag |
| Dark site 30-50 km from major city | Underestimates by 1-3+ mag |
| Truly isolated dark site (>150 km) | <0.2 mag error |

For Maharashtra: Bhimashankar (~40 km from Pune, pop ~7M) would appear very dark by local-pixel reading, but Pune's light dome dominates its actual zenith sky brightness through atmospheric scattering.

**Recommendation:** Use pre-computed Falchi et al. (2016) World Atlas GeoTIFF data (freely available from [GFZ Data Services](https://dataservices.gfz-potsdam.de/contact/showshort.php?id=escidoc:1541893)) instead of computing from raw VIIRS radiance. At minimum, document that the current approach is a "local-pixel-only approximation" that systematically underestimates brightness at dark sites near urban areas.

---

### Finding CF2 — HIGH: Exponential Decay Light Dome Model Is Physically Incorrect

**Location:** `src/analysis/light_dome_modeling.py:_exp_decay()`, `src/formulas/fitting.py`

**Problem:** The codebase models light dome spatial extent using exponential decay:

```python
def _exp_decay(d, peak, decay_rate, background):
    return peak * np.exp(-decay_rate * d) + background
```

Cinzano & Falchi (2012) shows that atmospheric light propagation follows a **power-law decline (~d^(-2.5))**, not exponential decay. The critical distinction: the codebase fits exponential decay to **satellite-observed upward radiance** as a function of distance from city center. This measures **urban morphology** (where lights are physically located), not **atmospheric light propagation** (how far scattered glow extends).

A city whose VIIRS radiance drops to background at 30 km may produce visible sky glow at 100+ km through atmospheric scattering.

**Impact:** The dome radius and effective area computed by this model are properties of the urban footprint, not of the atmospheric light dome. The actual zone of influence extends far beyond where VIIRS radiance drops below threshold.

**Recommendation:** Rename from "light dome modeling" to "urban radiance footprint modeling." Document that this models ground-level light source extent, not atmospheric glow propagation.

---

### Finding CF3 — MEDIUM: The 0.177 Conversion Factor Lacks Physical Derivation

**Location:** `src/formulas/sky_brightness.py:RADIANCE_TO_MCD`

**Problem:** The constant `RADIANCE_TO_MCD = 0.177` converts nW/cm²/sr (radiometric) to mcd/m² (photometric), but:

1. This is a **category boundary** from Falchi (2016) Table 1, not a derived conversion factor
2. The radiometric-to-photometric conversion depends on source spectral power distribution — different for HPS, LED, metal halide
3. Cinzano & Falchi (2012) does not use a linear conversion; the relationship is non-linear and depends on atmospheric conditions, geometry, and distance
4. VIIRS DNB (500-900 nm) does not match V-band photometry; ~34% of white LED radiant power falls outside the DNB bandpass

**Impact:** Systematic spectral bias that changes as India transitions from HPS to LED lighting.

**Recommendation:** Document the spectral assumptions and limitations. Note the factor is calibrated for HPS-era lighting and may underestimate sky brightness in LED-transitioning regions.

---

### Finding CF4 — MEDIUM: No Elevation Correction in Sky Brightness Conversion

**Location:** `src/analysis/sky_brightness_model.py:radiance_to_sky_brightness()`

**Problem:** The conversion applies a single global factor regardless of elevation. Cinzano & Falchi (2012) shows significant elevation dependence: higher-elevation observers are above more of the scattering atmosphere, receiving less scattered artificial light.

Maharashtra's terrain ranges from sea level (Mumbai coast) to ~1400m (Western Ghats dark-sky sites). This elevation difference changes the atmospheric column substantially.

**Impact:** Western Ghats sites (Bhimashankar ~900m, Mahabaleshwar ~1400m) have a natural elevation advantage that the codebase ignores. A first-order correction would be ~exp(-h/H_a) where H_a ≈ 1.5 km (aerosol scale height).

**Recommendation:** Add elevation-dependent correction or document the limitation.

---

### Finding CF5 — MEDIUM: Aerosol Optical Depth Not Parameterized

**Location:** `src/analysis/sky_brightness_model.py`, `src/formulas/sky_brightness.py`

**Problem:** Cinzano & Falchi (2012) identifies aerosol content as "of paramount importance." Maharashtra experiences significant seasonal variation: monsoon season vs. dry/haze season, with winter haze events substantially increasing near-source scattering.

The codebase has no atmospheric condition parameter.

**Impact:** During haze events, actual sky brightness can be significantly higher than clear-sky predictions. Post-monsoon (October-November) conditions are typically clearest and best for dark-sky assessment.

**Recommendation:** Document that conversion assumes standard clear atmosphere. Add seasonal aerosol note for Maharashtra.

---

### Finding CF6 — LOW: Light Dome Background Threshold Is Arbitrary

**Location:** `src/formulas/fitting.py:LIGHT_DOME_BACKGROUND_THRESHOLD = 0.5`

**Problem:** The 0.5 nW/cm²/sr threshold defines the "dome edge" in the exponential model but has no connection to propagation physics. In the Cinzano & Falchi framework, there is no sharp edge — brightness decreases continuously following the scattering kernel, extending to 195-350 km.

0.5 nW/cm²/sr ≈ 50% of natural sky background — still significant light pollution, so this threshold is generous.

**Recommendation:** Document as "satellite-visible urban extent" rather than "atmospheric influence extent."

---

### Finding CF7 — LOW: No Self-Contribution vs Remote-Contribution Decomposition

**Location:** `src/analysis/sky_brightness_model.py` — structural gap

**Problem:** The Cinzano & Falchi model inherently decomposes sky brightness into self-contribution (local pixel scattering back) and remote contribution (all other sources within 195 km). At dark-sky sites, remote contributions from nearby cities can provide 80-90% of total artificial sky brightness.

The codebase has no concept of this decomposition, potentially framing light pollution as a local problem when it is fundamentally regional.

**Recommendation:** Document the regional nature of sky brightness. Consider a simplified propagation integral or use the Falchi atlas data.

---

## Error Magnitude Summary

| Context | Approximate Error vs Full Propagation Model |
|---------|---------------------------------------------|
| Urban core | Underestimates by 0.2-0.5 mag |
| Suburban transition | Underestimates by 0.5-1.5 mag |
| Dark site 30-50 km from city | **Underestimates by 1-3+ mag** |
| Isolated dark site >150 km | <0.2 mag |

---

## Summary Table

| # | Finding | Severity | Category |
|---|---------|----------|----------|
| CF1 | Local-pixel-only sky brightness ignores 195 km integration of scattered light | HIGH | Scientific accuracy |
| CF2 | Exponential decay models urban morphology, not atmospheric propagation (power-law ~d^-2.5) | HIGH | Scientific accuracy |
| CF3 | 0.177 conversion factor is a category boundary, not a derived spectral constant | MEDIUM | Scientific accuracy |
| CF4 | No elevation correction despite 0-1400m range in Maharashtra | MEDIUM | Missing physics |
| CF5 | No aerosol optical depth parameterization despite seasonal variation | MEDIUM | Missing physics |
| CF6 | Light dome threshold (0.5 nW) defines urban extent, not atmospheric influence | LOW | Methodology |
| CF7 | No self/remote contribution decomposition; light pollution framed as local not regional | LOW | Methodology |

---

## References

- Cinzano, P. & Falchi, F. (2012). The propagation of light pollution in the atmosphere. *MNRAS*, 427(4), 3337-3357. [Oxford Academic](https://academic.oup.com/mnras/article/427/4/3337/973668), [arXiv:1209.2031](https://arxiv.org/abs/1209.2031)
- Falchi, F. et al. (2016). The new world atlas of artificial night sky brightness. *Science Advances*, 2(6), e1600377.
- Bara, S. (2019). Fast Fourier-transform calculation of artificial night sky brightness maps.
- [GFZ World Atlas Data](https://dataservices.gfz-potsdam.de/contact/showshort.php?id=escidoc:1541893)
