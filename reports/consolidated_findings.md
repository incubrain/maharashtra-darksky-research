# Consolidated Research Paper Review Findings

**Date:** 2026-02-27
**Papers reviewed:** 6 total (2 from previous round + 4 new)

---

## All Papers Reviewed

| Paper | Focus | Report |
|---|---|---|
| Elvidge et al. (2017) | VIIRS DNB product quality filtering | [research_paper_review.md](research_paper_review.md) |
| Elvidge et al. (2021) | Annual composite methodology & benchmarks | [research_paper_review.md](research_paper_review.md) |
| Falchi et al. (2016) | Sky brightness atlas & radiance→magnitude conversion | [review_falchi_2016.md](review_falchi_2016.md) |
| Zheng et al. (2019) | Anisotropic VIIRS viewing angle effects | [review_zheng_2019.md](review_zheng_2019.md) |
| Levin et al. (2020) | Comprehensive NTL review & best practices | [review_levin_2020.md](review_levin_2020.md) |
| Kyba et al. (2023) | Citizen science sky brightness trends | [review_kyba_2023.md](review_kyba_2023.md) |

---

## All Findings by Severity

### HIGH Severity (5 findings)

| ID | Finding | Source Paper | Location |
|---|---------|-------------|----------|
| **E1** | Negative radiance not handled — `log(negative + 1e-6)` produces NaN, silently corrupts trend model | Elvidge (2017) | `viirs_process.py`, `trend.py` |
| **F4** | No spectral bias warning for HPS→LED transition — VIIRS cannot detect blue light, may show false radiance decreases | Falchi (2016) | Pipeline-wide |
| **Z1** | Incorrect/fabricated citation for radial gradient radii — DOI 10.3390/rs11182132 resolves to unrelated thermal plume paper | Zheng (2019) | `config.py:46-51` |
| **K1** | VIIRS trends underestimate actual sky brightness changes by ~5x (2% satellite vs 9.6% ground-based) | Kyba (2023) | `trend.py`, `benchmarks.py` |
| **K2** | Bortle classification from VIIRS increasingly inaccurate over time — cumulative error of 1-2 Bortle classes over study period | Kyba (2023) | `sky_brightness_model.py` |

### MEDIUM Severity (7 findings)

| ID | Finding | Source Paper | Location |
|---|---------|-------------|----------|
| **E2** | `LIT_MASK_THRESHOLD = 0.5` is dead code — actual filter uses `> 0` | Elvidge (2017/2021) | `quality.py` |
| **E3** | Global 2.2% benchmark may be lit-area growth (not radiance) — pipeline compares radiance trends against lit-area rate | Elvidge (2021) | `benchmarks.py` |
| **F1** | Linear radiance→magnitude conversion skips atmospheric propagation model — underestimates brightness near light domes | Falchi (2016) | `sky_brightness_model.py` |
| **F2** | RADIANCE_TO_MCD=0.177 sourced as category boundary, not derived conversion factor — spectral-dependent | Falchi (2016) | `sky_brightness.py` |
| **Z2** | Anisotropic viewing angle effects (~50% variability) not acknowledged or corrected | Zheng (2019) | Pipeline-wide |
| **K3** | Published benchmarks missing ground-based growth rate (9.6% global, Kyba 2023) | Kyba (2023) | `benchmarks.py` |
| **L1** | No airglow correction — solar-cycle signal may bias dark-site trends over 13-year study | Levin (2020) | `trend.py` |

### LOW Severity (6 findings)

| ID | Finding | Source Paper | Location |
|---|---------|-------------|----------|
| **E4** | No vcmcfg/vcmslcfg preference in layer identification | Elvidge (2017) | `viirs_process.py` |
| **E5** | Bootstrap CI doesn't account for temporal autocorrelation | Elvidge (2021) | `trend.py` |
| **F3** | Bortle boundaries use Crumey (2014) numerical mapping, not Falchi's own categories | Falchi (2016) | `sky_brightness.py` |
| **Z3** | Gradient analysis uses DBS inconsistently with main pipeline | Zheng (2019) | `gradient_analysis.py` |
| **L2** | No PSF/adjacency correction for urban-proximate dark-sky sites | Levin (2020) | `site_analysis.py` |
| **L3** | Levin et al. (2020) not cited in codebase | Levin (2020) | General |

---

## Findings Grouped by Theme

### Theme 1: Spectral Bias & LED Transition (3 findings: F4, K1, K2)

The most impactful cross-cutting issue. Falchi (2016) warned that VIIRS cannot detect blue light pollution from LEDs. Kyba (2023) proved this empirically: ground-measured sky brightness is increasing 5x faster than VIIRS detects. This means:
- All VIIRS-based trend conclusions are optimistic
- Bortle classifications drift increasingly wrong over time
- Dark-sky site assessments may give false assurance

**Action required:** Pipeline-wide documentation caveat + consideration of spectral correction factor.

### Theme 2: Data Quality & Negative Radiance (2 findings: E1, L1)

Negative radiance values silently corrupt the trend model. Airglow adds solar-cycle noise to dark-site measurements. Both issues primarily affect low-radiance targets (dark-sky sites).

**Action required:** Add `np.maximum(0, radiance)` clipping or increase `LOG_EPSILON`. Document airglow as a confound.

### Theme 3: Citation & Documentation Integrity (3 findings: Z1, E3, F2)

A fabricated citation (Z1), a potentially misattributed benchmark (E3), and an unclear constant derivation (F2) undermine the paper's scientific credibility.

**Action required:** Fix the Zheng citation immediately. Verify the 2.2% benchmark source. Document the RADIANCE_TO_MCD derivation.

### Theme 4: Benchmarks & Interpretation (3 findings: E3, K3, K4)

The benchmark comparison module is incomplete — it only has satellite-derived rates and may conflate lit-area vs radiance growth. Adding the Kyba ground-truth rate would provide crucial context.

**Action required:** Add ground-based benchmark. Clarify lit-area vs radiance distinction.

### Theme 5: Spatial Methodology (3 findings: Z2, Z3, L2)

Anisotropic viewing effects, inconsistent DBS application, and PSF adjacency effects all introduce spatial biases. These are known limitations rather than bugs.

**Action required:** Documentation + consistency fixes.

---

## Priority Action Items

### Immediate (code fixes)

1. **Fix negative radiance handling** (E1) — Add clipping or increase LOG_EPSILON
2. **Fix Zheng citation** (Z1) — Remove fabricated DOI, find correct source
3. **Remove dead LIT_MASK_THRESHOLD constant** (E2) — Or wire it into the filter

### Short-term (documentation & benchmarks)

4. **Add VIIRS spectral bias caveat** (F4, K1, K2) — All trend and Bortle outputs
5. **Add Kyba ground-based benchmark** (K3) — 9.6% global rate
6. **Verify 2.2% benchmark attribution** (E3) — Lit-area vs radiance growth
7. **Document RADIANCE_TO_MCD derivation** (F2) — Note spectral assumptions

### Medium-term (methodology improvements)

8. **Add LED transition flag to trends** (F4) — Flag districts showing post-2015 decrease
9. **Consider block bootstrap or HAC-corrected CIs** (E5) — Temporal autocorrelation
10. **Document airglow confound for dark sites** (L1) — Solar cycle note
11. **Prefer vcmcfg over vcmslcfg in layer identification** (E4)
12. **Make DBS usage consistent across analyses** (Z3)

### Long-term (research improvements)

13. **Consider using Falchi World Atlas data directly** (F1) — Instead of linear conversion
14. **Add anisotropic viewing angle caveat** (Z2)
15. **Add PSF adjacency warning for near-urban sites** (L2)
16. **Cite Levin et al. (2020) as methodological reference** (L3)

---

## Cross-Paper Validation

Several findings were independently corroborated by multiple papers:

| Issue | Papers Confirming |
|---|---|
| VIIRS spectral blindness to blue light | Falchi (2016), Levin (2020), Kyba (2023) |
| LED transition creates measurement bias | Falchi (2016), Kyba (2023) |
| VIIRS underestimates actual brightness change | Kyba (2023), Levin (2020) |
| Negative radiance in VIIRS data | Elvidge (2017), Elvidge (2021) |
| Median preferred over mean for composites | Elvidge (2017), Elvidge (2021), Levin (2020) |

---

## Impact Assessment

| Category | Finding Count | Pipeline Impact |
|---|---|---|
| May produce incorrect scientific conclusions | 5 (E1, E3, K1, K2, F4) | High |
| Citation/documentation integrity | 3 (Z1, F2, L3) | Medium |
| Known limitations needing documentation | 5 (Z2, L1, L2, Z3, L4) | Low-Medium |
| Code hygiene | 2 (E2, E4) | Low |
| Statistical methodology | 2 (E5, K4) | Low-Medium |
