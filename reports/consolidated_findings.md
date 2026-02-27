# Consolidated Research Paper Review Findings

**Date:** 2026-02-27
**Papers reviewed:** 11 total (across 3 rounds)

---

## All Papers Reviewed

### Round 1 — Core VIIRS Processing

| Paper | Focus | Report |
|---|---|---|
| Elvidge et al. (2017) | VIIRS DNB product quality filtering | [research_paper_review.md](research_paper_review.md) |
| Elvidge et al. (2021) | Annual composite methodology & benchmarks | [research_paper_review.md](research_paper_review.md) |

### Round 2 — Sky Brightness, Viewing Geometry, Best Practices, Ground Truth

| Paper | Focus | Report |
|---|---|---|
| Falchi et al. (2016) | Sky brightness atlas & radiance→magnitude conversion | [review_falchi_2016.md](review_falchi_2016.md) |
| Zheng et al. (2019) | Anisotropic VIIRS viewing angle effects | [review_zheng_2019.md](review_zheng_2019.md) |
| Levin et al. (2020) | Comprehensive NTL review & best practices | [review_levin_2020.md](review_levin_2020.md) |
| Kyba et al. (2023) | Citizen science sky brightness trends | [review_kyba_2023.md](review_kyba_2023.md) |

### Round 3 — Benchmarks, Stability, Propagation, Electrification, Trend Heterogeneity

| Paper | Focus | Report |
|---|---|---|
| Kyba et al. (2017) | Global growth rates, LED spectral bias | [review_kyba_2017.md](review_kyba_2017.md) |
| Small & Elvidge (2022) | Multi-moment stability classification | [review_small_elvidge_2022.md](review_small_elvidge_2022.md) |
| Cinzano & Falchi (2012) | Atmospheric light propagation model | [review_cinzano_falchi_2012.md](review_cinzano_falchi_2012.md) |
| Min et al. (2017) / Mann et al. (2016) | Maharashtra electricity reliability & VIIRS | [review_min_2017.md](review_min_2017.md) |
| Bennie et al. (2014) | Heterogeneous NTL trends & ecological framework | [review_bennie_2014.md](review_bennie_2014.md) |

---

## All Findings by Severity

### HIGH Severity (13 findings)

| ID | Finding | Source Paper | Location |
|---|---------|-------------|----------|
| **E1** | Negative radiance not handled — `log(negative + 1e-6)` produces NaN, silently corrupts trend model | Elvidge (2017) | `viirs_process.py`, `trend.py` |
| **F4** | No spectral bias warning for HPS→LED transition — VIIRS cannot detect blue light, may show false radiance decreases | Falchi (2016) | Pipeline-wide |
| **Z1** | Incorrect/fabricated citation for radial gradient radii — DOI 10.3390/rs11182132 resolves to unrelated thermal plume paper | Zheng (2019) | `config.py:46-51` |
| **K1** | VIIRS trends underestimate actual sky brightness changes by ~5x (2% satellite vs 9.6% ground-based) | Kyba (2023) | `trend.py`, `benchmarks.py` |
| **K2** | Bortle classification from VIIRS increasingly inaccurate over time — cumulative error of 1-2 Bortle classes | Kyba (2023) | `sky_brightness_model.py` |
| **KY1** | Benchmark 2.2% misattributed to Elvidge (2021) — originates from Kyba (2017); CI [1.8, 2.6] unverifiable | Kyba (2017) | `benchmarks.py` |
| **KY2** | Benchmark metric type ambiguous — Kyba reports lit-area (2.2%), continuously-lit radiance (2.2%), and total radiance (1.8%) as separate metrics | Kyba (2017) | `benchmark_comparison.py` |
| **KY3** | No LED spectral bias quantification — Kyba (2017) estimates HPS→LED causes ~30% VIIRS underestimate; Maharashtra actively transitioning under UJALA | Kyba (2017) | Pipeline-wide |
| **SE1** | CV stability thresholds (0.2, 0.5) have no published justification in VIIRS literature | Small & Elvidge (2022) | `diagnostics_thresholds.py` |
| **SE2** | Missing skewness and kurtosis — half of available statistical information discarded from stability analysis | Small & Elvidge (2022) | `stability_metrics.py` |
| **CF1** | Sky brightness computed from local pixel only, ignoring 195 km integration of scattered light — 1-3 mag error at dark sites near cities | Cinzano & Falchi (2012) | `sky_brightness_model.py` |
| **CF2** | Exponential decay light dome model describes urban morphology, not atmospheric propagation (which follows ~d^(-2.5) power law) | Cinzano & Falchi (2012) | `light_dome_modeling.py` |
| **M1** | VIIRS trends conflate ALAN growth with electricity reliability improvements (DDUGJY/UJALA/Saubhagya) — cannot be separated without auxiliary data | Min (2017) | `trend.py`, pipeline-wide |

### MEDIUM Severity (15 findings)

| ID | Finding | Source Paper | Location |
|---|---------|-------------|----------|
| **E2** | `LIT_MASK_THRESHOLD = 0.5` is dead code — actual filter uses `> 0` | Elvidge (2017/2021) | `quality.py` |
| **E3** | Global 2.2% benchmark may be lit-area growth, not radiance growth | Elvidge (2021) | `benchmarks.py` |
| **F1** | Linear radiance→magnitude conversion skips atmospheric propagation | Falchi (2016) | `sky_brightness_model.py` |
| **F2** | RADIANCE_TO_MCD=0.177 sourced as category boundary, not derived conversion factor | Falchi (2016) | `sky_brightness.py` |
| **Z2** | Anisotropic viewing angle effects (~50% variability) not acknowledged | Zheng (2019) | Pipeline-wide |
| **K3** | Published benchmarks missing ground-based growth rate (9.6% global) | Kyba (2023) | `benchmarks.py` |
| **L1** | No airglow correction — solar-cycle signal may bias dark-site trends | Levin (2020) | `trend.py` |
| **SE3** | Single-dimension CV classification cannot distinguish physically distinct lighting zones | Small & Elvidge (2022) | `stability_metrics.py` |
| **SE4** | VIIRS data heteroskedasticity not acknowledged in trend diagnostics | Small & Elvidge (2022) | `trend_diagnostics.py` |
| **SE5** | "Stable/moderate/erratic" labels don't match published five-zone taxonomy | Small & Elvidge (2022) | `classification.py` |
| **CF3** | The 0.177 conversion factor lacks physical derivation; is a category boundary | Cinzano & Falchi (2012) | `sky_brightness.py` |
| **CF4** | No elevation correction despite Maharashtra's 0-1400m range | Cinzano & Falchi (2012) | `sky_brightness_model.py` |
| **CF5** | No aerosol optical depth parameterization despite seasonal variation | Cinzano & Falchi (2012) | `sky_brightness_model.py` |
| **M2** | 2016 breakpoint (34/36 districts) partly explained by electrification policy, not just VIIRS product transition | Min (2017) | `breakpoint_analysis.py` |
| **M3** | Rural dark-sky site trends especially unreliable — rural areas had 3x worse load-shedding and were primary electrification beneficiaries | Min (2017) | `site_analysis.py` |
| **B1** | No brightening/dimming pixel decomposition in analytical pipeline | Bennie (2014) | `pipeline_steps.py` |
| **B2** | Ecological sensitivity weights (0.1-0.9) are unsourced — not cited to any published study | Bennie (2014) | `ecology.py` |

### LOW Severity (14 findings)

| ID | Finding | Source Paper | Location |
|---|---------|-------------|----------|
| **E4** | No vcmcfg/vcmslcfg preference in layer identification | Elvidge (2017) | `viirs_process.py` |
| **E5** | Bootstrap CI doesn't account for temporal autocorrelation | Elvidge (2021) | `trend.py` |
| **F3** | Bortle boundaries use Crumey (2014) numerical mapping | Falchi (2016) | `sky_brightness.py` |
| **Z3** | Gradient analysis uses DBS inconsistently with main pipeline | Zheng (2019) | `gradient_analysis.py` |
| **L2** | No PSF/adjacency correction for urban-proximate dark-sky sites | Levin (2020) | `site_analysis.py` |
| **L3** | Levin et al. (2020) not cited in codebase | Levin (2020) | General |
| **L4** | VIIRS effective resolution (750m native) not documented | Levin (2020) | `config.py` |
| **SE6** | Scatter plot axes differ from published convention | Small & Elvidge (2022) | Visualization |
| **SE7** | IQR computed but not leveraged for classification | Small & Elvidge (2022) | `stability_metrics.py` |
| **SE8** | Published zone boundary equations not referenced | Small & Elvidge (2022) | `stability_metrics.py` |
| **CF6** | Light dome threshold (0.5 nW) defines urban extent, not atmospheric influence | Cinzano & Falchi (2012) | `fitting.py` |
| **CF7** | No self/remote contribution decomposition for sky brightness | Cinzano & Falchi (2012) | `sky_brightness_model.py` |
| **M4-M9** | Annual composites mask seasonal cycling; 01:30 AM overpass captures anomalous conditions; published benchmarks include electrification signal; no VIIRS version covariate; no outage-dark vs genuinely-dark distinction; no sub-district spatial heterogeneity | Min (2017) | Various |
| **B3-B8** | District aggregation loses heterogeneity; breakpoint module doesn't classify change type; no LED correction; simplistic impact formula; direction-agnostic stability; percentile tiers can't distinguish absolute dimming from rank shifts | Bennie (2014) | Various |

---

## Findings Grouped by Theme

### Theme 1: VIIRS Cannot Measure What We Claim It Measures (F4, K1, K2, KY3, M1)

The most consequential cross-cutting finding. Three independent lines of evidence converge:
- **Spectral blindness** (Falchi 2016, Kyba 2017): VIIRS misses blue LED light, underestimating radiance by ~30% for LED-transitioning areas
- **Ground truth divergence** (Kyba 2023): Sky brightness increasing 5x faster than VIIRS detects (9.6% vs 2%)
- **Electrification confound** (Min 2017): Maharashtra VIIRS trends conflate ALAN growth with electricity supply reliability improvements during India's largest-ever rural electrification drive

**Implication:** VIIRS-based trend conclusions for Maharashtra 2012-2024 require heavy caveats. Trends cannot be interpreted as pure light pollution growth.

### Theme 2: Sky Brightness Model Is Fundamentally Simplified (CF1, CF2, CF4, CF5, F1)

The Cinzano & Falchi (2012) review reveals that our radiance→sky-brightness conversion is a first-order approximation that ignores:
- Atmospheric scattering from sources within 195 km (1-3 mag error at dark sites)
- Elevation effects (0-1400m range in Maharashtra)
- Seasonal aerosol variation
- The fact that light domes follow power-law (~d^-2.5), not exponential decay

**Implication:** All Bortle classifications and sky brightness estimates for dark-sky sites are approximate. The Falchi World Atlas GeoTIFF should be used instead for accurate assessments.

### Theme 3: Stability Classification Lacks Published Foundation (SE1, SE2, SE3, SE5)

The Small & Elvidge (2022) review exposes that:
- Our CV thresholds (0.2/0.5) appear nowhere in peer-reviewed VIIRS literature
- Single-dimension CV cannot distinguish physically distinct lighting zones
- The accepted approach uses mean+variance (at minimum), ideally all four moments
- Our "stable/moderate/erratic" labels don't match published taxonomy

**Implication:** Stability classifications should be treated as exploratory, not definitive.

### Theme 4: Citation & Benchmark Integrity (Z1, KY1, KY2, E3, B2)

Multiple citation/attribution issues:
- **Fabricated Zheng citation** (Z1): DOI resolves to unrelated paper
- **Benchmark misattribution** (KY1): 2.2% originates from Kyba (2017), not Elvidge (2021)
- **Metric ambiguity** (KY2): lit-area vs radiance growth conflated
- **Unsourced ecological weights** (B2): Sensitivity scores have no published basis

**Implication:** These undermine scientific credibility and must be fixed before publication.

### Theme 5: Maharashtra-Specific Confounds (M1, M2, M3, M4)

The Min (2017) review reveals critical Maharashtra-specific issues:
- VIIRS trends conflate ALAN with electrification (DDUGJY 2014, UJALA 2015, Saubhagya 2017)
- The 2016 breakpoint (34/36 districts) likely reflects real policy change, not just VIIRS product transition
- Rural dark-sky site trends are especially unreliable (3x worse load-shedding + primary electrification beneficiaries)

**Implication:** The research paper must discuss the electrification confound prominently.

### Theme 6: Missing Spatial Decomposition (B1, B3, Z2, Z3, L2)

Bennie (2014) demonstrates that aggregate trends mask spatial heterogeneity. Our pipeline lacks:
- Brightening/dimming pixel decomposition within districts
- Sub-district analysis capability
- Consistent DBS treatment across analyses

---

## Priority Action Items (Updated)

### P0: Critical — Must Fix Before Publication

| # | Action | Findings | Type |
|---|--------|----------|------|
| 1 | Fix negative radiance handling (clip to 0 or increase LOG_EPSILON) | E1 | Code fix |
| 2 | Fix fabricated Zheng citation — remove DOI, find real source or mark as conventional | Z1 | Citation fix |
| 3 | Correct benchmark attribution (Kyba 2017, not Elvidge 2021) and clarify metric type | KY1, KY2, E3 | Citation fix |
| 4 | Add prominent VIIRS limitation caveat to all trend/Bortle outputs re: spectral bias and electrification confound | F4, K1, K2, KY3, M1 | Documentation |
| 5 | Source or remove ecological sensitivity weights | B2 | Citation fix |

### P1: High Priority — Should Fix Before Publication

| # | Action | Findings | Type |
|---|--------|----------|------|
| 6 | Add Kyba (2023) ground-based 9.6% benchmark + Kyba (2017) 1.8% total radiance rate | K3, KY2 | Code + data |
| 7 | Document electrification confound (DDUGJY/UJALA/Saubhagya timeline) in breakpoint analysis output | M1, M2, M3 | Documentation |
| 8 | Document that sky brightness conversion is local-pixel-only approximation; recommend Falchi atlas for accurate assessments | CF1, F1 | Documentation |
| 9 | Document CV thresholds as project-specific heuristics (not published) or adopt Small & Elvidge zone boundaries | SE1, SE5 | Documentation |
| 10 | Remove dead LIT_MASK_THRESHOLD constant or wire into filter | E2 | Code fix |
| 11 | Rename "light dome modeling" to "urban radiance footprint modeling" | CF2 | Documentation |

### P2: Medium Priority — Methodology Improvements

| # | Action | Findings | Type |
|---|--------|----------|------|
| 12 | Add skewness + kurtosis to stability metrics; consider two-dimensional classification | SE2, SE3 | Enhancement |
| 13 | Add LED transition flag for districts showing post-2015 radiance decrease | F4, KY3, B5 | Enhancement |
| 14 | Add brightening/dimming pixel decomposition per district | B1, B3 | Enhancement |
| 15 | Consider block bootstrap or HAC-corrected CIs for temporal autocorrelation | E5 | Enhancement |
| 16 | Add VIIRS version covariate to trend model (v21/v22 transition) | M7 | Enhancement |
| 17 | Document airglow as dark-site trend confound | L1 | Documentation |
| 18 | Make DBS usage consistent across analyses | Z3 | Code fix |
| 19 | Add elevation correction to sky brightness conversion | CF4 | Enhancement |

### P3: Long-Term — Research Quality Improvements

| # | Action | Findings | Type |
|---|--------|----------|------|
| 20 | Integrate Falchi World Atlas GeoTIFF for sky brightness instead of linear conversion | CF1, CF7 | Major enhancement |
| 21 | Add anisotropic viewing angle caveat | Z2 | Documentation |
| 22 | Add PSF adjacency warning for near-urban dark-sky sites | L2 | Documentation |
| 23 | Cite Levin et al. (2020) as methodological reference throughout | L3 | Documentation |
| 24 | Prefer vcmcfg over vcmslcfg in layer identification | E4 | Code fix |
| 25 | Add seasonal aerosol note for Maharashtra sky brightness | CF5 | Documentation |
| 26 | Classify breakpoint change types (acceleration, deceleration, reversal) | B4 | Enhancement |

---

## Cross-Paper Validation (Updated)

| Issue | Papers Confirming |
|---|---|
| VIIRS spectral blindness to blue/LED light | Falchi (2016), Levin (2020), Kyba (2017), Kyba (2023), Bennie (2014) |
| VIIRS underestimates actual brightness change | Kyba (2017), Kyba (2023), Levin (2020) |
| LED transition creates measurement bias | Falchi (2016), Kyba (2017), Kyba (2023), Bennie (2014) |
| Negative radiance in VIIRS data | Elvidge (2017), Elvidge (2021) |
| Median preferred over mean for composites | Elvidge (2017), Elvidge (2021), Levin (2020), Small & Elvidge (2022) |
| Multi-moment analysis superior to single-metric | Small & Elvidge (2022), Bennie (2014) |
| Sky brightness is regional, not local | Cinzano & Falchi (2012), Falchi (2016), Levin (2020) |
| Electrification confounds VIIRS trends in India | Min (2017), Mann (2016), Hsu (2021) |
| Trend heterogeneity masked by aggregation | Bennie (2014), Min (2017) |

---

## Impact Assessment (Updated)

| Category | Finding Count | Examples |
|---|---|---|
| **Conclusions may be scientifically incorrect** | 8 | E1, K1, K2, KY2, CF1, CF2, M1, M3 |
| **Citation/attribution errors** | 5 | Z1, KY1, E3, B2, F2 |
| **Missing methodology** | 7 | SE1, SE2, B1, CF4, CF5, M7, B4 |
| **Known limitations needing documentation** | 10 | F4, KY3, Z2, L1, L2, CF7, M2, M4-M6 |
| **Code hygiene** | 3 | E2, E4, Z3 |
| **Statistical methodology** | 3 | E5, SE4, SE3 |

---

## Final Assessment

Across 11 papers and 42+ findings, the most critical insight is that **VIIRS-based radiance trends in Maharashtra during 2012-2024 cannot be straightforwardly interpreted as light pollution growth rates**. Three independent factors converge:

1. **Spectral bias:** VIIRS misses LED blue-light emissions (~30% underestimate)
2. **Electrification confound:** India's largest rural electrification drive coincides with the study period
3. **Ground-truth divergence:** Actual sky brightness is increasing ~5x faster than VIIRS detects

The codebase's quality filtering, spatial analysis, and trend computation are technically competent, but the **interpretation framework** needs significant strengthening with caveats, auxiliary data, and multi-source validation before the results can support dark-sky policy recommendations.
