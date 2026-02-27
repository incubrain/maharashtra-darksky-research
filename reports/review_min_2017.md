# Paper Review: Min, O'Keeffe & Zhang (2017) + Mann, Melaas & Malik (2016) + Hsu et al. (2021) vs. Codebase

**Primary Paper:** Min, B., O'Keeffe, Z., & Zhang, F. (2017). "Whose Power Gets Cut? Using High-Frequency Satellite Images to Measure Power Supply Irregularity." *World Bank Policy Research Working Paper* 8131.

**Companion Paper (Maharashtra-specific):** Mann, M.L., Melaas, E.K., & Malik, A. (2016). "Using VIIRS Day/Night Band to Measure Electricity Supply Reliability: Preliminary Results from Maharashtra, India." *Remote Sensing*, 8(9), 711. https://doi.org/10.3390/rs8090711

**Supporting Paper (Annual Cycling):** Hsu, F., Zhizhin, M., Ghosh, T., Elvidge, C., & Taneja, J. (2021). "The Annual Cycling of Nighttime Lights in India." *Remote Sensing*, 13(6), 1199.

**Date:** 2026-02-27

---

## Paper Summaries

### Min, O'Keeffe & Zhang (2017) — Power Supply Irregularity Index

Min et al. (2017) developed the **Power Supply Irregularity (PSI) index**, a satellite-derived measure of electricity supply stability for approximately 600,000 villages across India using the complete archive of DMSP-OLS nighttime imagery from 1993--2013 (over 30,000 nightly images, 5+ terabytes of data).

#### Methodology
- **PSI Definition:** The unexplained variability in night-to-night light output for a village, computed as the residual of a regression that predicts the standard deviation of light output given a village's mean brightness. Areas with frequent power outages exhibit excess brightness variability (bright when power is on, dark during outages).
- **Data Processing:** Raw DMSP-OLS data was processed using a Statistically Recalibrated Visible Band (SR-VIS) method: quality filtering (cloud cover, stray light removal), background noise correction using unpopulated "dark" reference areas, and normalization to correct for unrecorded dynamic gain calibration settings.
- **Validation:** Correlated PSI with (1) Maharashtra utility SAIFI data (System Average Interruption Frequency Index) across 163 district-year observations from 33 districts (2009--2013), and (2) Indian Human Development Survey self-reported outage data from 217 districts.

#### Key Findings
- Consistent, statistically significant positive correlation between PSI and SAIFI across all model specifications for Maharashtra.
- Annual composites "depict overly prolonged timespans, smooth away substantial variation in light output over the calendar year, and do not enable precise evaluation of discrete interventions like new village electrification projects."
- Rural areas experience disproportionately worse power irregularity. One Indian utility's 2008 policy dictated maximum 4-hour daily cuts for largest cities but up to 12 hours for rural villages.
- Wide variation in power stability *within* states, sometimes exceeding inter-state variation.
- Sharp discontinuities in PSI across state borders, reflecting different utility jurisdictions and policies.

### Mann, Melaas & Malik (2016) — VIIRS for Maharashtra Electricity Reliability

This is the first study to use VIIRS daily DNB data specifically to measure electricity reliability at fine spatiotemporal scales, focusing on Maharashtra.

#### Methodology
- **Ground Truth:** Electricity Supply Monitoring Initiative (ESMI) data from 39 voltage monitoring locations across Maharashtra (Nagpur, Akola, Pune, Solapur, Mumbai, Nashik districts), operated by the Indian NGO Prayas.
- **Classification:** Voltage readings below 100 volts classified as outages; above 100 volts as normal supply.
- **Satellite Data:** VIIRS DNB pixel-level radiance extracted at each monitoring location, matched to the closest satellite overpass time.
- **Model:** Random forest classifier trained on VIIRS radiance features (mean, median, standard deviation of DNB pixel stacks) combined with ESMI voltage data.
- **VIIRS Overpass Time:** Suomi NPP crosses the equator at approximately **01:30 local solar time** (descending node). Over Maharashtra (~18-22 degrees N latitude), the actual overpass occurs at roughly **00:00--02:00 AM IST**.

#### Key Findings
- Limited ability to detect individual outages with available training data, likely due to the small number of outages observed in the preliminary monitoring period.
- The after-midnight (0--2 AM) overpass timing "casts doubt on whether electricity reliability estimated during the overpass time is representative of reliability at other times of the day when power outages can be more disruptive."
- However, Min et al. (2017) separately found a correlation of **r = 0.85** between daytime outage frequency (6 AM--6 PM) and nighttime outage frequency (midnight--2 AM) using seven years of Maharashtra feeder-line voltage data, partially mitigating this concern.

### Hsu et al. (2021) — Annual Cycling of Nighttime Lights in India

#### Methodology
- Used autocorrelation function (ACF) analysis on VIIRS monthly composites at 15 arc-second resolution across all of India.
- Classified each grid cell into three archetypes: **acyclic**, **single peak**, and **dual peak** seasonal patterns.
- Validated against IHDS and ESMI power stability data.

#### Key Findings
- India exhibits pervasive annual cycling in VIIRS nighttime lights, driven by unstable power supply creating systematic seasonal radiance variation.
- Locations with less reliable electricity show stronger annual cycling (higher ACF amplitudes).
- This cycling is confounded with seasonal agricultural and cultural lighting patterns.
- **Implication:** Annual composites for Indian locations average over systematic seasonal variability that is itself an indicator of electricity reliability, not lighting infrastructure.

---

## Cross-Reference with Codebase

| Paper Finding | Codebase Assumption | Gap |
|---|---|---|
| Annual composites smooth away critical night-to-night variability in Indian power supply | Codebase uses VNL annual composites exclusively (`config.py:STUDY_YEARS = range(2012, 2025)`) | Cannot detect electricity reliability changes vs. lighting infrastructure changes |
| VIIRS overpass at ~01:30 local time captures off-peak electricity usage | Trend model assumes radiance reflects actual lighting infrastructure (`trend.py`) | Radiance at 1:30 AM may reflect load-shedding patterns, not installed capacity |
| PSI varies dramatically within states, exceeding inter-state variation | District-level analysis assumes radiance trends reflect ALAN growth (`breakpoint_analysis.py`) | Within-Maharashtra PSI variation could dominate the radiance signal |
| r = 0.85 correlation between daytime and nighttime outages in Maharashtra | No electricity reliability covariate in trend models | Trend changes could reflect reliability improvements, not new lighting |
| Rural areas experience 3x worse load-shedding than urban areas | Rural dark-sky sites compared against urban benchmarks without reliability adjustment (`benchmark_comparison.py`) | Rural site radiance trends conflate electrification with reliability improvement |
| DDUGJY launched Dec 2014, Saubhagya launched Oct 2017, UJALA LED programme Jan 2015 | 2016 breakpoint identified across 34/36 districts (`breakpoint_analysis.py` header) | Breakpoint partially or substantially caused by real electrification policy, not just VIIRS artifact |
| India national VIIRS growth (5.3%/yr) is conflated with electrification and reliability | Benchmark comparison uses satellite-derived growth rates only (`benchmarks.py`) | No benchmark for electricity-reliability-adjusted lighting growth |
| Annual cycling in VIIRS data correlates with electricity supply instability | Log-linear trend assumes monotonic exponential growth (`trend.py` line 8--11) | Seasonal reliability cycles create residual structure that violates model assumptions |

---

## Detailed Findings

### Finding M1 — HIGH: VIIRS Radiance Trends in Maharashtra Conflate Lighting Growth with Electricity Reliability Improvements

**Location:** `src/formulas/trend.py` (lines 8--11, 82--87), `src/analysis/benchmark_comparison.py` (lines 35--44)

**Problem:** The codebase fits log-linear trends to annual median VIIRS radiance and interprets the slope as "ALAN growth rate" (annual percent change in artificial light at night). However, Min et al. (2017) and Mann et al. (2016) demonstrate that VIIRS radiance in India is fundamentally a joint signal of (a) installed lighting infrastructure and (b) electricity supply reliability at the ~01:30 AM satellite overpass time. The codebase methodology (`trend.py` line 8: "Log-linear regression log(radiance + epsilon) ~ year models approximately exponential ALAN growth") assumes the signal is purely (a).

Between 2012 and 2024, Maharashtra underwent massive electricity reliability improvements through DDUGJY (launched December 2014), the UJALA LED programme (launched January 2015, with 1.68 crore LEDs distributed in Maharashtra), and Saubhagya (launched October 2017). These programs simultaneously increased the number of electrified households, improved feeder infrastructure reducing outage frequency, and changed the lighting technology (incandescent to LED). All three effects increase VIIRS-observed radiance independently of new *light pollution* per se.

**Impact:** A district showing +8% annual radiance growth may be experiencing +2% actual ALAN growth (new lighting) and +6% from reduced load-shedding and electrification of previously dark households. The codebase cannot distinguish these contributions, making all trend interpretations ambiguous.

**Recommendation:**
1. Add an explicit caveat to all trend outputs: "VIIRS radiance trends in Maharashtra reflect both lighting infrastructure changes and electricity supply reliability improvements. During the study period (2012--2024), major electrification programs (DDUGJY, UJALA, Saubhagya) systematically improved power supply, likely contributing substantially to observed radiance increases (Min et al. 2017; Mann et al. 2016)."
2. Consider adding an `electricity_reliability_adjustment` parameter to `fit_log_linear_trend()` that allows users to supply an estimated reliability contribution for sensitivity analysis.
3. Add electricity-reliability-adjusted benchmarks to `src/formulas/benchmarks.py`.

---

### Finding M2 — HIGH: The Ubiquitous 2016 Breakpoint Is Partially Explained by Electrification Policy, Not Just VIIRS Product Artifacts

**Location:** `src/analysis/breakpoint_analysis.py` (lines 7--30, header documentation)

**Problem:** The breakpoint analysis header correctly identifies two confounded factors for the 34/36 district 2016 breakpoint: (1) VIIRS product evolution (vcmcfg to vcmslcfg transition in 2014), and (2) real-world events including DDUGJY and UJALA. However, the documentation underweights the electrification explanation and frames it primarily as a "VIIRS artifact."

The evidence from the reviewed papers and policy timeline strongly suggests the 2016 breakpoint has a substantial *real-world* electrification component:

- **DDUGJY** was launched in December 2014 and began actual field implementation in 2015--2016 (feeder separation, sub-transmission infrastructure strengthening).
- **UJALA** launched January 2015 and rapidly distributed 1.68 crore LED bulbs in Maharashtra (LEDs are brighter per watt than incandescent, directly increasing VIIRS radiance).
- Maharashtra declared 100% village electrification on April 28, 2018, meaning the most intensive electrification activity occurred in **2015--2017**.
- Min et al. (2017) documented that rural areas experienced 3x worse load-shedding than urban areas. As rural feeders were upgraded through DDUGJY, rural VIIRS radiance would show a step-change upward.

A 2016 breakpoint is therefore *expected* from policy alone, even without VIIRS instrument changes. The codebase header (lines 25--29) recommends treating it primarily as an artifact to be modeled out. This risks removing genuine electrification signals.

**Impact:** If researchers follow the header's recommendation to "separate pre-2014 vs post-2014 periods" or "include VIIRS product version as a covariate," they may inadvertently remove the DDUGJY/UJALA signal along with the instrument artifact, since both transitions occur in the same timeframe. This could lead to underestimating actual ALAN growth.

**Recommendation:**
1. Revise the `breakpoint_analysis.py` header to give equal weight to the electrification explanation. Replace "The 2016 detection is an artifact" with "The 2016 detection reflects both VIIRS product evolution and real-world electrification policy (DDUGJY, UJALA, Saubhagya)."
2. Consider a three-regime model: (a) 2012--2013 (pre-correction, pre-DDUGJY), (b) 2014--2017 (post-correction, active electrification ramp-up), (c) 2018--2024 (post-universal-electrification, steady-state growth).
3. Add a `policy_events` dictionary to `config.py` documenting the electrification program timeline for reference in analyses.

---

### Finding M3 — HIGH: Rural Dark-Sky Site Trends Are Especially Unreliable Without Electricity Reliability Context

**Location:** `src/config.py` (lines 132--155, `DARKSKY_SITES`), `src/formulas/trend.py`, `src/analysis/benchmark_comparison.py`

**Problem:** The codebase defines 11 dark-sky candidate sites (tiger reserves, wildlife sanctuaries, tribal villages, remote locations) and computes VIIRS radiance trends for them. Min et al. (2017) found that rural areas in India experience systematically worse and more variable power supply, with load-shedding of up to 12 hours daily for rural villages versus 4 hours for major cities.

For rural dark-sky sites like "Udmal Tribal Village" (Nashik district) or "Toranmal" (Nandurbar district), the VIIRS radiance time series is dominated by electricity reliability, not lighting infrastructure. A trend showing "+15% annual radiance increase" at a tribal village almost certainly reflects DDUGJY/Saubhagya electrification (the village getting reliable power for the first time) rather than light pollution encroachment.

Nandurbar district, which hosts the Toranmal dark-sky site, is one of India's most underserved tribal areas. It would have been among the last districts to receive reliable power under DDUGJY. An upward VIIRS trend there has a completely different interpretation than the same trend in Pune or Mumbai.

**Impact:** Dark-sky viability assessments based on VIIRS trends at rural sites are unreliable. A site could appear to be "rapidly brightening" when it is actually gaining basic electricity access. Conversely, a genuinely threatened site might show a flat VIIRS trend because lighting growth is masked by declining power reliability.

**Recommendation:**
1. Add an `electrification_context` field to `DARKSKY_SITES` entries (e.g., `"electrification_status": "tribal_area_late_electrification"`) to flag sites where VIIRS trends require special interpretation.
2. When reporting dark-sky site trends, cross-reference against district-level electrification completion dates.
3. Consider excluding pre-electrification years from dark-sky trend analysis for sites in late-electrified areas.

---

### Finding M4 — MEDIUM: Annual Composites Mask Electricity-Driven Seasonal Cycling That Violates Trend Model Assumptions

**Location:** `src/formulas/trend.py` (lines 82--87), `src/config.py` (lines 177--199)

**Problem:** Hsu et al. (2021) demonstrated that Indian VIIRS nighttime lights exhibit pervasive annual cycling driven by unstable power supply. The codebase uses VNL annual composites (which average over this cycling) and fits log-linear trends assuming approximately monotonic exponential growth.

However, the *amplitude* of annual cycling is itself changing over time as electricity reliability improves. In early years (2012--2015), a district with poor power supply would have high seasonal variability (summer load-shedding reduces radiance, winter is more stable). As DDUGJY improves feeders, the seasonal amplitude decreases, meaning the annual composite *average* shifts upward even if the *peak* radiance (representing full lighting with full power) stays constant.

This creates a systematic upward bias in annual composite trends that is entirely an electricity reliability signal, not an ALAN growth signal. The log-linear model in `trend.py` captures this bias as "growth."

**Impact:** Trend slopes for districts with improving electricity reliability (most of rural Maharashtra, 2014--2019) are biased upward. The magnitude of bias depends on the initial severity of load-shedding, which was worst in rural and tribal areas.

**Recommendation:**
1. Add a note to the trend model docstring explaining this limitation for Indian data.
2. Consider supplementing annual median radiance with annual *maximum* radiance (or 90th percentile) as an alternative metric less affected by load-shedding. If maximum radiance is stable while median increases, the change is likely reliability-driven.
3. Add the Hsu et al. (2021) citation to the trend module as a methodological caveat.

---

### Finding M5 — MEDIUM: VIIRS Overpass at 01:30 AM Captures Anomalous Lighting Conditions for Maharashtra

**Location:** `src/formulas/trend.py` (line 8), `src/config.py`

**Problem:** The Suomi NPP satellite crosses the equator at approximately 01:30 local solar time. Over Maharashtra (latitude 15.5--22.1 degrees N), the overpass occurs roughly between midnight and 2:00 AM IST. Mann et al. (2016) explicitly noted this timing "casts doubt on whether electricity reliability estimated during the overpass time is representative of reliability at other times of the day."

At 01:30 AM in Maharashtra:
- Most commercial and industrial lighting is off.
- Street lighting and residential lighting (where power is available) dominate the signal.
- Load-shedding schedules in rural Maharashtra historically concentrate cuts during late night/early morning hours, precisely when VIIRS observes.
- The proportion of lighting types visible at 01:30 AM has changed dramatically: pre-UJALA (2015), late-night lighting was primarily incandescent and sodium vapor. Post-UJALA, LED lighting dominates, with different spectral and directional characteristics.

The codebase interprets VIIRS radiance trends without acknowledging that 01:30 AM is not representative of the full diurnal lighting pattern, and that the lighting *composition* at that hour has changed independently of total lighting stock.

**Impact:** Trend estimates may reflect changes in late-night lighting behavior (e.g., more people keeping lights on later due to reliable power, shift from incandescent to LED) rather than changes in total ALAN burden.

**Recommendation:**
1. Add a comment in `config.py` documenting the VIIRS overpass time and its implications: "VIIRS DNB overpass at ~01:30 local solar time; Maharashtra ALAN metrics reflect late-night/early-morning lighting conditions only."
2. Consider using NASA Black Marble VNP46A products (which provide multiple overpass composites) as a supplementary data source for robustness checks.

---

### Finding M6 — MEDIUM: Benchmark Comparisons Do Not Account for India's Electrification-Driven Growth

**Location:** `src/formulas/benchmarks.py` (lines 11--36), `src/analysis/benchmark_comparison.py` (lines 25--67)

**Problem:** The codebase compares Maharashtra district ALAN growth rates against three benchmarks: global average (2.2%/yr), developing Asia (4.1%/yr), and India national (5.3%/yr). All three are satellite-derived and therefore include the same electrification-reliability conflation described in Finding M1.

The India national benchmark of 5.3%/yr (Li et al. 2020, period 2012--2018) *includes* the massive DDUGJY/UJALA/Saubhagya-driven radiance increase. Comparing Maharashtra against this benchmark will show Maharashtra as "similar" to the national average, which is technically correct but misleading: both the benchmark and the measurement are dominated by the same electrification signal.

For a dark-sky research application, the relevant question is: "How fast is *light pollution* growing?" not "How fast is VIIRS radiance growing due to electrification, LED adoption, and reliability improvements combined?"

**Impact:** The benchmark comparison gives a false sense of validation. A Maharashtra district growing at 5% per year looks "normal" against the India benchmark, but the underlying components might be 1% actual new lighting and 4% electrification/reliability improvement.

**Recommendation:**
1. Add a "mature grid" benchmark from a country with stable electricity (e.g., Japan, Western Europe) where VIIRS trends reflect actual ALAN changes rather than electrification. This would provide a baseline for comparison once Maharashtra's grid stabilizes.
2. Add a caveat to the benchmark comparison output: "India-period benchmarks (2012--2018) include substantial contributions from electrification programs and may not represent steady-state ALAN growth."
3. Consider a post-2019 benchmark (after universal electrification in Maharashtra) as a more representative comparison period.

---

### Finding M7 — MEDIUM: No Covariate for VIIRS Product Version Transition Despite Documented Radiometric Discontinuity

**Location:** `src/config.py` (lines 194--199, `VIIRS_VERSION_MAPPING`), `src/formulas/trend.py`

**Problem:** The codebase correctly maps VIIRS product versions by year (v21 for 2012--2013, v22 for 2014+) in `config.py`, reflecting the transition from vcmcfg (no stray-light correction) to vcmslcfg (stray-light corrected). However, the trend model in `trend.py` fits a single log-linear regression across the entire 2012--2024 period without using this version information as a covariate or dummy variable.

The breakpoint analysis header (lines 14--18) documents that the 2012--2013 data has "baseline noise lower, broad-area radiance systematically fainter" compared to 2014+ data. This creates a step change in the radiance time series that the log-linear model absorbs into the slope estimate, biasing the trend upward.

**Impact:** Trend slopes for the full 2012--2024 period are inflated by the VIIRS product transition. For a 13-year series, including two years of systematically lower radiance at the beginning has an outsized leverage effect on the OLS slope.

**Recommendation:**
1. Add a `viirs_version` dummy variable to `fit_log_linear_trend()` that shifts the intercept for 2012--2013 observations.
2. Alternatively, offer a `start_year` parameter defaulting to 2014 to exclude the pre-stray-light-correction period.
3. This is complementary to (not a substitute for) the electrification covariate recommended in Finding M1.

---

### Finding M8 — LOW: PSI-Type Variability Analysis Would Strengthen Dark-Sky Site Assessments

**Location:** `src/analysis/breakpoint_analysis.py`, `src/analysis/benchmark_comparison.py`

**Problem:** Min et al. (2017) demonstrated that night-to-night variability in light output (PSI) is a powerful indicator of electricity reliability, and that this information is invisible in annual composites. The codebase uses only annual composite data and computes no variability metrics.

For dark-sky site assessments, knowing whether a site has *stable* low radiance (consistent dark sky) versus *variable* low radiance (sometimes dark due to outages, sometimes bright when power is on) is critical. A PSI-type analysis using VIIRS daily DNB data could distinguish genuinely dark sites from sites that appear dark only due to unreliable power.

**Impact:** Some dark-sky candidate sites may have low annual composite radiance because of frequent power outages rather than genuine darkness. These sites would be poor candidates for dark-sky designation since their sky brightness is unreliable.

**Recommendation:**
1. Consider incorporating VIIRS daily DNB data (VNP46A1/VNP46A2) for dark-sky candidate sites to compute radiance variability metrics.
2. A simplified PSI (standard deviation of nightly radiance / mean radiance) could be added as a quality indicator for each dark-sky site.
3. This is a longer-term enhancement and would require significant new data infrastructure.

---

### Finding M9 — LOW: No Acknowledgment of Feeder-Level Spatial Heterogeneity in Electricity Supply

**Location:** `src/config.py` (lines 54--59, `MAHARASHTRA_BBOX`), `src/analysis/breakpoint_analysis.py`

**Problem:** Min et al. (2017) found that power supply irregularity varies at sub-district scales, often following distribution feeder boundaries rather than administrative boundaries. The codebase operates at district level (36 districts) and site level (point-buffer), but does not account for within-district heterogeneity in electricity supply.

Two locations within the same district could have very different VIIRS radiance trajectories if one is on a reliable urban feeder and the other is on a load-shed rural feeder. District-level aggregation (median radiance) blends these signals.

**Impact:** District-level trends may not represent either urban or rural conditions within the district. The median radiance of a district shifts as its worst-served feeders improve, even if no new lighting is installed.

**Recommendation:**
1. Note this limitation in documentation. District-level analysis is the practical minimum for this study but users should be aware of sub-district heterogeneity.
2. For dark-sky sites, the point-buffer approach (10 km radius) partially addresses this by focusing on local conditions rather than district aggregates.

---

## Summary Table

| # | Finding | Severity | Location | Core Issue |
|---|---------|----------|----------|------------|
| M1 | VIIRS trends conflate lighting growth with electricity reliability | HIGH | `trend.py`, `benchmark_comparison.py` | Cannot separate ALAN growth from electrification |
| M2 | 2016 breakpoint reflects real electrification policy, not just VIIRS artifact | HIGH | `breakpoint_analysis.py` header | Documentation underweights policy explanation |
| M3 | Rural dark-sky site trends are unreliable without electrification context | HIGH | `config.py` DARKSKY_SITES, `trend.py` | Tribal/rural sites may show electrification, not light pollution |
| M4 | Annual composites mask seasonal cycling driven by power instability | MEDIUM | `trend.py`, `config.py` | Changing seasonal amplitude biases trend slopes upward |
| M5 | 01:30 AM overpass captures anomalous lighting conditions | MEDIUM | `trend.py`, `config.py` | Late-night observations not representative of full diurnal ALAN |
| M6 | Benchmarks include electrification signal, providing false validation | MEDIUM | `benchmarks.py`, `benchmark_comparison.py` | Maharashtra looks "normal" but benchmark is equally conflated |
| M7 | No VIIRS version covariate despite radiometric discontinuity | MEDIUM | `config.py`, `trend.py` | 2012--2013 data biases trend slopes upward |
| M8 | No variability analysis to distinguish dark sites from outage sites | LOW | `breakpoint_analysis.py` | Cannot tell if low radiance = dark sky or power outage |
| M9 | No feeder-level spatial heterogeneity in district analysis | LOW | `config.py`, `breakpoint_analysis.py` | District median blends heterogeneous feeder conditions |

---

## Key Takeaway

The most consequential insight from this review is that **VIIRS radiance trends in Maharashtra during 2012--2024 cannot be straightforwardly interpreted as ALAN growth rates**. The study period coincides with India's most intensive rural electrification drive in history (DDUGJY, UJALA, Saubhagya), which simultaneously increased the number of electrified households, improved power supply reliability, and changed lighting technology from incandescent to LED. All three factors increase VIIRS-observed radiance independently of new light pollution.

The ubiquitous 2016 breakpoint detected in 34/36 districts is best understood as a superposition of VIIRS product evolution (vcmcfg to vcmslcfg) *and* the initial impact of DDUGJY/UJALA implementation, which began in late 2014/early 2015 and would manifest in VIIRS composites by 2015--2016. Separating these contributions requires either (a) external electrification data as a covariate, or (b) restricting trend analysis to the post-universal-electrification period (2019+) when Maharashtra's grid had largely stabilized.

For the dark-sky research application, this means that sites showing upward VIIRS trends in rural/tribal areas should not automatically be flagged as threatened by light pollution. They may simply be gaining basic electricity access for the first time.

---

## Sources

- [Min, O'Keeffe & Zhang (2017) - World Bank WPS8131](https://documents.worldbank.org/en/publication/documents-reports/documentdetail/125911498758273922/whose-power-gets-cut-using-high-frequency-satellite-images-to-measure-power-supply-irregularity)
- [Mann, Melaas & Malik (2016) - Remote Sensing 8(9), 711](https://www.mdpi.com/2072-4292/8/9/711)
- [Hsu et al. (2021) - Annual Cycling of Nighttime Lights in India](https://www.mdpi.com/2072-4292/13/6/1199)
- [DDUGJY Official Portal](https://recindia.nic.in/ddugjy)
- [Saubhagya Scheme - Ministry of Power](https://powermin.gov.in/en/content/saubhagya)
- [UJALA Scheme - 10 Years of LED Distribution](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2090639)
- [Maharashtra DDUGJY Information - Energy Department](https://energy.maharashtra.gov.in/en/scheme/deen-dayal-upadhyay-gram-jyoti-scheme/)
- [Maharashtra Saubhagya Information - Energy Department](https://energy.maharashtra.gov.in/en/scheme/pradhan-mantri-sahaj-bijli-har-ghar-yojana-saubhagya/)
- [Brian Min - HREA Electricity Data](https://hrea.isr.umich.edu/)
- [Suomi NPP VIIRS Basics - SSEC](https://www.ssec.wisc.edu/suomi_npp/Atmosphere_Team/VIIRS_Basics.html)
