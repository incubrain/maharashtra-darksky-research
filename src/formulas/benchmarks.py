"""
Published ALAN growth rate benchmarks for comparison.

Citations:
- Kyba, C.C.M. et al. (2017). Artificially lit surface of Earth at night
  increasing in radiance and extent. Science Advances, 3(11), e1701528.
  Origin of the 2.2% global lit-area growth rate.
- Li, X. et al. (2020). A harmonized global nighttime light dataset
  1992-2018. Scientific Data, 7, 168.
- Kyba, C.C.M. et al. (2023). Citizen scientists report global rapid
  reductions in the visibility of stars from 2011 to 2022. Science,
  379(6629), 265-268.
  Ground-based sky brightness growth rate (9.6%/yr).

NOTE on benchmark attribution (finding KY1, review 2026-02-27):
The 2.2% global growth rate was previously misattributed to Elvidge et al.
(2021). It originates from Kyba et al. (2017), Table 1. Kyba reports three
distinct metrics: lit-area growth (2.2%/yr), continuously-lit radiance
growth (2.2%/yr), and total radiance growth (1.8%/yr). The metric_type
field now disambiguates these (finding KY2).
"""

PUBLISHED_BENCHMARKS = {
    "global_lit_area": {
        "source": "Kyba et al. (2017)",
        "region": "Global (all countries)",
        "period": "2012-2016",
        "annual_growth_pct": 2.2,
        "ci_low": 1.8,
        "ci_high": 2.6,
        "metric_type": "lit_area",
        "note": "Growth in area of artificially lit outdoor surface",
    },
    "global_total_radiance": {
        "source": "Kyba et al. (2017)",
        "region": "Global (all countries)",
        "period": "2012-2016",
        "annual_growth_pct": 1.8,
        "ci_low": None,
        "ci_high": None,
        "metric_type": "total_radiance",
        "note": "Growth in total VIIRS-detected radiance (lit area x intensity)",
    },
    "global_ground_based": {
        "source": "Kyba et al. (2023)",
        "region": "Global (citizen science)",
        "period": "2011-2022",
        "annual_growth_pct": 9.6,
        "ci_low": 8.3,
        "ci_high": 10.9,
        "metric_type": "sky_brightness_ground",
        "note": ("Ground-based naked-eye star visibility; ~5x higher than "
                 "VIIRS-detected growth due to LED spectral shift"),
    },
    "developing_asia": {
        "source": "Kyba et al. (2017)",
        "region": "Developing Asia",
        "period": "2012-2016",
        "annual_growth_pct": 4.1,
        "ci_low": 3.5,
        "ci_high": 4.7,
        "metric_type": "lit_area",
    },
    "india_national": {
        "source": "Li et al. (2020)",
        "region": "India (national)",
        "period": "2012-2018",
        "annual_growth_pct": 5.3,
        "ci_low": 4.8,
        "ci_high": 5.8,
        "metric_type": "radiance",
    },
}

# Percentage-point threshold for interpreting growth relative to benchmark.
# |district_growth - benchmark_growth| <= threshold -> "similar".
BENCHMARK_INTERPRETATION_THRESHOLD = 1.0
