"""
Published ALAN growth rate benchmarks for comparison.

Citations:
- Elvidge, C.D. et al. (2021). Annual time series of global VIIRS
  nighttime lights. Remote Sensing, 13(5), 922.
- Li, X. et al. (2020). A harmonized global nighttime light dataset
  1992-2018. Scientific Data, 7, 168.
"""

PUBLISHED_BENCHMARKS = {
    "global_average": {
        "source": "Elvidge et al. (2021)",
        "region": "Global (all countries)",
        "period": "2012-2019",
        "annual_growth_pct": 2.2,
        "ci_low": 1.8,
        "ci_high": 2.6,
    },
    "developing_asia": {
        "source": "Elvidge et al. (2021)",
        "region": "Developing Asia",
        "period": "2012-2019",
        "annual_growth_pct": 4.1,
        "ci_low": 3.5,
        "ci_high": 4.7,
    },
    "india_national": {
        "source": "Li et al. (2020)",
        "region": "India (national)",
        "period": "2012-2018",
        "annual_growth_pct": 5.3,
        "ci_low": 4.8,
        "ci_high": 5.8,
    },
}

# Percentage-point threshold for interpreting growth relative to benchmark.
# |district_growth - benchmark_growth| <= threshold → "similar".
BENCHMARK_INTERPRETATION_THRESHOLD = 1.0
