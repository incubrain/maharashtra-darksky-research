"""
Benchmark comparison: Maharashtra ALAN trends vs. published global/regional values.

Validates local results against published literature benchmarks.
"""

from src.logging_config import get_pipeline_logger
import os

import numpy as np
import pandas as pd

from src import config
from src.formulas.benchmarks import (
    PUBLISHED_BENCHMARKS as _PUBLISHED_BENCHMARKS,
    BENCHMARK_INTERPRETATION_THRESHOLD,
)

log = get_pipeline_logger(__name__)

# Re-export from src.formulas.benchmarks for backwards compatibility
PUBLISHED_BENCHMARKS = _PUBLISHED_BENCHMARKS


def compare_to_benchmarks(trends_df, output_csv=None):
    """Compare Maharashtra results to published benchmarks.

    Args:
        trends_df: DataFrame from districts_trends.csv.
        output_csv: Path to save comparison.

    Returns:
        DataFrame with comparison results.
    """
    growth_values = trends_df["annual_pct_change"].dropna()
    mah_median = growth_values.median()
    mah_mean = growth_values.mean()
    mah_min = growth_values.min()
    mah_max = growth_values.max()

    results = []
    for name, benchmark in PUBLISHED_BENCHMARKS.items():
        diff = mah_median - benchmark["annual_growth_pct"]

        if abs(diff) < BENCHMARK_INTERPRETATION_THRESHOLD:
            interpretation = "similar"
        elif diff > 0:
            interpretation = "faster"
        else:
            interpretation = "slower"

        results.append({
            "benchmark_name": name,
            "benchmark_source": benchmark["source"],
            "benchmark_region": benchmark["region"],
            "benchmark_period": benchmark["period"],
            "benchmark_growth_pct": benchmark["annual_growth_pct"],
            "benchmark_ci_low": benchmark.get("ci_low"),
            "benchmark_ci_high": benchmark.get("ci_high"),
            "benchmark_metric_type": benchmark.get("metric_type", "radiance"),
            "maharashtra_median_growth": round(mah_median, 2),
            "maharashtra_mean_growth": round(mah_mean, 2),
            "maharashtra_range_low": round(mah_min, 2),
            "maharashtra_range_high": round(mah_max, 2),
            "difference": round(diff, 2),
            "interpretation": interpretation,
        })

    df = pd.DataFrame(results)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info("Saved benchmark comparison: %s", output_csv)

    # Print summary
    log.info("Benchmark Comparison:")
    for _, row in df.iterrows():
        log.info("  %s (%s): Maharashtra is %s (%.1f%% vs %.1f%%)",
                 row["benchmark_name"], row["benchmark_source"],
                 row["interpretation"], row["maharashtra_median_growth"],
                 row["benchmark_growth_pct"])

    return df
