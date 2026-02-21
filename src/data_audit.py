"""
Consolidated data audit module for VIIRS raster diagnostics.

Merges the functionality of the former debug_audit_all_years.py,
debug_deep_dive.py, and debug_stats.py into a formal, reusable module
that integrates with the pipeline logging and StepResult infrastructure.

Usage (standalone):
    python -m src.data_audit --viirs-dir ./outputs --years 2012-2024
    python -m src.data_audit --viirs-dir ./outputs --years 2012,2020,2024 --histograms

Usage (as pipeline step):
    from src.data_audit import audit_raster_statistics
    results_df = audit_raster_statistics(years, subset_root)
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.logging_config import StepTimer, get_pipeline_logger, log_step_summary
from src.pipeline_types import StepResult

log = get_pipeline_logger(__name__)


def audit_raster_statistics(years, subset_root, output_dir=None):
    """Audit raster statistics across all available years.

    For each year, reads the median raster and computes distributional
    statistics on valid (finite, >0) pixels to identify background floor
    shifts, data quality issues, and temporal anomalies.

    Parameters
    ----------
    years : list[int]
        Years to audit.
    subset_root : str
        Root directory containing ``{year}/maharashtra_median_{year}.tif``.
    output_dir : str, optional
        Directory to save CSV report. If None, no file is written.

    Returns
    -------
    pd.DataFrame
        One row per year with columns: year, n_valid, min, p01, p05,
        median, p95, p99, max, mean, std, pct_below_025, pct_below_050.
    """
    import rasterio

    audit_results = []

    for year in years:
        median_path = os.path.join(
            subset_root, str(year), f"maharashtra_median_{year}.tif"
        )
        if not os.path.exists(median_path):
            log.debug("Skipping year %d: %s not found", year, median_path)
            continue

        with rasterio.open(median_path) as src:
            data = src.read(1)

        valid_mask = np.isfinite(data) & (data > 0)
        valid_data = data[valid_mask]

        if len(valid_data) == 0:
            log.warning("Year %d has no valid data > 0", year)
            continue

        row = {
            "year": year,
            "n_valid": len(valid_data),
            "min": float(np.min(valid_data)),
            "p01": float(np.percentile(valid_data, 1)),
            "p05": float(np.percentile(valid_data, 5)),
            "median": float(np.median(valid_data)),
            "p95": float(np.percentile(valid_data, 95)),
            "p99": float(np.percentile(valid_data, 99)),
            "max": float(np.max(valid_data)),
            "mean": float(np.mean(valid_data)),
            "std": float(np.std(valid_data)),
            "pct_below_025": float(
                (np.sum(valid_data < 0.25) / len(valid_data)) * 100
            ),
            "pct_below_050": float(
                (np.sum(valid_data < 0.50) / len(valid_data)) * 100
            ),
        }

        log.info(
            "Year %d | n=%d | Min: %.4f | P01: %.4f | Median: %.4f | "
            "<0.25: %.1f%% | <0.50: %.1f%%",
            year, row["n_valid"], row["min"], row["p01"], row["median"],
            row["pct_below_025"], row["pct_below_050"],
        )
        audit_results.append(row)

    df = pd.DataFrame(audit_results)

    if output_dir and not df.empty:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "data_audit_report.csv")
        df.to_csv(csv_path, index=False)
        log.info("Audit report saved: %s", csv_path)

    return df


def audit_radiance_distributions(years, subset_root, output_dir=None, max_radiance=2.0):
    """Plot radiance distributions to identify background shifts and noise.

    Generates a histogram overlay comparing low-radiance distributions
    across years — useful for diagnosing the rising DBS background floor.

    Parameters
    ----------
    years : list[int]
        Years to plot (recommend 3-5 for readability).
    subset_root : str
        Root directory containing yearly raster subsets.
    output_dir : str, optional
        Directory to save histogram PNG.
    max_radiance : float
        Upper radiance limit for the histogram (filters to [0, max_radiance]).

    Returns
    -------
    str or None
        Path to saved histogram PNG, or None if no data.
    """
    import rasterio

    fig, ax = plt.subplots(figsize=(12, 6))
    any_data = False

    for year in years:
        path = os.path.join(
            subset_root, str(year), f"maharashtra_median_{year}.tif"
        )
        if not os.path.exists(path):
            log.warning("Path missing: %s", path)
            continue

        with rasterio.open(path) as src:
            data = src.read(1).flatten()

        valid = data[(data > 0) & (data < max_radiance) & np.isfinite(data)]
        if len(valid) == 0:
            continue

        ax.hist(valid, bins=100, alpha=0.5, label=f"Year {year}", density=True)
        any_data = True

    if not any_data:
        plt.close(fig)
        return None

    ax.axvline(0.25, color="r", linestyle="--", label="Threshold (0.25 nW)")
    ax.set_title(f"Radiance Distribution (0.0 – {max_radiance} nW/cm²/sr)")
    ax.set_xlabel("Radiance (nW/cm²/sr)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "radiance_distribution_histograms.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        log.info("Histogram saved: %s", out_path)

    plt.close(fig)
    return out_path


def audit_threshold_sensitivity(years, subset_root, output_dir=None,
                                 thresholds=None):
    """Test how different low-radiance thresholds affect pixel counts.

    For each year and threshold, compute the percentage of valid pixels
    that would be filtered out. Helps tune DARKNESS_THRESHOLD_NW.

    Parameters
    ----------
    years : list[int]
    subset_root : str
    output_dir : str, optional
    thresholds : list[float], optional
        Thresholds to test. Default: [0.1, 0.2, 0.25, 0.3, 0.5, 1.0].

    Returns
    -------
    pd.DataFrame
        Columns: year, threshold, pct_below, n_below, n_total.
    """
    import rasterio

    if thresholds is None:
        thresholds = [0.1, 0.2, 0.25, 0.3, 0.5, 1.0]

    rows = []
    for year in years:
        path = os.path.join(
            subset_root, str(year), f"maharashtra_median_{year}.tif"
        )
        if not os.path.exists(path):
            continue

        with rasterio.open(path) as src:
            data = src.read(1)

        valid = data[np.isfinite(data) & (data > 0)]
        if len(valid) == 0:
            continue

        for thresh in thresholds:
            n_below = int(np.sum(valid < thresh))
            rows.append({
                "year": year,
                "threshold": thresh,
                "pct_below": (n_below / len(valid)) * 100,
                "n_below": n_below,
                "n_total": len(valid),
            })

    df = pd.DataFrame(rows)

    if output_dir and not df.empty:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "threshold_sensitivity.csv")
        df.to_csv(csv_path, index=False)
        log.info("Threshold sensitivity report saved: %s", csv_path)

    return df


# ── Pipeline step wrapper ───────────────────────────────────────────────


def step_data_audit(years, output_dir, audit_output_dir=None):
    """Pipeline step wrapper for data audit.

    Parameters
    ----------
    years : list[int]
    output_dir : str
        Pipeline output directory containing ``subsets/``.
    audit_output_dir : str, optional
        Where to save audit outputs. Defaults to ``{output_dir}/diagnostics/data_audit/``.

    Returns
    -------
    tuple[StepResult, pd.DataFrame | None]
    """
    subset_root = os.path.join(output_dir, "subsets")
    if audit_output_dir is None:
        audit_output_dir = os.path.join(output_dir, "diagnostics", "data_audit")

    error_tb = None
    audit_df = None

    with StepTimer() as timer:
        try:
            audit_df = audit_raster_statistics(years, subset_root, audit_output_dir)
            audit_radiance_distributions(
                years, subset_root, audit_output_dir
            )
            audit_threshold_sensitivity(
                years, subset_root, audit_output_dir
            )
        except Exception as exc:
            import traceback
            error_tb = traceback.format_exc()
            log.error("data_audit failed: %s", exc, exc_info=True)

    if error_tb:
        log_step_summary(log, "data_audit", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="data_audit",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    n_years = len(audit_df) if audit_df is not None else 0
    log_step_summary(
        log, "data_audit", "success",
        input_summary={"years": len(years)},
        output_summary={"years_audited": n_years},
        timing_seconds=timer.elapsed,
    )
    return StepResult(
        step_name="data_audit",
        status="success",
        input_summary={"years": len(years)},
        output_summary={"years_audited": n_years},
        timing_seconds=timer.elapsed,
    ), audit_df


# ── CLI entry point ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="VIIRS raster data audit and diagnostics"
    )
    parser.add_argument(
        "--viirs-dir",
        default=config.DEFAULT_OUTPUT_DIR,
        help="Output directory containing subsets/ (default: ./outputs)",
    )
    parser.add_argument(
        "--years",
        default="2012-2024",
        help="Year range (e.g. 2012-2024) or comma-separated list",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for audit reports (default: {viirs_dir}/diagnostics/data_audit)",
    )
    parser.add_argument(
        "--histograms",
        action="store_true",
        default=False,
        help="Generate radiance distribution histograms",
    )
    args = parser.parse_args()

    # Parse years
    if "-" in args.years:
        start, end = args.years.split("-")
        years = list(range(int(start), int(end) + 1))
    else:
        years = [int(y) for y in args.years.split(",")]

    subset_root = os.path.join(args.viirs_dir, "subsets")
    output_dir = args.output_dir or os.path.join(
        args.viirs_dir, "diagnostics", "data_audit"
    )

    audit_raster_statistics(years, subset_root, output_dir)

    if args.histograms:
        audit_radiance_distributions(years, subset_root, output_dir)

    audit_threshold_sensitivity(years, subset_root, output_dir)


if __name__ == "__main__":
    main()
