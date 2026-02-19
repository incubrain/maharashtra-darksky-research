#!/usr/bin/env python3
"""
Pipeline runner with validation gates and single-step execution.

Orchestrates the full analysis pipeline with:
- Validation between steps (column checks, NaN detection, shape verification)
- ``--step`` CLI flag to run a single step from saved intermediate CSV
- Entity-aware output routing (district/city/site subdirectories)
- Full PipelineRunResult provenance saved as JSON

Usage:
    # Run full pipeline
    python3 -m src.pipeline_runner --pipeline district --years 2012-2024

    # Run single step from saved CSV
    python3 -m src.pipeline_runner --pipeline district --step fit_trends

    # Run site pipeline for dark-sky sites only
    python3 -m src.pipeline_runner --pipeline site --years 2012-2024
"""

import argparse
import json
import logging
import os
import sys
import time

import pandas as pd

from src import config
from src.config import get_entity_dirs
from src.logging_config import StepTimer, get_pipeline_logger, setup_logging
from src.pipeline_types import PipelineRunResult, StepResult

log = get_pipeline_logger(__name__)


# ── Validation helpers ────────────────────────────────────────────────────


def validate_dataframe(df, required_columns, step_name, min_rows=1):
    """Validate a DataFrame between pipeline steps.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required_columns : list[str]
        Columns that must be present.
    step_name : str
        Name of the step that produced this DataFrame (for error messages).
    min_rows : int
        Minimum expected rows.

    Returns
    -------
    list[str]
        List of warning messages (empty if all checks pass).

    Raises
    ------
    ValueError
        If critical validation fails (missing columns).
    """
    warnings_list = []

    if df is None:
        raise ValueError(f"[{step_name}] DataFrame is None")

    if len(df) < min_rows:
        raise ValueError(
            f"[{step_name}] Expected >= {min_rows} rows, got {len(df)}"
        )

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"[{step_name}] Missing required columns: {missing}"
        )

    # Check for NaN in key columns
    for col in required_columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            pct = (nan_count / len(df)) * 100
            warnings_list.append(
                f"[{step_name}] Column '{col}' has {nan_count} NaN values ({pct:.1f}%)"
            )

    return warnings_list


def validate_yearly_radiance(df, step_name="yearly_radiance"):
    """Validate the yearly radiance DataFrame."""
    return validate_dataframe(
        df,
        required_columns=["district", "year", "median_radiance", "mean_radiance"],
        step_name=step_name,
    )


def validate_trends(df, step_name="trends"):
    """Validate the trends DataFrame."""
    return validate_dataframe(
        df,
        required_columns=["district", "annual_pct_change", "r_squared"],
        step_name=step_name,
    )


def validate_site_yearly(df, step_name="site_yearly"):
    """Validate the site yearly DataFrame."""
    return validate_dataframe(
        df,
        required_columns=["name", "year", "median_radiance", "type"],
        step_name=step_name,
    )


def validate_site_trends(df, step_name="site_trends"):
    """Validate the site trends DataFrame."""
    return validate_dataframe(
        df,
        required_columns=["name", "annual_pct_change", "type"],
        step_name=step_name,
    )


# ── District pipeline runner ─────────────────────────────────────────────


DISTRICT_STEPS = [
    "load_boundaries",
    "process_years",
    "save_yearly",
    "fit_trends",
    "basic_maps",
    "stability",
    "breakpoints",
    "trend_diagnostics",
    "quality_diagnostics",
    "benchmark",
    "radial_gradient",
    "light_dome",
    "statewide_viz",
    "graduated_classification",
    "district_reports",
    # Cross-dataset steps (gated by --datasets)
    "load_datasets",
    "merge_datasets",
    "cross_correlation",
    "cross_classification",
    "cross_dataset_reports",
]


def run_district_pipeline(args, single_step=None):
    """Run the district analysis pipeline with validation gates.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    single_step : str, optional
        If provided, run only this step from saved intermediate CSVs.

    Returns
    -------
    PipelineRunResult
    """
    from src.pipeline_steps import (
        step_load_boundaries,
        step_process_years,
        step_save_yearly_radiance,
        step_fit_trends,
        step_stability_analysis,
        step_breakpoint_detection,
        step_trend_diagnostics,
        step_quality_diagnostics,
        step_benchmark_comparison,
        step_radial_gradient_analysis,
        step_light_dome_modeling,
        step_generate_basic_maps,
        step_statewide_visualizations,
        step_graduated_classification,
        step_district_reports,
    )
    import geopandas as gpd

    pipeline_result = PipelineRunResult(
        run_dir=args.output_dir,
        entity_type="district",
    )
    start_time = time.time()

    dirs = get_entity_dirs(args.output_dir, "district")
    csv_dir = dirs["csv"]
    maps_dir = dirs["maps"]
    diagnostics_dir = dirs["diagnostics"]
    reports_dir = dirs["reports"]

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(diagnostics_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Parse years
    if "-" in args.years:
        start, end = args.years.split("-")
        years = list(range(int(start), int(end) + 1))
    else:
        years = [int(y) for y in args.years.split(",")]

    pipeline_result.years_processed = years

    # If running a single step, load data from CSVs
    if single_step:
        return _run_single_district_step(
            single_step, args, years, dirs, pipeline_result
        )

    # ── Full pipeline ────────────────────────────────────────────────

    # Step 1: Load boundaries
    result, gdf = step_load_boundaries(args.shapefile_path)
    pipeline_result.step_results.append(result)
    if not result.ok:
        log.error("Pipeline aborted at load_boundaries: %s", result.error)
        pipeline_result.total_time_seconds = time.time() - start_time
        return pipeline_result

    # Step 2: Process years
    result, yearly_df = step_process_years(
        years, args.viirs_dir, gdf, args.output_dir, args.cf_threshold
    )
    pipeline_result.step_results.append(result)
    if not result.ok:
        log.error("Pipeline aborted at process_years: %s", result.error)
        pipeline_result.total_time_seconds = time.time() - start_time
        return pipeline_result

    # Validation gate: yearly radiance
    try:
        warnings_list = validate_yearly_radiance(yearly_df)
        for w in warnings_list:
            log.warning(w)
    except ValueError as e:
        log.error("Validation failed after process_years: %s", e)
        pipeline_result.total_time_seconds = time.time() - start_time
        return pipeline_result

    # Step 3: Save yearly
    result, yearly_path = step_save_yearly_radiance(yearly_df, csv_dir)
    pipeline_result.step_results.append(result)
    pipeline_result.output_files.append(yearly_path)

    # Step 4: Fit trends
    result, trends_df = step_fit_trends(yearly_df, csv_dir)
    pipeline_result.step_results.append(result)
    if not result.ok:
        log.error("Pipeline aborted at fit_trends: %s", result.error)
        pipeline_result.total_time_seconds = time.time() - start_time
        return pipeline_result

    # Validation gate: trends
    try:
        warnings_list = validate_trends(trends_df)
        for w in warnings_list:
            log.warning(w)
    except ValueError as e:
        log.error("Validation failed after fit_trends: %s", e)
        pipeline_result.total_time_seconds = time.time() - start_time
        return pipeline_result

    # ── Non-critical steps (log warning, continue on failure) ────────

    # Step 5: Basic maps
    result, _ = step_generate_basic_maps(gdf, trends_df, yearly_df, args.output_dir, maps_dir=maps_dir)
    pipeline_result.step_results.append(result)
    if not result.ok:
        log.warning("Basic maps failed: %s", result.error)

    # Step 6: Stability
    result, stability_df = step_stability_analysis(yearly_df, csv_dir, diagnostics_dir)
    pipeline_result.step_results.append(result)
    if not result.ok:
        log.warning("Stability failed: %s", result.error)
        stability_df = None

    # Step 7: Breakpoints
    result, _ = step_breakpoint_detection(yearly_df, csv_dir, diagnostics_dir)
    pipeline_result.step_results.append(result)
    if not result.ok:
        log.warning("Breakpoints failed: %s", result.error)

    # Step 8: Trend diagnostics
    result, _ = step_trend_diagnostics(yearly_df, csv_dir, diagnostics_dir)
    pipeline_result.step_results.append(result)
    if not result.ok:
        log.warning("Trend diagnostics failed: %s", result.error)

    # Step 9: Quality diagnostics
    result, quality_df = step_quality_diagnostics(
        years, args.output_dir, gdf, csv_dir, diagnostics_dir
    )
    pipeline_result.step_results.append(result)
    if not result.ok:
        log.warning("Quality diagnostics failed: %s", result.error)
        quality_df = None

    # Step 10: Benchmark
    result, _ = step_benchmark_comparison(trends_df, csv_dir)
    pipeline_result.step_results.append(result)
    if not result.ok:
        log.warning("Benchmark failed: %s", result.error)

    # Step 11: Radial gradient
    latest_year = max(years)
    result, profiles_df = step_radial_gradient_analysis(
        args.output_dir, latest_year, csv_dir, maps_dir
    )
    pipeline_result.step_results.append(result)
    if not result.ok:
        log.warning("Radial gradient failed: %s", result.error)
        profiles_df = None

    # Step 12: Light dome
    if profiles_df is not None:
        result, _ = step_light_dome_modeling(profiles_df, csv_dir, maps_dir)
        pipeline_result.step_results.append(result)
        if not result.ok:
            log.warning("Light dome failed: %s", result.error)

    # Step 13: Statewide viz
    result, _ = step_statewide_visualizations(
        yearly_df, trends_df, gdf, quality_df, args.output_dir, maps_dir=maps_dir
    )
    pipeline_result.step_results.append(result)
    if not result.ok:
        log.warning("Statewide viz failed: %s", result.error)

    # Step 14: Graduated classification
    result, _ = step_graduated_classification(yearly_df, csv_dir, maps_dir)
    pipeline_result.step_results.append(result)
    if not result.ok:
        log.warning("Graduated classification failed: %s", result.error)

    # Step 15: District reports
    result, _ = step_district_reports(
        yearly_df, trends_df, stability_df, gdf, args.output_dir,
        reports_dir=reports_dir,
    )
    pipeline_result.step_results.append(result)
    if not result.ok:
        log.warning("District reports failed: %s", result.error)

    # ── Cross-dataset steps (only if datasets enabled) ────────────
    from src.dataset_aggregator import get_enabled_datasets, get_dataset_suffix

    enabled_datasets = get_enabled_datasets(args)

    if enabled_datasets:
        from src.cross_dataset_steps import (
            step_load_datasets,
            step_merge_datasets,
            step_cross_correlation,
            step_cross_classification,
            step_cross_dataset_reports,
        )

        suffix = get_dataset_suffix(enabled_datasets)
        vnl_names = yearly_df["district"].unique().tolist()

        log.info("Cross-dataset analysis: %s (suffix: %s)", enabled_datasets, suffix)

        # Step 16: Load datasets
        result, datasets = step_load_datasets(
            enabled_datasets, args, csv_dir, vnl_district_names=vnl_names
        )
        pipeline_result.step_results.append(result)
        if not result.ok:
            log.warning("Load datasets failed: %s", result.error)
        else:
            # Step 17: Merge
            result, merged = step_merge_datasets(
                yearly_df, trends_df, datasets, csv_dir, suffix
            )
            pipeline_result.step_results.append(result)
            if result.ok:
                merged_trends_df = merged["trends"]

                # Step 18: Correlation
                result, corr_df = step_cross_correlation(
                    merged_trends_df, datasets, csv_dir, maps_dir, suffix
                )
                pipeline_result.step_results.append(result)

                # Step 19: Classification
                result, class_df = step_cross_classification(
                    merged_trends_df, datasets, csv_dir, maps_dir, suffix
                )
                pipeline_result.step_results.append(result)

                # Step 20: Reports
                result, _ = step_cross_dataset_reports(
                    merged_trends_df, corr_df, class_df,
                    datasets, reports_dir, maps_dir, suffix
                )
                pipeline_result.step_results.append(result)

    pipeline_result.total_time_seconds = time.time() - start_time
    return pipeline_result


def _run_single_district_step(step_name, args, years, dirs, pipeline_result):
    """Run a single district pipeline step from saved CSVs."""
    from src.pipeline_steps import (
        step_fit_trends,
        step_stability_analysis,
        step_breakpoint_detection,
        step_trend_diagnostics,
        step_benchmark_comparison,
        step_graduated_classification,
    )
    import geopandas as gpd

    csv_dir = dirs["csv"]
    diagnostics_dir = dirs["diagnostics"]
    maps_dir = dirs["maps"]

    yearly_path = os.path.join(csv_dir, "districts_yearly_radiance.csv")
    trends_path = os.path.join(csv_dir, "districts_trends.csv")

    if step_name == "fit_trends":
        if not os.path.exists(yearly_path):
            log.error("Cannot run fit_trends: %s not found", yearly_path)
            return pipeline_result
        yearly_df = pd.read_csv(yearly_path)
        result, trends_df = step_fit_trends(yearly_df, csv_dir)
        pipeline_result.step_results.append(result)

    elif step_name == "stability":
        if not os.path.exists(yearly_path):
            log.error("Cannot run stability: %s not found", yearly_path)
            return pipeline_result
        yearly_df = pd.read_csv(yearly_path)
        result, _ = step_stability_analysis(yearly_df, csv_dir, diagnostics_dir)
        pipeline_result.step_results.append(result)

    elif step_name == "breakpoints":
        if not os.path.exists(yearly_path):
            log.error("Cannot run breakpoints: %s not found", yearly_path)
            return pipeline_result
        yearly_df = pd.read_csv(yearly_path)
        result, _ = step_breakpoint_detection(yearly_df, csv_dir, diagnostics_dir)
        pipeline_result.step_results.append(result)

    elif step_name == "trend_diagnostics":
        if not os.path.exists(yearly_path):
            log.error("Cannot run trend_diagnostics: %s not found", yearly_path)
            return pipeline_result
        yearly_df = pd.read_csv(yearly_path)
        result, _ = step_trend_diagnostics(yearly_df, csv_dir, diagnostics_dir)
        pipeline_result.step_results.append(result)

    elif step_name == "benchmark":
        if not os.path.exists(trends_path):
            log.error("Cannot run benchmark: %s not found", trends_path)
            return pipeline_result
        trends_df = pd.read_csv(trends_path)
        result, _ = step_benchmark_comparison(trends_df, csv_dir)
        pipeline_result.step_results.append(result)

    elif step_name == "graduated_classification":
        if not os.path.exists(yearly_path):
            log.error("Cannot run graduated_classification: %s not found", yearly_path)
            return pipeline_result
        yearly_df = pd.read_csv(yearly_path)
        result, _ = step_graduated_classification(yearly_df, csv_dir, maps_dir)
        pipeline_result.step_results.append(result)

    else:
        log.error(
            "Step '%s' not supported for single-step execution. "
            "Available: fit_trends, stability, breakpoints, trend_diagnostics, "
            "benchmark, graduated_classification",
            step_name,
        )

    return pipeline_result


# ── Main entry point ─────────────────────────────────────────────────────


def save_pipeline_result(pipeline_result, output_dir):
    """Save PipelineRunResult as JSON for provenance."""
    result_path = os.path.join(output_dir, "pipeline_run.json")
    with open(result_path, "w") as f:
        json.dump(pipeline_result.to_dict(), f, indent=2, default=str)
    log.info("Pipeline result saved: %s", result_path)
    return result_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline runner with validation gates"
    )
    parser.add_argument(
        "--pipeline",
        choices=["district", "city", "site", "all"],
        default="district",
        help="Which pipeline to run",
    )
    parser.add_argument(
        "--step",
        default=None,
        help="Run a single step from saved CSVs (e.g., 'fit_trends')",
    )
    parser.add_argument(
        "--viirs-dir",
        default=config.DEFAULT_VIIRS_DIR,
        help="Root directory containing year folders with .gz files",
    )
    parser.add_argument(
        "--shapefile-path",
        default=config.DEFAULT_SHAPEFILE_PATH,
        help="Path to Maharashtra district boundaries (GeoJSON)",
    )
    parser.add_argument(
        "--output-dir",
        default=config.DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--cf-threshold",
        type=int,
        default=config.CF_COVERAGE_THRESHOLD,
        help="Cloud-free coverage threshold",
    )
    parser.add_argument(
        "--years",
        default="2012-2024",
        help="Year range or comma-separated years",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset names to enable (e.g. 'census_2011_pca') "
             "or 'all' to enable all configured datasets. Overrides config.py.",
    )
    parser.add_argument(
        "--census-dir",
        default=None,
        dest="census_dir",
        help="Override census data directory (default: from config.py)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve output dir
    base_dir = args.output_dir
    latest_dir = os.path.join(base_dir, "latest")
    if os.path.isdir(latest_dir):
        args.output_dir = latest_dir
        log.info("Using latest run directory: %s", args.output_dir)

    # Set up structured logging
    setup_logging(run_dir=args.output_dir)

    log.info("Pipeline runner: %s", args.pipeline)
    if args.step:
        log.info("Single step mode: %s", args.step)

    pipelines = (
        ["district", "city", "site"] if args.pipeline == "all" else [args.pipeline]
    )

    for pipeline_type in pipelines:
        log.info("=" * 60)
        log.info("Running %s pipeline", pipeline_type)
        log.info("=" * 60)

        if pipeline_type == "district":
            result = run_district_pipeline(args, single_step=args.step)
        elif pipeline_type in ("city", "site"):
            # Delegate to site_analysis with --type flag
            log.info(
                "Use: python3 -m src.site_analysis --type %s", pipeline_type
            )
            result = PipelineRunResult(
                run_dir=args.output_dir,
                entity_type=pipeline_type,
            )
            result.step_results.append(
                StepResult(
                    step_name="delegate",
                    status="skipped",
                    output_summary={
                        "message": f"Use python3 -m src.site_analysis --type {pipeline_type}"
                    },
                )
            )

        # Save provenance
        save_pipeline_result(result, args.output_dir)

        # Summary
        log.info("Pipeline %s complete in %.1fs", pipeline_type, result.total_time_seconds)
        if result.failed_steps:
            log.warning(
                "Failed steps: %s",
                [s.step_name for s in result.failed_steps],
            )
        else:
            log.info("All steps succeeded.")


if __name__ == "__main__":
    main()
