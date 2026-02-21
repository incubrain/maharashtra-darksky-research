#!/usr/bin/env python3
"""
Pipeline runner with validation gates and single-step execution.

Orchestrates the full analysis pipeline with:
- Pandera schema validation between steps (structure + data quality)
- NaN propagation tracking between steps
- ``--step`` CLI flag to run a single step from saved intermediate CSV
- ``--strict-validation`` flag to abort on schema violations
- Entity-aware output routing (district/city/site subdirectories)
- Full PipelineRunResult provenance saved as JSON
- Diagnostic report generation at the end of each run

Usage:
    # Run full pipeline
    python3 -m src.pipeline_runner --pipeline district --years 2012-2024

    # Run single step from saved CSV
    python3 -m src.pipeline_runner --pipeline district --step fit_trends

    # Run site pipeline for dark-sky sites only
    python3 -m src.pipeline_runner --pipeline site --years 2012-2024

    # Run with strict validation (abort on schema violations)
    python3 -m src.pipeline_runner --pipeline district --strict-validation
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
from src.logging_config import StepTimer, get_pipeline_logger, setup_logging, set_run_id
from src.pipeline_types import PipelineRunResult, StepResult
from src.schemas import (
    YearlyRadianceSchema,
    TrendsSchema,
    SiteYearlySchema,
    SiteTrendsSchema,
    validate_schema,
)

log = get_pipeline_logger(__name__)


# ── NaN tracking helper ──────────────────────────────────────────────────


def track_nan_counts(df, step_name, prev_nan_counts=None):
    """Track NaN counts per column and warn on propagation.

    Parameters
    ----------
    df : pd.DataFrame or None
        DataFrame to inspect.
    step_name : str
        Pipeline step name for logging.
    prev_nan_counts : dict or None
        NaN counts from the previous step for delta comparison.

    Returns
    -------
    dict
        Column → NaN count mapping for this step.
    """
    if df is None:
        return {}

    nan_counts = df.isna().sum().to_dict()
    nan_counts = {k: int(v) for k, v in nan_counts.items() if v > 0}

    if nan_counts:
        log.debug(
            "[%s] NaN counts: %s",
            step_name,
            nan_counts,
            extra={"step_name": step_name, "nan_summary": nan_counts},
        )

    # Warn if NaN count increased compared to previous step
    if prev_nan_counts:
        for col, count in nan_counts.items():
            prev = prev_nan_counts.get(col, 0)
            if count > prev:
                log.warning(
                    "[%s] NaN count increased for '%s': %d → %d (+%d)",
                    step_name, col, prev, count, count - prev,
                )

    return nan_counts


# ── Validation helpers (legacy wrappers using Pandera) ───────────────────


def validate_yearly_radiance(df, step_name="yearly_radiance", strict=False):
    """Validate the yearly radiance DataFrame using Pandera schema."""
    return validate_schema(df, YearlyRadianceSchema, step_name, strict=strict)


def validate_trends(df, step_name="trends", strict=False):
    """Validate the trends DataFrame using Pandera schema."""
    return validate_schema(df, TrendsSchema, step_name, strict=strict)


def validate_site_yearly(df, step_name="site_yearly", strict=False):
    """Validate the site yearly DataFrame using Pandera schema."""
    return validate_schema(df, SiteYearlySchema, step_name, strict=strict)


def validate_site_trends(df, step_name="site_trends", strict=False):
    """Validate the site trends DataFrame using Pandera schema."""
    return validate_schema(df, SiteTrendsSchema, step_name, strict=strict)


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
    "animation_frames",
    "per_district_radiance_maps",
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
        step_animation_frames,
        step_per_district_radiance_maps,
    )
    import geopandas as gpd

    strict = getattr(args, "strict_validation", False)

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

    # NaN tracking state
    prev_nan_counts = {}

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

    # Validation gate: yearly radiance (Pandera)
    try:
        warnings_list = validate_yearly_radiance(yearly_df, strict=strict)
        for w in warnings_list:
            log.warning(w)
    except ValueError as e:
        log.error("Validation failed after process_years: %s", e)
        pipeline_result.total_time_seconds = time.time() - start_time
        return pipeline_result

    # NaN tracking after process_years
    prev_nan_counts = track_nan_counts(yearly_df, "process_years")

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

    # Validation gate: trends (Pandera)
    try:
        warnings_list = validate_trends(trends_df, strict=strict)
        for w in warnings_list:
            log.warning(w)
    except ValueError as e:
        log.error("Validation failed after fit_trends: %s", e)
        pipeline_result.total_time_seconds = time.time() - start_time
        return pipeline_result

    # NaN tracking after fit_trends
    prev_nan_counts = track_nan_counts(trends_df, "fit_trends", prev_nan_counts)

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

    # Step 16: Animation frames (sprawl, differential, darkness, trend map)
    result, _ = step_animation_frames(years, args.output_dir, gdf, maps_dir=maps_dir)
    pipeline_result.step_results.append(result)
    if not result.ok:
        log.warning("Animation frames failed: %s", result.error)

    # Step 17: Per-district radiance maps
    result, _ = step_per_district_radiance_maps(
        args.output_dir, max(years), gdf, maps_dir=maps_dir,
    )
    pipeline_result.step_results.append(result)
    if not result.ok:
        log.warning("Per-district radiance maps failed: %s", result.error)

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

    # ── Generate diagnostic report ───────────────────────────────────
    try:
        from src.outputs.diagnostic_report import generate_diagnostic_report

        report_path = generate_diagnostic_report(
            pipeline_result,
            yearly_df=yearly_df,
            trends_df=trends_df,
            output_dir=args.output_dir,
        )
        if report_path:
            log.info("Diagnostic report generated: %s", report_path)
    except Exception as exc:
        log.warning("Diagnostic report generation failed: %s", exc)

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
        step_animation_frames,
        step_per_district_radiance_maps,
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

    elif step_name == "animation_frames":
        gdf = gpd.read_file(args.shapefile_path)
        result, _ = step_animation_frames(years, args.output_dir, gdf, maps_dir=maps_dir)
        pipeline_result.step_results.append(result)

    elif step_name == "per_district_radiance_maps":
        gdf = gpd.read_file(args.shapefile_path)
        result, _ = step_per_district_radiance_maps(
            args.output_dir, max(years), gdf, maps_dir=maps_dir,
        )
        pipeline_result.step_results.append(result)

    else:
        log.error(
            "Step '%s' not supported for single-step execution. "
            "Available: fit_trends, stability, breakpoints, trend_diagnostics, "
            "benchmark, graduated_classification, animation_frames, "
            "per_district_radiance_maps",
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
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        default=False,
        dest="strict_validation",
        help="Abort pipeline on schema validation failures (default: warn only)",
    )
    parser.add_argument(
        "--compare-run",
        default=None,
        dest="compare_run",
        help="Path to a previous run directory to compare against",
    )
    parser.add_argument(
        "--buffer-km",
        type=float,
        default=config.SITE_BUFFER_RADIUS_KM,
        dest="buffer_km",
        help="Buffer radius in km for site/city analysis (default: from config)",
    )
    parser.add_argument(
        "--city-source",
        choices=["config", "census"],
        default="config",
        dest="city_source",
        help="City locations source: 'config' (hand-picked) or 'census' (geocoded census towns)",
    )
    return parser.parse_args()


# ── City / Site pipeline runner ──────────────────────────────────────────


def run_entity_pipeline(args, entity_type):
    """Run city or site pipeline and return PipelineRunResult.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (must include output_dir, buffer_km,
        cf_threshold, shapefile_path, years).
    entity_type : str
        "city" or "site".

    Returns
    -------
    PipelineRunResult
    """
    import traceback
    from src.site.site_analysis import _run_entity_pipeline

    pipeline_result = PipelineRunResult(
        run_dir=args.output_dir,
        entity_type=entity_type,
    )
    start_time = time.time()

    # Parse years
    if "-" in args.years:
        start, end = args.years.split("-")
        years = list(range(int(start), int(end) + 1))
    else:
        years = [int(y) for y in args.years.split(",")]

    pipeline_result.years_processed = years

    city_source = getattr(args, "city_source", "config")

    try:
        steps = _run_entity_pipeline(args, years, entity_type, city_source=city_source)
        if steps:
            pipeline_result.step_results.extend(steps)
    except Exception as exc:
        pipeline_result.step_results.append(
            StepResult(
                step_name=f"{entity_type}_pipeline",
                status="error",
                error=traceback.format_exc(),
            )
        )
        log.error("%s pipeline failed: %s", entity_type, exc)

    pipeline_result.total_time_seconds = time.time() - start_time
    return pipeline_result


# ── Main entry point ─────────────────────────────────────────────────────


def main():
    args = parse_args()

    # Generate a fresh run_id for this pipeline invocation
    run_id = set_run_id()

    # Resolve output dir
    base_dir = args.output_dir
    latest_dir = os.path.join(base_dir, "latest")
    if os.path.isdir(latest_dir):
        args.output_dir = latest_dir
        log.info("Using latest run directory: %s", args.output_dir)

    # Set up structured logging
    setup_logging(run_dir=args.output_dir)

    log.info("Pipeline runner: %s (run_id=%s)", args.pipeline, run_id)
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
            result = run_entity_pipeline(args, pipeline_type)

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
