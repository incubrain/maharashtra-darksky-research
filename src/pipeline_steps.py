"""
District pipeline step functions.

Each function is a discrete, testable pipeline step with explicit
inputs/outputs and StepResult tracking.
"""

import logging
import os
import sys
import traceback

import geopandas as gpd
import numpy as np
import pandas as pd

from src.pipeline_types import StepResult
from src.logging_config import StepTimer, get_pipeline_logger, log_step_summary

log = get_pipeline_logger(__name__)


def step_load_boundaries(shapefile_path: str) -> tuple[StepResult, gpd.GeoDataFrame | None]:
    """Load and validate district boundaries."""
    from src import config
    from src.validate_names import validate_or_exit

    gdf = None
    error_tb = None

    with StepTimer() as timer:
        try:
            gdf = gpd.read_file(shapefile_path)
            log.info("Loaded boundaries: %d districts", len(gdf))

            # Validate district count
            if len(gdf) != config.EXPECTED_DISTRICT_COUNT:
                log.warning(
                    "Expected %d districts but found %d in %s",
                    config.EXPECTED_DISTRICT_COUNT, len(gdf), shapefile_path,
                )

            # Validate district names
            validate_or_exit(shapefile_path, check_config=True)

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("load_boundaries failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("load_boundaries failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "load_boundaries", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="load_boundaries",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "load_boundaries", "success", input_summary={"shapefile_path": shapefile_path}, output_summary={"districts": len(gdf)}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="load_boundaries",
        status="success",
        input_summary={"shapefile_path": shapefile_path},
        output_summary={"districts": len(gdf)},
        timing_seconds=timer.elapsed,
    ), gdf


def step_process_years(
    years: list[int],
    viirs_dir: str,
    gdf: gpd.GeoDataFrame,
    output_dir: str,
    cf_threshold: int,
) -> tuple[StepResult, pd.DataFrame | None]:
    """Process all years and concatenate results."""
    from src.viirs_process import process_single_year

    all_yearly = []
    error_tb = None

    with StepTimer() as timer:
        try:
            for year in years:
                df = process_single_year(year, viirs_dir, gdf, output_dir, cf_threshold)
                if df is not None:
                    all_yearly.append(df)

            if not all_yearly:
                error_tb = "No data processed for any year"
                log.error(error_tb)
            else:
                yearly_df = pd.concat(all_yearly, ignore_index=True)
                log.info(
                    "Total records: %d (%d years × %d districts)",
                    len(yearly_df),
                    yearly_df["year"].nunique(),
                    yearly_df["district"].nunique(),
                )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("process_years failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("process_years failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "process_years", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="process_years",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "process_years", "success", input_summary={"years": years, "viirs_dir": viirs_dir}, output_summary={"total_records": len(yearly_df), "years_processed": yearly_df["year"].nunique(), "districts": yearly_df["district"].nunique()}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="process_years",
        status="success",
        input_summary={"years": years, "viirs_dir": viirs_dir},
        output_summary={
            "total_records": len(yearly_df),
            "years_processed": yearly_df["year"].nunique(),
            "districts": yearly_df["district"].nunique(),
        },
        timing_seconds=timer.elapsed,
    ), yearly_df


def step_save_yearly_radiance(yearly_df: pd.DataFrame, csv_dir: str) -> tuple[StepResult, str]:
    """Save yearly radiance data to CSV."""
    result_path = None
    error_tb = None

    with StepTimer() as timer:
        try:
            os.makedirs(csv_dir, exist_ok=True)
            result_path = os.path.join(csv_dir, "districts_yearly_radiance.csv")
            yearly_df.to_csv(result_path, index=False)
            log.info("Saved yearly data: %s", result_path)

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("save_yearly_radiance failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("save_yearly_radiance failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "save_yearly_radiance", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="save_yearly_radiance",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "save_yearly_radiance", "success", input_summary={"records": len(yearly_df)}, output_summary={"csv_path": result_path}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="save_yearly_radiance",
        status="success",
        input_summary={"records": len(yearly_df)},
        output_summary={"csv_path": result_path},
        timing_seconds=timer.elapsed,
    ), result_path


def step_fit_trends(yearly_df: pd.DataFrame, csv_dir: str) -> tuple[StepResult, pd.DataFrame | None]:
    """Fit log-linear trends per district and classify ALAN levels."""
    from src.viirs_process import fit_log_linear_trend
    from src.formulas.classification import classify_alan

    trends_df = None
    error_tb = None

    with StepTimer() as timer:
        try:
            districts = yearly_df["district"].unique()
            trend_results = []

            for d in districts:
                result = fit_log_linear_trend(yearly_df, d)

                # Add latest year radiance
                latest = yearly_df[yearly_df["district"] == d].sort_values("year").iloc[-1]
                result["mean_radiance_latest"] = latest["mean_radiance"]
                result["median_radiance_latest"] = latest["median_radiance"]

                # Classify ALAN level
                med = latest["median_radiance"]
                result["alan_class"] = classify_alan(med)

                trend_results.append(result)

            trends_df = pd.DataFrame(trend_results)

            # Save to CSV
            os.makedirs(csv_dir, exist_ok=True)
            trends_path = os.path.join(csv_dir, "districts_trends.csv")
            trends_df.to_csv(trends_path, index=False)
            log.info("Saved trends: %s", trends_path)

            # Log summary
            log.info("\n" + "=" * 60)
            log.info("TREND SUMMARY")
            log.info("=" * 60)
            for _, row in trends_df.sort_values("annual_pct_change", ascending=False).iterrows():
                log.info(
                    "%-20s %+6.2f%% [%+.2f, %+.2f] | %.2f nW (%s)",
                    row["district"],
                    row["annual_pct_change"],
                    row["ci_low"],
                    row["ci_high"],
                    row["median_radiance_latest"],
                    row["alan_class"],
                )

            low_alan = trends_df[trends_df["alan_class"] == "low"]
            log.info(
                "\nLow-ALAN districts (median < 1 nW/cm²/sr): %s",
                ", ".join(low_alan["district"].tolist()) if len(low_alan) > 0 else "None",
            )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("fit_trends failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("fit_trends failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "fit_trends", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="fit_trends",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "fit_trends", "success", input_summary={"districts": len(districts)}, output_summary={"trends_computed": len(trends_df)}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="fit_trends",
        status="success",
        input_summary={"districts": len(districts)},
        output_summary={"trends_computed": len(trends_df)},
        timing_seconds=timer.elapsed,
    ), trends_df


def step_stability_analysis(
    yearly_df: pd.DataFrame, csv_dir: str, diagnostics_dir: str
) -> tuple[StepResult, pd.DataFrame | None]:
    """Compute temporal stability metrics for each district."""
    from src.stability_metrics import compute_stability_metrics, plot_stability_scatter

    stability_df = None
    error_tb = None

    with StepTimer() as timer:
        try:
            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(diagnostics_dir, exist_ok=True)

            stability_df = compute_stability_metrics(
                yearly_df,
                entity_col="district",
                output_csv=os.path.join(csv_dir, "district_stability_metrics.csv"),
            )

            plot_stability_scatter(
                stability_df,
                entity_col="district",
                output_path=os.path.join(diagnostics_dir, "stability_scatter.png"),
            )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("stability_analysis failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("stability_analysis failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "stability_analysis", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="stability_analysis",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "stability_analysis", "success", input_summary={"districts": yearly_df["district"].nunique()}, output_summary={"metrics_computed": len(stability_df) if stability_df is not None else 0}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="stability_analysis",
        status="success",
        input_summary={"districts": yearly_df["district"].nunique()},
        output_summary={"metrics_computed": len(stability_df) if stability_df is not None else 0},
        timing_seconds=timer.elapsed,
    ), stability_df


def step_breakpoint_detection(
    yearly_df: pd.DataFrame, csv_dir: str, diagnostics_dir: str
) -> tuple[StepResult, pd.DataFrame | None]:
    """Detect temporal breakpoints in radiance trends."""
    from src.breakpoint_analysis import analyze_all_breakpoints, plot_breakpoint_timeline

    breakpoints_df = None
    error_tb = None

    with StepTimer() as timer:
        try:
            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(diagnostics_dir, exist_ok=True)

            breakpoints_df = analyze_all_breakpoints(
                yearly_df,
                entity_col="district",
                output_csv=os.path.join(csv_dir, "district_breakpoints.csv"),
            )

            plot_breakpoint_timeline(
                breakpoints_df,
                output_path=os.path.join(diagnostics_dir, "breakpoint_timeline.png"),
            )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("breakpoint_detection failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("breakpoint_detection failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "breakpoint_detection", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="breakpoint_detection",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "breakpoint_detection", "success", input_summary={"districts": yearly_df["district"].nunique()}, output_summary={"breakpoints_found": len(breakpoints_df) if breakpoints_df is not None else 0}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="breakpoint_detection",
        status="success",
        input_summary={"districts": yearly_df["district"].nunique()},
        output_summary={"breakpoints_found": len(breakpoints_df) if breakpoints_df is not None else 0},
        timing_seconds=timer.elapsed,
    ), breakpoints_df


def step_trend_diagnostics(
    yearly_df: pd.DataFrame, csv_dir: str, diagnostics_dir: str
) -> tuple[StepResult, pd.DataFrame | None]:
    """Compute trend model diagnostics and generate plots for flagged districts."""
    from src.trend_diagnostics import compute_all_diagnostics, plot_diagnostic_panel

    diag_df = None
    error_tb = None

    with StepTimer() as timer:
        try:
            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(diagnostics_dir, exist_ok=True)

            diag_df = compute_all_diagnostics(
                yearly_df,
                entity_col="district",
                output_csv=os.path.join(csv_dir, "trend_diagnostics.csv"),
            )

            # Generate diagnostic plots for flagged districts
            for _, row in diag_df.iterrows():
                r2 = row.get("r_squared", 1.0)
                outliers = row.get("outlier_years", "")
                n_outliers = len(outliers.split("; ")) if outliers else 0

                if (not np.isnan(r2) and r2 < 0.5) or n_outliers > 2:
                    plot_diagnostic_panel(
                        yearly_df,
                        row["district"],
                        entity_col="district",
                        output_path=os.path.join(
                            diagnostics_dir, f"{row['district']}_diagnostics.png"
                        ),
                    )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("trend_diagnostics failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("trend_diagnostics failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "trend_diagnostics", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="trend_diagnostics",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "trend_diagnostics", "success", input_summary={"districts": yearly_df["district"].nunique()}, output_summary={"diagnostics_computed": len(diag_df) if diag_df is not None else 0}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="trend_diagnostics",
        status="success",
        input_summary={"districts": yearly_df["district"].nunique()},
        output_summary={"diagnostics_computed": len(diag_df) if diag_df is not None else 0},
        timing_seconds=timer.elapsed,
    ), diag_df


def step_quality_diagnostics(
    years: list[int], output_dir: str, gdf: gpd.GeoDataFrame, csv_dir: str, diagnostics_dir: str
) -> tuple[StepResult, pd.DataFrame | None]:
    """Generate quality reports for each year and consolidate results."""
    from src.quality_diagnostics import generate_quality_report, plot_quality_heatmap

    quality_df = None
    error_tb = None

    with StepTimer() as timer:
        try:
            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(diagnostics_dir, exist_ok=True)

            quality_dfs = []
            for year in years:
                subset_dir = os.path.join(output_dir, "subsets", str(year))
                median_path = os.path.join(subset_dir, f"maharashtra_median_{year}.tif")
                lit_path = os.path.join(subset_dir, f"maharashtra_lit_mask_{year}.tif")
                cf_path = os.path.join(subset_dir, f"maharashtra_cf_cvg_{year}.tif")

                if os.path.exists(median_path):
                    qdf = generate_quality_report(
                        median_path,
                        lit_path,
                        cf_path,
                        gdf,
                        year,
                        output_csv=None,  # consolidated into quality_all_years.csv
                    )
                    quality_dfs.append(qdf)

            if quality_dfs:
                quality_df = pd.concat(quality_dfs, ignore_index=True)
                quality_df.to_csv(os.path.join(csv_dir, "quality_all_years.csv"), index=False)

                plot_quality_heatmap(
                    quality_df,
                    output_path=os.path.join(diagnostics_dir, "quality_heatmap.png"),
                )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("quality_diagnostics failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("quality_diagnostics failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "quality_diagnostics", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="quality_diagnostics",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "quality_diagnostics", "success", input_summary={"years": len(years)}, output_summary={"reports_generated": len(quality_dfs) if quality_df is not None else 0}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="quality_diagnostics",
        status="success",
        input_summary={"years": len(years)},
        output_summary={"reports_generated": len(quality_dfs) if quality_df is not None else 0},
        timing_seconds=timer.elapsed,
    ), quality_df


def step_benchmark_comparison(trends_df: pd.DataFrame, csv_dir: str) -> tuple[StepResult, None]:
    """Compare district trends to benchmark locations."""
    from src.benchmark_comparison import compare_to_benchmarks

    error_tb = None

    with StepTimer() as timer:
        try:
            os.makedirs(csv_dir, exist_ok=True)

            compare_to_benchmarks(
                trends_df,
                output_csv=os.path.join(csv_dir, "benchmark_comparison.csv"),
            )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("benchmark_comparison failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("benchmark_comparison failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "benchmark_comparison", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="benchmark_comparison",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "benchmark_comparison", "success", input_summary={"trends_count": len(trends_df)}, output_summary={}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="benchmark_comparison",
        status="success",
        input_summary={"trends_count": len(trends_df)},
        output_summary={},
        timing_seconds=timer.elapsed,
    ), None


def step_radial_gradient_analysis(
    output_dir: str, latest_year: int, csv_dir: str, maps_dir: str
) -> tuple[StepResult, pd.DataFrame | None]:
    """Extract radial profiles from urban centers."""
    from src import config
    from src.gradient_analysis import extract_radial_profiles, plot_radial_decay_curves

    profiles_df = None
    error_tb = None

    with StepTimer() as timer:
        try:
            subset_dir = os.path.join(output_dir, "subsets", str(latest_year))
            median_raster = os.path.join(subset_dir, f"maharashtra_median_{latest_year}.tif")

            if not os.path.exists(median_raster):
                error_tb = f"Median raster not found: {median_raster}"
                log.warning(error_tb)
            else:
                os.makedirs(csv_dir, exist_ok=True)
                os.makedirs(maps_dir, exist_ok=True)

                profiles_df = extract_radial_profiles(
                    raster_path=median_raster,
                    city_locations=config.URBAN_BENCHMARKS,
                    output_csv=os.path.join(csv_dir, f"urban_radial_profiles_{latest_year}.csv"),
                )

                plot_radial_decay_curves(
                    profiles_df,
                    output_path=os.path.join(maps_dir, "urban_radial_profiles.png"),
                )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("radial_gradient_analysis failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("radial_gradient_analysis failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "radial_gradient_analysis", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="radial_gradient_analysis",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "radial_gradient_analysis", "success", input_summary={"year": latest_year}, output_summary={"profiles_extracted": len(profiles_df) if profiles_df is not None else 0}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="radial_gradient_analysis",
        status="success",
        input_summary={"year": latest_year},
        output_summary={"profiles_extracted": len(profiles_df) if profiles_df is not None else 0},
        timing_seconds=timer.elapsed,
    ), profiles_df


def step_light_dome_modeling(
    profiles: pd.DataFrame, csv_dir: str, maps_dir: str
) -> tuple[StepResult, pd.DataFrame | None]:
    """Model light dome characteristics for urban centers."""
    from src.light_dome_modeling import model_all_city_domes, plot_dome_comparison

    dome_metrics = None
    error_tb = None

    with StepTimer() as timer:
        try:
            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(maps_dir, exist_ok=True)

            dome_metrics = model_all_city_domes(
                profiles,
                output_csv=os.path.join(csv_dir, "light_dome_metrics.csv"),
            )

            plot_dome_comparison(
                dome_metrics,
                profiles,
                output_path=os.path.join(maps_dir, "light_dome_comparison.png"),
            )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("light_dome_modeling failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("light_dome_modeling failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "light_dome_modeling", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="light_dome_modeling",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "light_dome_modeling", "success", input_summary={"profiles_count": len(profiles)}, output_summary={"domes_modeled": len(dome_metrics) if dome_metrics is not None else 0}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="light_dome_modeling",
        status="success",
        input_summary={"profiles_count": len(profiles)},
        output_summary={"domes_modeled": len(dome_metrics) if dome_metrics is not None else 0},
        timing_seconds=timer.elapsed,
    ), dome_metrics


def step_generate_basic_maps(
    gdf: gpd.GeoDataFrame,
    trends_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    output_dir: str,
    maps_dir: str = None,
) -> tuple[StepResult, None]:
    """Generate basic choropleth and time series maps."""
    from src.viirs_process import generate_maps

    error_tb = None

    with StepTimer() as timer:
        try:
            # Use provided maps_dir or construct entity output dir
            entity_output_dir = os.path.dirname(maps_dir) if maps_dir else output_dir
            generate_maps(gdf, trends_df, yearly_df, entity_output_dir)

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("generate_basic_maps failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("generate_basic_maps failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "generate_basic_maps", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="generate_basic_maps",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "generate_basic_maps", "success", input_summary={"districts": len(gdf)}, output_summary={}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="generate_basic_maps",
        status="success",
        input_summary={"districts": len(gdf)},
        output_summary={},
        timing_seconds=timer.elapsed,
    ), None


def step_statewide_visualizations(
    yearly_df: pd.DataFrame,
    trends_df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    quality_df: pd.DataFrame | None,
    output_dir: str,
    maps_dir: str = None,
) -> tuple[StepResult, None]:
    """Generate comprehensive statewide visualization suite."""
    from src.visualization_suite import (
        create_multi_year_comparison_grid,
        create_growth_classification_map,
        create_enhanced_radiance_heatmap,
        create_data_quality_map,
    )

    error_tb = None

    with StepTimer() as timer:
        try:
            if maps_dir is None:
                maps_dir = os.path.join(output_dir, "maps")
            os.makedirs(maps_dir, exist_ok=True)

            create_multi_year_comparison_grid(
                yearly_df,
                gdf,
                output_path=os.path.join(maps_dir, "multi_year_comparison.png"),
            )

            create_growth_classification_map(
                trends_df,
                gdf,
                output_path=os.path.join(maps_dir, "growth_classification.png"),
            )

            create_enhanced_radiance_heatmap(
                yearly_df,
                output_path=os.path.join(maps_dir, "radiance_heatmap_log.png"),
            )

            if quality_df is not None:
                create_data_quality_map(
                    quality_df,
                    gdf,
                    output_path=os.path.join(maps_dir, "data_quality_map.png"),
                )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("statewide_visualizations failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("statewide_visualizations failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "statewide_visualizations", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="statewide_visualizations",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "statewide_visualizations", "success", input_summary={"districts": len(gdf)}, output_summary={}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="statewide_visualizations",
        status="success",
        input_summary={"districts": len(gdf)},
        output_summary={},
        timing_seconds=timer.elapsed,
    ), None


def step_graduated_classification(
    yearly_df: pd.DataFrame, csv_dir: str, maps_dir: str
) -> tuple[StepResult, None]:
    """Classify districts by temporal trajectory and tier transitions."""
    from src.graduated_classification import (
        classify_temporal_trajectory,
        plot_tier_distribution,
        plot_tier_transition_matrix,
    )

    error_tb = None

    with StepTimer() as timer:
        try:
            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(maps_dir, exist_ok=True)

            trajectory = classify_temporal_trajectory(
                yearly_df,
                output_csv=os.path.join(csv_dir, "graduated_classification.csv"),
            )

            if not trajectory.empty:
                plot_tier_distribution(
                    trajectory,
                    output_path=os.path.join(maps_dir, "tier_distribution.png"),
                )

                first_year = yearly_df["year"].min()
                latest_year = yearly_df["year"].max()
                plot_tier_transition_matrix(
                    trajectory,
                    first_year,
                    latest_year,
                    output_path=os.path.join(
                        maps_dir, f"tier_transitions_{first_year}_{latest_year}.png"
                    ),
                )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("graduated_classification failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("graduated_classification failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "graduated_classification", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="graduated_classification",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "graduated_classification", "success", input_summary={"districts": yearly_df["district"].nunique()}, output_summary={}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="graduated_classification",
        status="success",
        input_summary={"districts": yearly_df["district"].nunique()},
        output_summary={},
        timing_seconds=timer.elapsed,
    ), None


def step_district_reports(
    yearly_df: pd.DataFrame,
    trends_df: pd.DataFrame,
    stability_df: pd.DataFrame | None,
    gdf: gpd.GeoDataFrame,
    output_dir: str,
    reports_dir: str = None,
) -> tuple[StepResult, None]:
    """Generate individual PDF reports for each district."""
    from src import config
    from src.district_reports import generate_all_district_reports

    error_tb = None

    with StepTimer() as timer:
        try:
            if reports_dir is None:
                reports_dir = os.path.join(output_dir, config.OUTPUT_DIRS["district_reports"])
            os.makedirs(reports_dir, exist_ok=True)

            generate_all_district_reports(
                yearly_df=yearly_df,
                trends_df=trends_df,
                stability_df=stability_df,
                gdf=gdf,
                output_dir=reports_dir,
            )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("district_reports failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("district_reports failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "district_reports", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="district_reports",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "district_reports", "success", input_summary={"districts": len(gdf)}, output_summary={}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="district_reports",
        status="success",
        input_summary={"districts": len(gdf)},
        output_summary={},
        timing_seconds=timer.elapsed,
    ), None
