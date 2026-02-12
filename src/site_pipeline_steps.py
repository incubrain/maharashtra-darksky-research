"""
Site/city pipeline step functions.

Each function is a discrete, testable pipeline step with explicit
inputs/outputs and StepResult tracking.
"""

import logging
import os
import traceback

import geopandas as gpd
import numpy as np
import pandas as pd

from src.pipeline_types import StepResult
from src.logging_config import StepTimer, get_pipeline_logger, log_step_summary

log = get_pipeline_logger(__name__)


def step_build_site_buffers(buffer_km: float, entity_type: str = "all") -> tuple[StepResult, gpd.GeoDataFrame | None]:
    """Build circular buffers around site/city locations."""
    from src.site_analysis import build_site_geodataframe

    gdf_sites = None
    error_tb = None

    with StepTimer() as timer:
        try:
            gdf_sites = build_site_geodataframe(buffer_km, entity_type)
        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("build_site_buffers failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("build_site_buffers failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "build_site_buffers", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="build_site_buffers",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "build_site_buffers", "success", input_summary={"buffer_km": buffer_km, "entity_type": entity_type}, output_summary={"sites": len(gdf_sites)}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="build_site_buffers",
        status="success",
        input_summary={"buffer_km": buffer_km, "entity_type": entity_type},
        output_summary={"sites": len(gdf_sites)},
        timing_seconds=timer.elapsed,
    ), gdf_sites


def step_compute_yearly_metrics(
    years: list[int], gdf_sites: gpd.GeoDataFrame, output_dir: str, cf_threshold: int
) -> tuple[StepResult, pd.DataFrame | None]:
    """Compute site metrics for each year and concatenate."""
    from src.site_analysis import compute_site_metrics

    all_yearly = []
    error_tb = None

    with StepTimer() as timer:
        try:
            for year in years:
                subset_dir = os.path.join(output_dir, "subsets", str(year))
                if not os.path.isdir(subset_dir):
                    log.warning("No subsets for %d — skipping", year)
                    continue
                df = compute_site_metrics(gdf_sites, subset_dir, year, cf_threshold)
                if df is not None:
                    all_yearly.append(df)
                    log.info("Year %d: %d sites processed", year, len(df))

            if not all_yearly:
                error_tb = "No data processed for any year"
                log.error(error_tb)
            else:
                yearly_df = pd.concat(all_yearly, ignore_index=True)
                log.info(
                    "Total records: %d (%d years × %d sites)",
                    len(yearly_df),
                    yearly_df["year"].nunique(),
                    yearly_df["name"].nunique(),
                )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("compute_yearly_metrics failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("compute_yearly_metrics failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "compute_yearly_metrics", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="compute_yearly_metrics",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "compute_yearly_metrics", "success", input_summary={"years": years, "sites": len(gdf_sites)}, output_summary={"total_records": len(yearly_df), "years_processed": yearly_df["year"].nunique(), "sites": yearly_df["name"].nunique()}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="compute_yearly_metrics",
        status="success",
        input_summary={"years": years, "sites": len(gdf_sites)},
        output_summary={
            "total_records": len(yearly_df),
            "years_processed": yearly_df["year"].nunique(),
            "sites": yearly_df["name"].nunique(),
        },
        timing_seconds=timer.elapsed,
    ), yearly_df


def step_save_site_yearly(yearly_df: pd.DataFrame, csv_dir: str) -> tuple[StepResult, str]:
    """Save site yearly radiance data to CSV."""
    result_path = None
    error_tb = None

    with StepTimer() as timer:
        try:
            os.makedirs(csv_dir, exist_ok=True)
            result_path = os.path.join(csv_dir, "site_yearly_radiance.csv")
            yearly_df.to_csv(result_path, index=False)
            log.info("Saved: %s", result_path)

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("save_site_yearly failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("save_site_yearly failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "save_site_yearly", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="save_site_yearly",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "save_site_yearly", "success", input_summary={"records": len(yearly_df)}, output_summary={"csv_path": result_path}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="save_site_yearly",
        status="success",
        input_summary={"records": len(yearly_df)},
        output_summary={"csv_path": result_path},
        timing_seconds=timer.elapsed,
    ), result_path


def step_fit_site_trends(yearly_df: pd.DataFrame, csv_dir: str) -> tuple[StepResult, pd.DataFrame | None]:
    """Fit log-linear trends for each site and classify ALAN levels."""
    from src.site_analysis import fit_site_trends
    from src.formulas.classification import classify_alan

    trends_df = None
    error_tb = None

    with StepTimer() as timer:
        try:
            trends_df = fit_site_trends(yearly_df)

            # Add ALAN classification based on latest year median
            for i, row in trends_df.iterrows():
                if "median_radiance_latest" not in row or pd.isna(row.get("median_radiance_latest")):
                    latest = yearly_df[(yearly_df["name"] == row["name"])].sort_values("year").iloc[-1]
                    trends_df.at[i, "median_radiance_latest"] = latest["median_radiance"]
                    trends_df.at[i, "mean_radiance_latest"] = latest["mean_radiance"]
                med = trends_df.at[i, "median_radiance_latest"]
                trends_df.at[i, "alan_class"] = classify_alan(med)

            # Save to CSV
            os.makedirs(csv_dir, exist_ok=True)
            trends_path = os.path.join(csv_dir, "site_trends.csv")
            trends_df.to_csv(trends_path, index=False)
            log.info("Saved: %s", trends_path)

            # Print summary
            log.info("\n" + "=" * 70)
            log.info("SITE TREND SUMMARY")
            log.info("=" * 70)

            cities_t = trends_df[trends_df["type"] == "city"].sort_values(
                "annual_pct_change", ascending=False
            )
            sites_t = trends_df[trends_df["type"] == "site"].sort_values(
                "annual_pct_change", ascending=False
            )

            log.info("\n--- CITIES ---")
            for _, r in cities_t.iterrows():
                log.info(
                    "  %-20s %+6.2f%% [%+.2f, %+.2f]  R²=%.3f  p=%.1e  latest=%.2f nW  [%s]",
                    r["name"],
                    r["annual_pct_change"],
                    r["ci_low"],
                    r["ci_high"],
                    r["r_squared"],
                    r["p_value"],
                    r["median_radiance_latest"],
                    r["alan_class"],
                )

            log.info("\n--- DARK-SKY SITES ---")
            for _, r in sites_t.iterrows():
                log.info(
                    "  %-30s %+6.2f%% [%+.2f, %+.2f]  R²=%.3f  p=%.1e  latest=%.2f nW  [%s]",
                    r["name"],
                    r["annual_pct_change"],
                    r["ci_low"],
                    r["ci_high"],
                    r["r_squared"],
                    r["p_value"],
                    r["median_radiance_latest"],
                    r["alan_class"],
                )

            city_avg = cities_t["annual_pct_change"].mean()
            site_avg = sites_t["annual_pct_change"].mean()
            log.info("\nCities avg growth: %+.2f%%/yr", city_avg)
            log.info("Sites avg growth:  %+.2f%%/yr", site_avg)

            low_alan = sites_t[sites_t["alan_class"] == "low"]
            latest_year = yearly_df["year"].max()
            log.info(
                "\nDark-sky viable (median < 1 nW in %d): %d of %d sites",
                latest_year,
                len(low_alan),
                len(sites_t),
            )
            if len(low_alan) > 0:
                log.info("  %s", ", ".join(low_alan["name"].tolist()))

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("fit_site_trends failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("fit_site_trends failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "fit_site_trends", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="fit_site_trends",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "fit_site_trends", "success", input_summary={"sites": yearly_df["name"].nunique()}, output_summary={"trends_computed": len(trends_df)}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="fit_site_trends",
        status="success",
        input_summary={"sites": yearly_df["name"].nunique()},
        output_summary={"trends_computed": len(trends_df)},
        timing_seconds=timer.elapsed,
    ), trends_df


def step_site_maps(
    gdf_sites: gpd.GeoDataFrame,
    yearly_df: pd.DataFrame,
    district_gdf: gpd.GeoDataFrame,
    output_dir: str,
    latest_year: int,
    maps_dir: str = None,
) -> tuple[StepResult, None]:
    """Generate site overlay and comparison maps."""
    from src.site_analysis import generate_site_maps, generate_site_timeseries

    error_tb = None

    with StepTimer() as timer:
        try:
            latest_subset_dir = os.path.join(output_dir, "subsets", str(latest_year))
            latest_metrics = yearly_df[yearly_df["year"] == latest_year]

            # Use provided maps_dir or construct entity output dir
            entity_output_dir = os.path.dirname(maps_dir) if maps_dir else output_dir

            generate_site_maps(
                gdf_sites, latest_metrics, district_gdf, latest_subset_dir, entity_output_dir, latest_year
            )
            generate_site_timeseries(yearly_df, entity_output_dir)

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("site_maps failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("site_maps failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "site_maps", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="site_maps",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "site_maps", "success", input_summary={"sites": len(gdf_sites), "year": latest_year}, output_summary={}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="site_maps",
        status="success",
        input_summary={"sites": len(gdf_sites), "year": latest_year},
        output_summary={},
        timing_seconds=timer.elapsed,
    ), None


def step_spatial_analysis(
    gdf_sites: gpd.GeoDataFrame, output_dir: str, latest_year: int,
    csv_dir: str = None, maps_dir: str = None
) -> tuple[StepResult, dict | None]:
    """Run spatial analysis: buffer comparison, directional, proximity."""
    from src.buffer_comparison import compare_inside_outside_buffers, plot_inside_outside_comparison
    from src.directional_analysis import compute_directional_brightness, plot_directional_polar
    from src.proximity_analysis import compute_nearest_city_distances

    spatial_results = None
    error_tb = None

    with StepTimer() as timer:
        try:
            # Use provided directories or construct defaults
            if csv_dir is None:
                csv_dir = os.path.join(output_dir, "csv")
            if maps_dir is None:
                maps_dir = os.path.join(output_dir, "maps")
            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(maps_dir, exist_ok=True)

            latest_subset_dir = os.path.join(output_dir, "subsets", str(latest_year))
            median_path = os.path.join(latest_subset_dir, f"maharashtra_median_{latest_year}.tif")

            spatial_results = {}

            if os.path.exists(median_path):
                # Buffer comparison
                buffer_comparison = compare_inside_outside_buffers(
                    site_gdf=gdf_sites,
                    raster_path=median_path,
                    output_csv=os.path.join(csv_dir, f"site_buffer_comparison_{latest_year}.csv"),
                )
                plot_inside_outside_comparison(
                    buffer_comparison,
                    output_path=os.path.join(maps_dir, "site_buffer_comparison.png"),
                )
                spatial_results["buffer_comparison"] = buffer_comparison

                # Directional analysis
                directional = compute_directional_brightness(
                    raster_path=median_path,
                    output_csv=os.path.join(csv_dir, f"directional_brightness_{latest_year}.csv"),
                )
                plot_directional_polar(
                    directional,
                    output_path=os.path.join(maps_dir, "directional_brightness_polar.pdf"),
                )
                spatial_results["directional"] = directional

            # Proximity analysis
            proximity = compute_nearest_city_distances(
                output_csv=os.path.join(csv_dir, "site_proximity_metrics.csv"),
            )
            spatial_results["proximity"] = proximity

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("spatial_analysis failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("spatial_analysis failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "spatial_analysis", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="spatial_analysis",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "spatial_analysis", "success", input_summary={"sites": len(gdf_sites), "year": latest_year}, output_summary={}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="spatial_analysis",
        status="success",
        input_summary={"sites": len(gdf_sites), "year": latest_year},
        output_summary={},
        timing_seconds=timer.elapsed,
    ), spatial_results


def step_sky_brightness(
    latest_metrics: pd.DataFrame, csv_dir: str, maps_dir: str, latest_year: int
) -> tuple[StepResult, pd.DataFrame | None]:
    """Compute sky brightness metrics and generate distribution plot."""
    from src.sky_brightness_model import compute_sky_brightness_metrics, plot_sky_brightness_distribution

    sky_metrics = None
    error_tb = None

    with StepTimer() as timer:
        try:
            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(maps_dir, exist_ok=True)

            sky_metrics = compute_sky_brightness_metrics(
                latest_metrics,
                output_csv=os.path.join(csv_dir, f"sky_brightness_{latest_year}.csv"),
            )
            plot_sky_brightness_distribution(
                sky_metrics,
                output_path=os.path.join(maps_dir, "sky_brightness_distribution.png"),
            )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("sky_brightness failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("sky_brightness failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "sky_brightness", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="sky_brightness",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "sky_brightness", "success", input_summary={"sites": len(latest_metrics)}, output_summary={"metrics_computed": len(sky_metrics) if sky_metrics is not None else 0}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="sky_brightness",
        status="success",
        input_summary={"sites": len(latest_metrics)},
        output_summary={"metrics_computed": len(sky_metrics) if sky_metrics is not None else 0},
        timing_seconds=timer.elapsed,
    ), sky_metrics


def step_site_stability(
    yearly_df: pd.DataFrame, csv_dir: str, diagnostics_dir: str
) -> tuple[StepResult, pd.DataFrame | None]:
    """Compute temporal stability metrics and generate scatter plot for sites."""
    from src.stability_metrics import compute_stability_metrics, plot_stability_scatter

    site_stability = None
    error_tb = None

    with StepTimer() as timer:
        try:
            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(diagnostics_dir, exist_ok=True)

            site_stability = compute_stability_metrics(
                yearly_df,
                entity_col="name",
                output_csv=os.path.join(csv_dir, "site_stability_metrics.csv"),
            )
            plot_stability_scatter(
                site_stability,
                entity_col="name",
                output_path=os.path.join(diagnostics_dir, "site_stability_scatter.png"),
            )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("site_stability failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("site_stability failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "site_stability", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="site_stability",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "site_stability", "success", input_summary={"sites": yearly_df["name"].nunique()}, output_summary={"metrics_computed": len(site_stability) if site_stability is not None else 0}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="site_stability",
        status="success",
        input_summary={"sites": yearly_df["name"].nunique()},
        output_summary={"metrics_computed": len(site_stability) if site_stability is not None else 0},
        timing_seconds=timer.elapsed,
    ), site_stability


def step_site_breakpoints(
    yearly_df: pd.DataFrame, csv_dir: str
) -> tuple[StepResult, pd.DataFrame | None]:
    """Detect temporal breakpoints in site radiance trends."""
    from src.breakpoint_analysis import analyze_all_breakpoints

    site_breakpoints = None
    error_tb = None

    with StepTimer() as timer:
        try:
            os.makedirs(csv_dir, exist_ok=True)

            site_breakpoints = analyze_all_breakpoints(
                yearly_df,
                entity_col="name",
                output_csv=os.path.join(csv_dir, "site_breakpoints.csv"),
            )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("site_breakpoints failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("site_breakpoints failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "site_breakpoints", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="site_breakpoints",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "site_breakpoints", "success", input_summary={"sites": yearly_df["name"].nunique()}, output_summary={"breakpoints_found": len(site_breakpoints) if site_breakpoints is not None else 0}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="site_breakpoints",
        status="success",
        input_summary={"sites": yearly_df["name"].nunique()},
        output_summary={"breakpoints_found": len(site_breakpoints) if site_breakpoints is not None else 0},
        timing_seconds=timer.elapsed,
    ), site_breakpoints


def step_site_benchmark(trends_df: pd.DataFrame, csv_dir: str) -> tuple[StepResult, None]:
    """Compare site trends to benchmark locations."""
    from src.benchmark_comparison import compare_to_benchmarks

    error_tb = None

    with StepTimer() as timer:
        try:
            os.makedirs(csv_dir, exist_ok=True)

            compare_to_benchmarks(
                trends_df,
                output_csv=os.path.join(csv_dir, "site_benchmark_comparison.csv"),
            )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("site_benchmark failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("site_benchmark failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "site_benchmark", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="site_benchmark",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "site_benchmark", "success", input_summary={"trends_count": len(trends_df)}, output_summary={}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="site_benchmark",
        status="success",
        input_summary={"trends_count": len(trends_df)},
        output_summary={},
        timing_seconds=timer.elapsed,
    ), None


def step_site_visualizations(
    latest_metrics: pd.DataFrame, maps_dir: str
) -> tuple[StepResult, None]:
    """Generate city vs site boxplot visualization."""
    from src.visualization_suite import create_city_vs_site_boxplot

    error_tb = None

    with StepTimer() as timer:
        try:
            os.makedirs(maps_dir, exist_ok=True)

            create_city_vs_site_boxplot(
                latest_metrics,
                output_path=os.path.join(maps_dir, "city_vs_site_boxplot.png"),
            )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("site_visualizations failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("site_visualizations failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "site_visualizations", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="site_visualizations",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "site_visualizations", "success", input_summary={"sites": len(latest_metrics)}, output_summary={}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="site_visualizations",
        status="success",
        input_summary={"sites": len(latest_metrics)},
        output_summary={},
        timing_seconds=timer.elapsed,
    ), None


def step_site_reports(
    latest_metrics: pd.DataFrame, yearly_df: pd.DataFrame, reports_dir: str,
    entity_type: str = "all"
) -> tuple[StepResult, None]:
    """Generate individual PDF reports for each site."""
    from src.site_reports import generate_all_site_reports

    error_tb = None

    with StepTimer() as timer:
        try:
            os.makedirs(reports_dir, exist_ok=True)

            generate_all_site_reports(
                all_site_data=latest_metrics,
                yearly_df=yearly_df,
                output_dir=reports_dir,
            )

        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            error_tb = traceback.format_exc()
            log.error("site_reports failed: %s", exc, exc_info=True)
        except Exception:
            error_tb = traceback.format_exc()
            log.error("site_reports failed unexpectedly", exc_info=True)

    if error_tb:
        log_step_summary(log, "site_reports", "error", timing_seconds=timer.elapsed)
        return StepResult(
            step_name="site_reports",
            status="error",
            error=error_tb,
            timing_seconds=timer.elapsed,
        ), None

    log_step_summary(log, "site_reports", "success", input_summary={"sites": len(latest_metrics)}, output_summary={}, timing_seconds=timer.elapsed)
    return StepResult(
        step_name="site_reports",
        status="success",
        input_summary={"sites": len(latest_metrics)},
        output_summary={},
        timing_seconds=timer.elapsed,
    ), None
