"""
District pipeline step functions.

Each function is a discrete, testable pipeline step with explicit
inputs/outputs and StepResult tracking.  Boilerplate (timing, error
handling, logging) is handled by ``run_step()``.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd

from src.logging_config import get_pipeline_logger
from src.step_runner import run_step

log = get_pipeline_logger(__name__)


def step_load_boundaries(shapefile_path: str) -> tuple:
    """Load and validate district boundaries."""
    from src import config
    from src.validate_names import validate_or_exit

    def _work():
        gdf = gpd.read_file(shapefile_path)
        log.info("Loaded boundaries: %d districts", len(gdf))
        if len(gdf) != config.EXPECTED_DISTRICT_COUNT:
            log.warning(
                "Expected %d districts but found %d in %s",
                config.EXPECTED_DISTRICT_COUNT, len(gdf), shapefile_path,
            )
        validate_or_exit(shapefile_path, check_config=True)
        return gdf

    return run_step(
        "load_boundaries", _work,
        input_summary={"shapefile_path": shapefile_path},
        output_summary_fn=lambda gdf: {"districts": len(gdf)},
    )


def step_process_years(
    years: list[int],
    viirs_dir: str,
    gdf: gpd.GeoDataFrame,
    output_dir: str,
    cf_threshold: int,
) -> tuple:
    """Process all years and concatenate results."""
    from src.viirs_process import process_single_year

    def _work():
        all_yearly = []
        for year in years:
            df = process_single_year(year, viirs_dir, gdf, output_dir, cf_threshold)
            if df is not None:
                all_yearly.append(df)

        if not all_yearly:
            raise ValueError("No data processed for any year")

        yearly_df = pd.concat(all_yearly, ignore_index=True)
        log.info(
            "Total records: %d (%d years × %d districts)",
            len(yearly_df),
            yearly_df["year"].nunique(),
            yearly_df["district"].nunique(),
        )
        return yearly_df

    return run_step(
        "process_years", _work,
        input_summary={"years": years, "viirs_dir": viirs_dir},
        output_summary_fn=lambda df: {
            "total_records": len(df),
            "years_processed": df["year"].nunique(),
            "districts": df["district"].nunique(),
        },
    )


def step_save_yearly_radiance(yearly_df: pd.DataFrame, csv_dir: str) -> tuple:
    """Save yearly radiance data to CSV."""

    def _work():
        os.makedirs(csv_dir, exist_ok=True)
        result_path = os.path.join(csv_dir, "districts_yearly_radiance.csv")
        yearly_df.to_csv(result_path, index=False)
        log.info("Saved yearly data: %s", result_path)
        return result_path

    return run_step(
        "save_yearly_radiance", _work,
        input_summary={"records": len(yearly_df)},
        output_summary_fn=lambda p: {"csv_path": p},
    )


def step_fit_trends(yearly_df: pd.DataFrame, csv_dir: str) -> tuple:
    """Fit log-linear trends per district and classify ALAN levels."""
    from src.viirs_process import fit_log_linear_trend
    from src.formulas.classification import classify_alan

    def _work():
        districts = yearly_df["district"].unique()
        trend_results = []

        for d in districts:
            result = fit_log_linear_trend(yearly_df, d)
            latest = yearly_df[yearly_df["district"] == d].sort_values("year").iloc[-1]
            result["mean_radiance_latest"] = latest["mean_radiance"]
            result["median_radiance_latest"] = latest["median_radiance"]
            med = latest["median_radiance"]
            result["alan_class"] = classify_alan(med)
            trend_results.append(result)

        trends_df = pd.DataFrame(trend_results)

        os.makedirs(csv_dir, exist_ok=True)
        trends_path = os.path.join(csv_dir, "districts_trends.csv")
        trends_df.to_csv(trends_path, index=False)
        log.info("Saved trends: %s", trends_path)

        log.info("\n" + "=" * 60)
        log.info("TREND SUMMARY")
        log.info("=" * 60)
        for _, row in trends_df.sort_values("annual_pct_change", ascending=False).iterrows():
            rad = row["median_radiance_latest"]
            log.info(
                "%-20s %+6.2f%% [%+.2f, %+.2f] | %.2f nW (%s)",
                row["district"],
                row["annual_pct_change"],
                row["ci_low"],
                row["ci_high"],
                rad if rad is not None else 0.0,
                row["alan_class"],
            )

        low_alan = trends_df[trends_df["alan_class"] == "low"]
        log.info(
            "\nLow-ALAN districts (median < 1 nW/cm²/sr): %s",
            ", ".join(low_alan["district"].tolist()) if len(low_alan) > 0 else "None",
        )
        return trends_df

    return run_step(
        "fit_trends", _work,
        input_summary={"districts": len(yearly_df["district"].unique())},
        output_summary_fn=lambda df: {"trends_computed": len(df)},
    )


def step_stability_analysis(
    yearly_df: pd.DataFrame, csv_dir: str, diagnostics_dir: str
) -> tuple:
    """Compute temporal stability metrics for each district."""
    from src.analysis.stability_metrics import compute_stability_metrics, plot_stability_scatter

    def _work():
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(diagnostics_dir, exist_ok=True)
        stability_df = compute_stability_metrics(
            yearly_df, entity_col="district",
            output_csv=os.path.join(csv_dir, "district_stability_metrics.csv"),
        )
        plot_stability_scatter(
            stability_df, entity_col="district",
            output_path=os.path.join(diagnostics_dir, "stability_scatter.png"),
        )
        return stability_df

    return run_step(
        "stability_analysis", _work,
        input_summary={"districts": yearly_df["district"].nunique()},
        output_summary_fn=lambda df: {"metrics_computed": len(df) if df is not None else 0},
    )


def step_breakpoint_detection(
    yearly_df: pd.DataFrame, csv_dir: str, diagnostics_dir: str
) -> tuple:
    """Detect temporal breakpoints in radiance trends."""
    from src.analysis.breakpoint_analysis import analyze_all_breakpoints, plot_breakpoint_timeline

    def _work():
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(diagnostics_dir, exist_ok=True)
        breakpoints_df = analyze_all_breakpoints(
            yearly_df, entity_col="district",
            output_csv=os.path.join(csv_dir, "district_breakpoints.csv"),
        )
        plot_breakpoint_timeline(
            breakpoints_df,
            output_path=os.path.join(diagnostics_dir, "breakpoint_timeline.png"),
        )
        return breakpoints_df

    return run_step(
        "breakpoint_detection", _work,
        input_summary={"districts": yearly_df["district"].nunique()},
        output_summary_fn=lambda df: {"breakpoints_found": len(df) if df is not None else 0},
    )


def step_trend_diagnostics(
    yearly_df: pd.DataFrame, csv_dir: str, diagnostics_dir: str
) -> tuple:
    """Compute trend model diagnostics and generate plots for flagged districts."""
    from src.analysis.trend_diagnostics import compute_all_diagnostics, plot_diagnostic_panel

    def _work():
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(diagnostics_dir, exist_ok=True)
        diag_df = compute_all_diagnostics(
            yearly_df, entity_col="district",
            output_csv=os.path.join(csv_dir, "trend_diagnostics.csv"),
        )
        for _, row in diag_df.iterrows():
            r2 = row.get("r_squared", 1.0)
            outliers = row.get("outlier_years", "")
            n_outliers = len(outliers.split("; ")) if outliers else 0
            if (not np.isnan(r2) and r2 < 0.5) or n_outliers > 2:
                plot_diagnostic_panel(
                    yearly_df, row["district"], entity_col="district",
                    output_path=os.path.join(diagnostics_dir, f"{row['district']}_diagnostics.png"),
                )
        return diag_df

    return run_step(
        "trend_diagnostics", _work,
        input_summary={"districts": yearly_df["district"].nunique()},
        output_summary_fn=lambda df: {"diagnostics_computed": len(df) if df is not None else 0},
    )


def step_quality_diagnostics(
    years: list[int], output_dir: str, gdf: gpd.GeoDataFrame, csv_dir: str, diagnostics_dir: str
) -> tuple:
    """Generate quality reports for each year and consolidate results."""
    from src.analysis.quality_diagnostics import generate_quality_report, plot_quality_heatmap

    def _work():
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
                    median_path, lit_path, cf_path, gdf, year, output_csv=None,
                )
                quality_dfs.append(qdf)

        if quality_dfs:
            quality_df = pd.concat(quality_dfs, ignore_index=True)
            quality_df.to_csv(os.path.join(csv_dir, "quality_all_years.csv"), index=False)
            plot_quality_heatmap(
                quality_df,
                output_path=os.path.join(diagnostics_dir, "quality_heatmap.png"),
            )
            return quality_df
        return None

    return run_step(
        "quality_diagnostics", _work,
        input_summary={"years": len(years)},
        output_summary_fn=lambda df: {"reports_generated": len(df) if df is not None else 0},
    )


def step_benchmark_comparison(trends_df: pd.DataFrame, csv_dir: str) -> tuple:
    """Compare district trends to benchmark locations."""
    from src.analysis.benchmark_comparison import compare_to_benchmarks

    def _work():
        os.makedirs(csv_dir, exist_ok=True)
        compare_to_benchmarks(
            trends_df,
            output_csv=os.path.join(csv_dir, "benchmark_comparison.csv"),
        )
        return None

    return run_step(
        "benchmark_comparison", _work,
        input_summary={"trends_count": len(trends_df)},
    )


def step_radial_gradient_analysis(
    output_dir: str, latest_year: int, csv_dir: str, maps_dir: str
) -> tuple:
    """Extract radial profiles from urban centers."""
    from src import config
    from src.analysis.gradient_analysis import extract_radial_profiles, plot_radial_decay_curves

    def _work():
        subset_dir = os.path.join(output_dir, "subsets", str(latest_year))
        median_raster = os.path.join(subset_dir, f"maharashtra_median_{latest_year}.tif")
        if not os.path.exists(median_raster):
            log.warning("Median raster not found: %s", median_raster)
            return None
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(maps_dir, exist_ok=True)
        profiles_df = extract_radial_profiles(
            raster_path=median_raster,
            city_locations=config.URBAN_CITIES,
            output_csv=os.path.join(csv_dir, f"urban_radial_profiles_{latest_year}.csv"),
        )
        plot_radial_decay_curves(
            profiles_df,
            output_path=os.path.join(maps_dir, "urban_radial_profiles.png"),
        )
        return profiles_df

    return run_step(
        "radial_gradient_analysis", _work,
        input_summary={"year": latest_year},
        output_summary_fn=lambda df: {"profiles_extracted": len(df) if df is not None else 0},
    )


def step_light_dome_modeling(
    profiles: pd.DataFrame, csv_dir: str, maps_dir: str
) -> tuple:
    """Model light dome characteristics for urban centers."""
    from src.analysis.light_dome_modeling import model_all_city_domes, plot_dome_comparison

    def _work():
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(maps_dir, exist_ok=True)
        dome_metrics = model_all_city_domes(
            profiles,
            output_csv=os.path.join(csv_dir, "light_dome_metrics.csv"),
        )
        plot_dome_comparison(
            dome_metrics, profiles,
            output_path=os.path.join(maps_dir, "light_dome_comparison.png"),
        )
        return dome_metrics

    return run_step(
        "light_dome_modeling", _work,
        input_summary={"profiles_count": len(profiles)},
        output_summary_fn=lambda df: {"domes_modeled": len(df) if df is not None else 0},
    )


def step_generate_basic_maps(
    gdf: gpd.GeoDataFrame,
    trends_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    output_dir: str,
    maps_dir: str = None,
) -> tuple:
    """Generate basic choropleth and time series maps."""
    from src.viirs_process import generate_maps

    def _work():
        entity_output_dir = os.path.dirname(maps_dir) if maps_dir else output_dir
        generate_maps(gdf, trends_df, yearly_df, entity_output_dir)
        return None

    return run_step(
        "generate_basic_maps", _work,
        input_summary={"districts": len(gdf)},
    )


def step_statewide_visualizations(
    yearly_df: pd.DataFrame,
    trends_df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    quality_df: pd.DataFrame | None,
    output_dir: str,
    maps_dir: str = None,
) -> tuple:
    """Generate comprehensive statewide visualization suite."""
    from src.outputs.visualization_suite import (
        create_multi_year_comparison_grid,
        create_growth_classification_map,
        create_enhanced_radiance_heatmap,
        create_data_quality_map,
    )

    def _work():
        target_dir = maps_dir if maps_dir is not None else os.path.join(output_dir, "maps")
        os.makedirs(target_dir, exist_ok=True)
        create_multi_year_comparison_grid(
            yearly_df, gdf,
            output_path=os.path.join(target_dir, "multi_year_comparison.png"),
        )
        create_growth_classification_map(
            trends_df, gdf,
            output_path=os.path.join(target_dir, "growth_classification.png"),
        )
        create_enhanced_radiance_heatmap(
            yearly_df,
            output_path=os.path.join(target_dir, "radiance_heatmap_log.png"),
        )
        if quality_df is not None:
            create_data_quality_map(
                quality_df, gdf,
                output_path=os.path.join(target_dir, "data_quality_map.png"),
            )
        return None

    return run_step(
        "statewide_visualizations", _work,
        input_summary={"districts": len(gdf)},
    )


def step_graduated_classification(
    yearly_df: pd.DataFrame, csv_dir: str, maps_dir: str
) -> tuple:
    """Classify districts by temporal trajectory and tier transitions."""
    from src.analysis.graduated_classification import (
        classify_temporal_trajectory,
        plot_tier_distribution,
        plot_tier_transition_matrix,
    )

    def _work():
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
                trajectory, first_year, latest_year,
                output_path=os.path.join(
                    maps_dir, f"tier_transitions_{first_year}_{latest_year}.png"
                ),
            )
        return None

    return run_step(
        "graduated_classification", _work,
        input_summary={"districts": yearly_df["district"].nunique()},
    )


def step_district_reports(
    yearly_df: pd.DataFrame,
    trends_df: pd.DataFrame,
    stability_df: pd.DataFrame | None,
    gdf: gpd.GeoDataFrame,
    output_dir: str,
    reports_dir: str = None,
) -> tuple:
    """Generate individual PDF reports for each district."""
    from src.outputs.district_reports import generate_all_district_reports

    def _work():
        target_dir = reports_dir if reports_dir is not None else os.path.join(output_dir, "district", "reports")
        os.makedirs(target_dir, exist_ok=True)
        generate_all_district_reports(
            yearly_df=yearly_df,
            trends_df=trends_df,
            stability_df=stability_df,
            gdf=gdf,
            output_dir=target_dir,
        )
        return None

    return run_step(
        "district_reports", _work,
        input_summary={"districts": len(gdf)},
    )


def step_animation_frames(
    years: list[int],
    output_dir: str,
    gdf: gpd.GeoDataFrame,
    maps_dir: str = None,
) -> tuple:
    """Generate animation frames (sprawl, differential, darkness) and trend map."""
    from src.outputs.visualizations import (
        generate_sprawl_frames,
        generate_differential_frames,
        generate_darkness_frames,
        generate_trend_map,
    )

    def _work():
        frame_counts = {}
        generate_sprawl_frames(years, output_dir, gdf, maps_output_dir=maps_dir)
        frame_counts["sprawl"] = len(years)
        generate_differential_frames(years, output_dir, gdf, maps_output_dir=maps_dir)
        frame_counts["differential"] = len(years)
        generate_darkness_frames(years, output_dir, gdf, maps_output_dir=maps_dir)
        frame_counts["darkness"] = len(years)
        generate_trend_map(years, output_dir, gdf, maps_output_dir=maps_dir)
        frame_counts["trend_map"] = 1
        return frame_counts

    return run_step(
        "animation_frames", _work,
        input_summary={"years": len(years)},
        output_summary_fn=lambda counts: counts,
    )


def step_per_district_radiance_maps(
    output_dir: str,
    latest_year: int,
    gdf: gpd.GeoDataFrame,
    maps_dir: str = None,
) -> tuple:
    """Generate per-district zoomed radiance raster maps."""
    from src.outputs.visualizations import generate_per_district_radiance_maps

    def _work():
        return generate_per_district_radiance_maps(
            output_dir, latest_year, gdf, maps_output_dir=maps_dir,
        )

    return run_step(
        "per_district_radiance_maps", _work,
        input_summary={"year": latest_year, "districts": len(gdf)},
        output_summary_fn=lambda count: {"maps_generated": count},
    )
