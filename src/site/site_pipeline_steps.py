"""
Site/city pipeline step functions.

Each function is a discrete, testable pipeline step with explicit
inputs/outputs and StepResult tracking.  Boilerplate is handled
by ``run_step()``.
"""

import os

import geopandas as gpd
import pandas as pd

from src.logging_config import get_pipeline_logger
from src.step_runner import run_step

log = get_pipeline_logger(__name__)


def _entity_prefix(yearly_df: pd.DataFrame) -> str:
    """Determine file prefix from entity type in data ('city' or 'site')."""
    if "type" in yearly_df.columns:
        types = yearly_df["type"].unique()
        if len(types) == 1:
            return types[0]
    return "site"


def step_build_site_buffers(buffer_km: float, entity_type: str = "all", city_source: str = "config") -> tuple:
    """Build circular buffers around site/city locations."""
    from src.site.site_analysis import build_site_geodataframe

    return run_step(
        "build_site_buffers",
        build_site_geodataframe,
        buffer_km, entity_type, city_source=city_source,
        input_summary={"buffer_km": buffer_km, "entity_type": entity_type},
        output_summary_fn=lambda gdf: {"sites": len(gdf)},
    )


def step_compute_yearly_metrics(
    years: list[int], gdf_sites: gpd.GeoDataFrame, output_dir: str, cf_threshold: int
) -> tuple:
    """Compute site metrics for each year and concatenate."""
    from src.site.site_analysis import compute_site_metrics

    def _work():
        all_yearly = []
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
            raise ValueError("No data processed for any year")

        yearly_df = pd.concat(all_yearly, ignore_index=True)
        log.info(
            "Total records: %d (%d years × %d sites)",
            len(yearly_df),
            yearly_df["year"].nunique(),
            yearly_df["name"].nunique(),
        )
        return yearly_df

    return run_step(
        "compute_yearly_metrics", _work,
        input_summary={"years": years, "sites": len(gdf_sites)},
        output_summary_fn=lambda df: {
            "total_records": len(df),
            "years_processed": df["year"].nunique(),
            "sites": df["name"].nunique(),
        },
    )


def step_save_site_yearly(yearly_df: pd.DataFrame, csv_dir: str) -> tuple:
    """Save site yearly radiance data to CSV."""

    def _work():
        os.makedirs(csv_dir, exist_ok=True)
        prefix = _entity_prefix(yearly_df)
        result_path = os.path.join(csv_dir, f"{prefix}_yearly_radiance.csv")
        yearly_df.to_csv(result_path, index=False)
        log.info("Saved: %s", result_path)
        return result_path

    return run_step(
        "save_site_yearly", _work,
        input_summary={"records": len(yearly_df)},
        output_summary_fn=lambda p: {"csv_path": p},
    )


def step_fit_site_trends(yearly_df: pd.DataFrame, csv_dir: str) -> tuple:
    """Fit log-linear trends for each site and classify ALAN levels."""
    from src.site.site_analysis import fit_site_trends
    from src.formulas.classification import classify_alan
    import numpy as np

    def _work():
        trends_df = fit_site_trends(yearly_df)

        for i, row in trends_df.iterrows():
            if "median_radiance_latest" not in row or pd.isna(row.get("median_radiance_latest")):
                latest = yearly_df[(yearly_df["name"] == row["name"])].sort_values("year").iloc[-1]
                trends_df.at[i, "median_radiance_latest"] = latest["median_radiance"]
                trends_df.at[i, "mean_radiance_latest"] = latest["mean_radiance"]
            med = trends_df.at[i, "median_radiance_latest"]
            trends_df.at[i, "alan_class"] = classify_alan(med)

        os.makedirs(csv_dir, exist_ok=True)
        prefix = _entity_prefix(yearly_df)
        trends_path = os.path.join(csv_dir, f"{prefix}_trends.csv")
        trends_df.to_csv(trends_path, index=False)
        log.info("Saved: %s", trends_path)

        log.info("\n" + "=" * 70)
        log.info("%s TREND SUMMARY", prefix.upper())
        log.info("=" * 70)
        sorted_trends = trends_df.sort_values("annual_pct_change", ascending=False)
        for _, r in sorted_trends.iterrows():
            log.info(
                "  %-30s %+6.2f%% [%+.2f, %+.2f]  R²=%.3f  p=%.1e  latest=%.2f nW  [%s]",
                r["name"], r["annual_pct_change"], r["ci_low"], r["ci_high"],
                r["r_squared"], r["p_value"], r["median_radiance_latest"], r["alan_class"],
            )
        avg_growth = sorted_trends["annual_pct_change"].mean()
        log.info("\nAvg growth: %+.2f%%/yr", avg_growth)

        if prefix == "site":
            low_alan = sorted_trends[sorted_trends["alan_class"] == "low"]
            latest_year = yearly_df["year"].max()
            log.info(
                "\nDark-sky viable (median < 1 nW in %d): %d of %d sites",
                latest_year, len(low_alan), len(sorted_trends),
            )
            if len(low_alan) > 0:
                log.info("  %s", ", ".join(low_alan["name"].tolist()))

        return trends_df

    return run_step(
        "fit_site_trends", _work,
        input_summary={"sites": yearly_df["name"].nunique()},
        output_summary_fn=lambda df: {"trends_computed": len(df)},
    )


def step_site_maps(
    gdf_sites: gpd.GeoDataFrame,
    yearly_df: pd.DataFrame,
    district_gdf: gpd.GeoDataFrame,
    output_dir: str,
    latest_year: int,
    maps_dir: str = None,
) -> tuple:
    """Generate site overlay and comparison maps."""
    from src.site.site_analysis import generate_site_maps, generate_site_timeseries

    def _work():
        latest_subset_dir = os.path.join(output_dir, "subsets", str(latest_year))
        latest_metrics = yearly_df[yearly_df["year"] == latest_year]
        entity_output_dir = os.path.dirname(maps_dir) if maps_dir else output_dir
        generate_site_maps(
            gdf_sites, latest_metrics, district_gdf, latest_subset_dir, entity_output_dir, latest_year
        )
        generate_site_timeseries(yearly_df, entity_output_dir)
        return None

    return run_step(
        "site_maps", _work,
        input_summary={"sites": len(gdf_sites), "year": latest_year},
    )


def step_spatial_analysis(
    gdf_sites: gpd.GeoDataFrame, output_dir: str, latest_year: int,
    csv_dir: str = None, maps_dir: str = None
) -> tuple:
    """Run spatial analysis: buffer comparison, directional, proximity."""
    from src.analysis.buffer_comparison import compare_inside_outside_buffers, plot_inside_outside_comparison
    from src.analysis.directional_analysis import compute_directional_brightness, plot_directional_polar
    from src.analysis.proximity_analysis import compute_nearest_city_distances

    def _work():
        _csv = csv_dir if csv_dir is not None else os.path.join(output_dir, "csv")
        _maps = maps_dir if maps_dir is not None else os.path.join(output_dir, "maps")
        os.makedirs(_csv, exist_ok=True)
        os.makedirs(_maps, exist_ok=True)

        latest_subset_dir = os.path.join(output_dir, "subsets", str(latest_year))
        median_path = os.path.join(latest_subset_dir, f"maharashtra_median_{latest_year}.tif")

        spatial_results = {}
        if os.path.exists(median_path):
            buffer_comparison = compare_inside_outside_buffers(
                site_gdf=gdf_sites, raster_path=median_path,
                output_csv=os.path.join(_csv, f"buffer_comparison_{latest_year}.csv"),
            )
            plot_inside_outside_comparison(
                buffer_comparison,
                output_path=os.path.join(_maps, "buffer_comparison.png"),
            )
            spatial_results["buffer_comparison"] = buffer_comparison

            directional = compute_directional_brightness(
                raster_path=median_path,
                output_csv=os.path.join(_csv, f"directional_brightness_{latest_year}.csv"),
            )
            plot_directional_polar(
                directional,
                output_path=os.path.join(_maps, "directional_brightness_polar.pdf"),
            )
            spatial_results["directional"] = directional

        proximity = compute_nearest_city_distances(
            output_csv=os.path.join(_csv, "proximity_metrics.csv"),
        )
        spatial_results["proximity"] = proximity
        return spatial_results

    return run_step(
        "spatial_analysis", _work,
        input_summary={"sites": len(gdf_sites), "year": latest_year},
    )


def step_sky_brightness(
    latest_metrics: pd.DataFrame, csv_dir: str, maps_dir: str, latest_year: int
) -> tuple:
    """Compute sky brightness metrics and generate distribution plot."""
    from src.analysis.sky_brightness_model import compute_sky_brightness_metrics, plot_sky_brightness_distribution

    def _work():
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
        return sky_metrics

    return run_step(
        "sky_brightness", _work,
        input_summary={"sites": len(latest_metrics)},
        output_summary_fn=lambda df: {"metrics_computed": len(df) if df is not None else 0},
    )


def step_site_stability(
    yearly_df: pd.DataFrame, csv_dir: str, diagnostics_dir: str
) -> tuple:
    """Compute temporal stability metrics and generate scatter plot for sites."""
    from src.analysis.stability_metrics import compute_stability_metrics, plot_stability_scatter

    def _work():
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(diagnostics_dir, exist_ok=True)
        prefix = _entity_prefix(yearly_df)
        site_stability = compute_stability_metrics(
            yearly_df, entity_col="name",
            output_csv=os.path.join(csv_dir, f"{prefix}_stability_metrics.csv"),
        )
        plot_stability_scatter(
            site_stability, entity_col="name",
            output_path=os.path.join(diagnostics_dir, f"{prefix}_stability_scatter.png"),
        )
        return site_stability

    return run_step(
        "site_stability", _work,
        input_summary={"sites": yearly_df["name"].nunique()},
        output_summary_fn=lambda df: {"metrics_computed": len(df) if df is not None else 0},
    )


def step_site_breakpoints(
    yearly_df: pd.DataFrame, csv_dir: str
) -> tuple:
    """Detect temporal breakpoints in site radiance trends."""
    from src.analysis.breakpoint_analysis import analyze_all_breakpoints

    def _work():
        os.makedirs(csv_dir, exist_ok=True)
        prefix = _entity_prefix(yearly_df)
        return analyze_all_breakpoints(
            yearly_df, entity_col="name",
            output_csv=os.path.join(csv_dir, f"{prefix}_breakpoints.csv"),
        )

    return run_step(
        "site_breakpoints", _work,
        input_summary={"sites": yearly_df["name"].nunique()},
        output_summary_fn=lambda df: {"breakpoints_found": len(df) if df is not None else 0},
    )


def step_site_benchmark(trends_df: pd.DataFrame, csv_dir: str) -> tuple:
    """Compare site trends to benchmark locations."""
    from src.analysis.benchmark_comparison import compare_to_benchmarks

    def _work():
        os.makedirs(csv_dir, exist_ok=True)
        prefix = _entity_prefix(trends_df)
        compare_to_benchmarks(
            trends_df,
            output_csv=os.path.join(csv_dir, f"{prefix}_benchmark_comparison.csv"),
        )
        return None

    return run_step(
        "site_benchmark", _work,
        input_summary={"trends_count": len(trends_df)},
    )


def step_site_reports(
    latest_metrics: pd.DataFrame, yearly_df: pd.DataFrame, reports_dir: str,
    entity_type: str = "all"
) -> tuple:
    """Generate individual PDF reports for each site."""
    from src.outputs.site_reports import generate_all_site_reports

    def _work():
        os.makedirs(reports_dir, exist_ok=True)
        generate_all_site_reports(
            all_site_data=latest_metrics,
            yearly_df=yearly_df,
            output_dir=reports_dir,
        )
        return None

    return run_step(
        "site_reports", _work,
        input_summary={"sites": len(latest_metrics)},
    )
