"""
Statewide comparison visualizations: publication-quality figures.

Creates multi-panel maps, classification choropleths, enhanced heatmaps,
and comparison charts for paper figures.
"""

from src.logging_config import get_pipeline_logger
import os

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

log = get_pipeline_logger(__name__)


def create_multi_year_comparison_grid(yearly_df, gdf, output_path,
                                     years=None):
    """5-panel choropleth grid showing district radiance across years."""
    if years is None:
        years = [2012, 2015, 2018, 2021, 2024]

    available = yearly_df["year"].unique()
    years = [y for y in years if y in available]
    if not years:
        log.warning("No matching years for comparison grid")
        return

    n = len(years)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))
    if n == 1:
        axes = [axes]

    # Consistent color scale
    vmin = yearly_df["median_radiance"].quantile(0.01)
    vmax = yearly_df["median_radiance"].quantile(0.99)

    for i, year in enumerate(years):
        ax = axes[i]
        year_data = yearly_df[yearly_df["year"] == year]
        merged = gdf.merge(year_data[["district", "median_radiance"]], on="district", how="left")
        merged.plot(column="median_radiance", ax=ax, cmap="YlOrRd",
                    vmin=vmin, vmax=vmax, edgecolor="black", linewidth=0.3,
                    missing_kwds={"color": "lightgrey"})
        ax.set_title(str(year), fontsize=12, fontweight="bold")
        ax.set_axis_off()

    fig.suptitle("Maharashtra: Median Radiance Evolution", fontsize=14, y=1.02)
    fig.colorbar(axes[-1].collections[0], ax=axes, shrink=0.6,
                 label="Median Radiance (nW/cm²/sr)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)


def create_growth_classification_map(trends_df, gdf, output_path):
    """Choropleth: districts classified by growth rate."""
    merged = gdf.merge(trends_df, on="district", how="left")

    def classify_growth(pct):
        if pd.isna(pct):
            return "No data"
        elif pct < 0:
            return "Declining (<0%)"
        elif pct < 5:
            return "Slow (0-5%)"
        elif pct < 10:
            return "Moderate (5-10%)"
        else:
            return "Rapid (>10%)"

    merged["growth_class"] = merged["annual_pct_change"].apply(classify_growth)

    color_map = {
        "Declining (<0%)": "#2166ac",
        "Slow (0-5%)": "#fdbf6f",
        "Moderate (5-10%)": "#f46d43",
        "Rapid (>10%)": "#a50026",
        "No data": "#d3d3d3",
    }

    fig, ax = plt.subplots(figsize=(12, 10))
    for cls, color in color_map.items():
        subset = merged[merged["growth_class"] == cls]
        if not subset.empty:
            subset.plot(ax=ax, color=color, edgecolor="black", linewidth=0.5,
                        label=cls)

    for _, row in merged.iterrows():
        c = row.geometry.centroid
        ax.annotate(row["district"], xy=(c.x, c.y), fontsize=5,
                    ha="center", va="center")

    ax.legend(loc="lower left", fontsize=9, title="Growth Rate")
    ax.set_title("Maharashtra: ALAN Growth Classification", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)


def create_data_quality_map(quality_summary_df, gdf, output_path):
    """Choropleth: mean quality % per district."""
    if quality_summary_df is None or quality_summary_df.empty:
        log.warning("No quality data for quality map")
        return

    avg_quality = quality_summary_df.groupby("district")["quality_percentage"].mean().reset_index()
    merged = gdf.merge(avg_quality, on="district", how="left")

    fig, ax = plt.subplots(figsize=(12, 10))
    merged.plot(column="quality_percentage", ax=ax, cmap="RdYlGn",
                legend=True, edgecolor="black", linewidth=0.5,
                legend_kwds={"label": "Mean Quality %"},
                missing_kwds={"color": "lightgrey"})
    for _, row in merged.iterrows():
        c = row.geometry.centroid
        ax.annotate(row["district"], xy=(c.x, c.y), fontsize=5,
                    ha="center", va="center")
    ax.set_title("Maharashtra: Data Quality (Mean % Passing All Filters)", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)


def create_enhanced_radiance_heatmap(yearly_df, output_path):
    """Enhanced heatmap with log scale and better formatting."""
    pivot = yearly_df.pivot_table(
        index="district", columns="year", values="median_radiance"
    )
    # Log transform for better visibility
    pivot_log = np.log10(pivot.clip(lower=0.01))
    pivot_log = pivot_log.sort_values(pivot_log.columns[-1], ascending=False)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(pivot_log.values, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(pivot_log.columns)))
    ax.set_xticklabels(pivot_log.columns.astype(int), rotation=45, fontsize=8)
    ax.set_yticks(range(len(pivot_log.index)))
    ax.set_yticklabels(pivot_log.index, fontsize=7)
    ax.set_title("Median Radiance Heatmap (log₁₀ scale)", fontsize=14)
    plt.colorbar(im, ax=ax, label="log₁₀(Median Radiance nW/cm²/sr)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)


def create_city_vs_site_boxplot(metrics_df, output_path):
    """Side-by-side boxplots comparing cities vs. dark-sky sites."""
    if "type" not in metrics_df.columns:
        log.warning("No 'type' column in metrics_df for boxplot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    types = ["city", "site"]
    data = [metrics_df[metrics_df["type"] == t]["median_radiance"].dropna().values
            for t in types]
    bp = ax.boxplot(data, labels=["Cities", "Dark-Sky Sites"], patch_artist=True)
    bp["boxes"][0].set_facecolor("crimson")
    bp["boxes"][0].set_alpha(0.5)
    bp["boxes"][1].set_facecolor("forestgreen")
    bp["boxes"][1].set_alpha(0.5)

    ax.set_yscale("log")
    ax.axhline(y=config.ALAN_LOW_THRESHOLD, color="orange", linestyle="--",
               linewidth=1, label=f"Low-ALAN threshold ({config.ALAN_LOW_THRESHOLD} nW)")
    ax.set_ylabel("Median Radiance (nW/cm²/sr, log scale)", fontsize=12)
    ax.set_title("ALAN Distribution: Cities vs. Dark-Sky Sites", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)
