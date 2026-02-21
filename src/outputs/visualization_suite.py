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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

log = get_pipeline_logger(__name__)


def create_multi_year_comparison_grid(yearly_df, gdf, output_path,
                                     years=None):
    """Multi-panel choropleth grid showing district radiance across years."""
    if years is None:
        years = [2012, 2015, 2018, 2021, 2024]

    available = yearly_df["year"].unique()
    years = [y for y in years if y in available]
    if not years:
        log.warning("No matching years for comparison grid")
        return

    n = len(years)
    # Use 2-row grid for 4+ panels so maps are large enough to read
    if n <= 3:
        nrows, ncols = 1, n
    else:
        nrows = 2
        ncols = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 7 * nrows))
    axes_flat = list(np.asarray(axes).flat) if n > 1 else [axes]

    # Hide unused axes
    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Log-scale color mapping for better visibility across dynamic range
    from matplotlib.colors import LogNorm
    vmin = max(yearly_df["median_radiance"].quantile(0.01), 0.1)
    vmax = yearly_df["median_radiance"].quantile(0.99)

    for i, year in enumerate(years):
        ax = axes_flat[i]
        year_data = yearly_df[yearly_df["year"] == year]
        merged = gdf.merge(year_data[["district", "median_radiance"]], on="district", how="left")
        merged.plot(column="median_radiance", ax=ax, cmap="YlOrRd",
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                    edgecolor="black", linewidth=0.3,
                    missing_kwds={"color": "lightgrey"})
        ax.set_title(str(year), fontsize=13, fontweight="bold")
        ax.set_axis_off()

    fig.suptitle("Maharashtra: Median Radiance Evolution (log scale)", fontsize=16, y=0.98)
    fig.colorbar(axes_flat[n - 1].collections[0], ax=axes_flat[:n], shrink=0.6,
                 label="Median Radiance (nW/cm²/sr, log scale)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)


def create_growth_classification_map(trends_df, gdf, output_path):
    """Choropleth: districts classified by growth rate.

    Uses adaptive thresholds based on the data distribution (quartiles)
    so that the map always shows meaningful variation across districts.
    """
    merged = gdf.merge(trends_df, on="district", how="left")

    # Compute adaptive thresholds from actual data distribution
    valid_pcts = merged["annual_pct_change"].dropna()
    if len(valid_pcts) > 0 and valid_pcts.min() >= 0:
        # All-positive growth: use quartile-based adaptive bins
        q25 = valid_pcts.quantile(0.25)
        q50 = valid_pcts.quantile(0.50)
        q75 = valid_pcts.quantile(0.75)

        def classify_growth(pct):
            if pd.isna(pct):
                return "No data"
            elif pct < q25:
                return f"Low (<{q25:.1f}%)"
            elif pct < q50:
                return f"Moderate ({q25:.1f}-{q50:.1f}%)"
            elif pct < q75:
                return f"High ({q50:.1f}-{q75:.1f}%)"
            else:
                return f"Rapid (>{q75:.1f}%)"

        color_map = {
            f"Low (<{q25:.1f}%)": "#ffffb2",
            f"Moderate ({q25:.1f}-{q50:.1f}%)": "#fecc5c",
            f"High ({q50:.1f}-{q75:.1f}%)": "#fd8d3c",
            f"Rapid (>{q75:.1f}%)": "#e31a1c",
            "No data": "#d3d3d3",
        }
    else:
        # Mixed positive/negative: use fixed threshold scheme
        def classify_growth(pct):
            if pd.isna(pct):
                return "No data"
            elif pct < 0:
                return "Declining (<0%)"
            elif pct < 2:
                return "Slow (0-2%)"
            elif pct < 5:
                return "Moderate (2-5%)"
            else:
                return "Rapid (>5%)"

        color_map = {
            "Declining (<0%)": "#2166ac",
            "Slow (0-2%)": "#ffffb2",
            "Moderate (2-5%)": "#fd8d3c",
            "Rapid (>5%)": "#e31a1c",
            "No data": "#d3d3d3",
        }

    merged["growth_class"] = merged["annual_pct_change"].apply(classify_growth)

    fig, ax = plt.subplots(figsize=(12, 10))
    for cls, color in color_map.items():
        subset = merged[merged["growth_class"] == cls]
        if not subset.empty:
            subset.plot(ax=ax, color=color, edgecolor="black", linewidth=0.5)

    for _, row in merged.iterrows():
        c = row.geometry.centroid
        ax.annotate(row["district"], xy=(c.x, c.y), fontsize=5,
                    ha="center", va="center")

    handles = [mpatches.Patch(color=color, label=cls)
               for cls, color in color_map.items()
               if not merged[merged["growth_class"] == cls].empty]
    ax.legend(handles=handles, loc="lower left", fontsize=9, title="Growth Rate")
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


