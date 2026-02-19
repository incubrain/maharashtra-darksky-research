"""
ALAN graduated percentile-based classification tiers.

Instead of fixed thresholds, classifies districts/sites by their
percentile rank within the statewide radiance distribution, providing
relative context for ALAN levels.
"""

from src.logging_config import get_pipeline_logger
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.formulas.classification import TIER_COLORS

log = get_pipeline_logger(__name__)


def classify_by_percentiles(yearly_df, year=None, bins=None, labels=None,
                            output_csv=None):
    """Classify districts into graduated ALAN tiers by percentile rank.

    Args:
        yearly_df: DataFrame with [district, year, median_radiance].
        year: Year to classify (default: latest available).
        bins: Percentile bin edges (default from config).
        labels: Tier labels (default from config).
        output_csv: Optional output path.

    Returns:
        DataFrame with district, median_radiance, percentile_rank,
        alan_tier, and tier boundaries.
    """
    if bins is None:
        bins = config.ALAN_PERCENTILE_BINS
    if labels is None:
        labels = config.ALAN_PERCENTILE_LABELS

    if year is None:
        year = yearly_df["year"].max()

    sub = yearly_df[yearly_df["year"] == year].copy()
    if sub.empty:
        log.warning("No data for year %d", year)
        return pd.DataFrame()

    radiance = sub["median_radiance"].dropna()
    if len(radiance) < 5:
        log.warning("Fewer than 5 districts with data for year %d", year)
        return pd.DataFrame()

    # Compute percentile thresholds
    thresholds = np.percentile(radiance, bins)

    # Assign tiers
    sub["percentile_rank"] = sub["median_radiance"].rank(pct=True) * 100
    sub["alan_tier"] = pd.cut(
        sub["percentile_rank"], bins=bins, labels=labels, include_lowest=True
    )

    # Add threshold info
    tier_info = []
    for i in range(len(labels)):
        tier_info.append({
            "tier": labels[i],
            "percentile_low": bins[i],
            "percentile_high": bins[i + 1],
            "radiance_low": round(thresholds[i], 2),
            "radiance_high": round(thresholds[i + 1], 2),
        })
    tier_df = pd.DataFrame(tier_info)

    result = sub[["district", "median_radiance", "percentile_rank", "alan_tier"]].copy()
    result["year"] = year
    result["percentile_rank"] = result["percentile_rank"].round(1)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        result.to_csv(output_csv, index=False)
        tier_csv = output_csv.replace(".csv", "_tiers.csv")
        tier_df.to_csv(tier_csv, index=False)
        log.info("Saved graduated classification: %s", output_csv)

    return result


def classify_temporal_trajectory(yearly_df, output_csv=None):
    """Track how each district's percentile tier changes over time.

    Args:
        yearly_df: DataFrame with [district, year, median_radiance].
        output_csv: Optional output path.

    Returns:
        DataFrame with district, year, tier for each year.
    """
    all_results = []
    for year in sorted(yearly_df["year"].unique()):
        classified = classify_by_percentiles(yearly_df, year=year)
        if not classified.empty:
            all_results.append(classified)

    if not all_results:
        return pd.DataFrame()

    df = pd.concat(all_results, ignore_index=True)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info("Saved temporal trajectory: %s", output_csv)

    return df


def plot_tier_distribution(classified_df, output_path):
    """Stacked bar chart showing tier distribution over time.

    Args:
        classified_df: Output from classify_temporal_trajectory().
        output_path: Path to save figure.
    """
    if classified_df.empty or "alan_tier" not in classified_df.columns:
        log.warning("No tier data for distribution plot")
        return

    tier_counts = classified_df.groupby(["year", "alan_tier"]).size().unstack(fill_value=0)

    labels = config.ALAN_PERCENTILE_LABELS
    tier_colors = TIER_COLORS

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(tier_counts))

    for tier in labels:
        if tier in tier_counts.columns:
            values = tier_counts[tier].values
            ax.bar(tier_counts.index.astype(int), values, bottom=bottom,
                   color=tier_colors.get(tier, "grey"), label=tier,
                   edgecolor="white", linewidth=0.3)
            bottom += values

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Districts", fontsize=12)
    ax.set_title("ALAN Tier Distribution Over Time (Percentile-Based)", fontsize=14)
    ax.legend(loc="upper left", fontsize=9, title="ALAN Tier")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)


def plot_tier_transition_matrix(classified_df, year_start, year_end, output_path):
    """Heatmap showing how many districts moved between tiers.

    Args:
        classified_df: Output from classify_temporal_trajectory().
        year_start: Start year for comparison.
        year_end: End year for comparison.
        output_path: Path to save figure.
    """
    labels = config.ALAN_PERCENTILE_LABELS

    start = classified_df[classified_df["year"] == year_start][["district", "alan_tier"]]
    end = classified_df[classified_df["year"] == year_end][["district", "alan_tier"]]

    if start.empty or end.empty:
        log.warning("Insufficient data for transition matrix (%d -> %d)", year_start, year_end)
        return

    merged = start.merge(end, on="district", suffixes=("_start", "_end"))

    matrix = pd.crosstab(
        merged["alan_tier_start"], merged["alan_tier_end"],
        rownames=[f"Tier {year_start}"], colnames=[f"Tier {year_end}"]
    )

    # Reindex to ensure all tiers present
    matrix = matrix.reindex(index=labels, columns=labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix.values[i, j]
            if val > 0:
                ax.text(j, i, str(int(val)), ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if val > matrix.values.max() * 0.5 else "black")

    ax.set_xlabel(f"Tier in {year_end}", fontsize=12)
    ax.set_ylabel(f"Tier in {year_start}", fontsize=12)
    ax.set_title(f"ALAN Tier Transitions: {year_start} → {year_end}", fontsize=14)
    plt.colorbar(im, ax=ax, label="Number of Districts")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)
