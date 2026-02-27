"""
Temporal stability metrics for ALAN time series.

Quantifies year-to-year variability to identify stable vs. erratic
districts/sites. Dark-sky sites need STABLE low-ALAN for certification.
"""

from src.logging_config import get_pipeline_logger
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.formulas.classification import classify_stability

log = get_pipeline_logger(__name__)


def compute_stability_metrics(yearly_df, entity_col="district", output_csv=None):
    """Compute temporal stability metrics for each district/site.

    Includes skewness and kurtosis following Small & Elvidge (2022) who
    demonstrate that multi-moment analysis (mean + variance + skewness +
    kurtosis) better distinguishes physically distinct lighting zones than
    single-metric CV classification alone.

    Ref: Small, C. & Elvidge, C.D. (2022). Mapping decadal change in
    anthropogenic night light. Sensors, 22(12), 4459.

    Args:
        yearly_df: DataFrame with [entity_col, year, median_radiance].
        entity_col: Column name for grouping ("district" or "name").
        output_csv: Path to save stability metrics.

    Returns:
        DataFrame with columns: [entity, mean_radiance_2012_2024,
        std_radiance, coefficient_of_variation, skewness, kurtosis,
        iqr, max_year_to_year_change, stability_class].
    """
    from scipy import stats as sp_stats

    results = []
    for entity in yearly_df[entity_col].unique():
        sub = yearly_df[yearly_df[entity_col] == entity].sort_values("year")
        values = sub["median_radiance"].dropna().values

        if len(values) < 2:
            results.append({entity_col: entity})
            continue

        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        cv = std_val / mean_val if mean_val > 0 else np.nan
        q25, q75 = np.percentile(values, [25, 75])
        iqr = q75 - q25
        diffs = np.abs(np.diff(values))
        max_change = np.max(diffs) if len(diffs) > 0 else 0.0

        # Higher moments (finding SE2): skewness and kurtosis provide
        # additional distributional information that CV alone discards.
        # Positive skew = growth acceleration; high kurtosis = outlier years.
        skewness = float(sp_stats.skew(values, bias=False)) if len(values) >= 3 else np.nan
        kurtosis = float(sp_stats.kurtosis(values, bias=False)) if len(values) >= 4 else np.nan

        # Classify stability
        stability = classify_stability(cv)

        # LED transition flag (findings F4, KY3, B5): if post-2015 data
        # shows radiance *decrease*, it may reflect HPS→LED transition rather
        # than genuine ALAN reduction. VIIRS DNB (500-900 nm) is blind to
        # blue LED emissions, so LED adoption can appear as dimming.
        # Ref: Kyba et al. (2017), Science Advances, 3(11), e1701528.
        led_flag = False
        pre_vals = sub.loc[sub["year"] < 2015, "median_radiance"].dropna().values
        post_vals = sub.loc[sub["year"] >= 2015, "median_radiance"].dropna().values
        if len(pre_vals) > 0 and len(post_vals) > 0:
            if np.mean(post_vals) < np.mean(pre_vals):
                led_flag = True

        results.append({
            entity_col: entity,
            "mean_radiance_2012_2024": round(mean_val, 4),
            "std_radiance": round(std_val, 4),
            "coefficient_of_variation": round(cv, 4),
            "skewness": round(skewness, 4) if not np.isnan(skewness) else np.nan,
            "kurtosis": round(kurtosis, 4) if not np.isnan(kurtosis) else np.nan,
            "iqr": round(iqr, 4),
            "max_year_to_year_change": round(max_change, 4),
            "stability_class": stability,
            "possible_led_transition": led_flag,
        })

    df = pd.DataFrame(results)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info("Saved stability metrics: %s", output_csv)

    return df


def plot_stability_scatter(stability_df, entity_col="district", output_path=None):
    """Scatter plot: mean radiance vs. coefficient of variation."""
    df = stability_df.dropna(subset=["mean_radiance_2012_2024", "coefficient_of_variation"])

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {"stable": "green", "moderate": "orange", "erratic": "red"}
    for cls, color in colors.items():
        subset = df[df["stability_class"] == cls]
        ax.scatter(subset["mean_radiance_2012_2024"],
                   subset["coefficient_of_variation"],
                   c=color, label=cls.capitalize(), s=60, alpha=0.7, edgecolors="grey")
        # Annotate outliers (high CV)
        for _, row in subset.iterrows():
            if row["coefficient_of_variation"] > 0.4:
                ax.annotate(row[entity_col], (row["mean_radiance_2012_2024"],
                            row["coefficient_of_variation"]),
                            fontsize=7, ha="left", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Mean Radiance 2012-2024 (nW/cm²/sr, log scale)", fontsize=12)
    ax.set_ylabel("Coefficient of Variation", fontsize=12)
    ax.set_title("ALAN Temporal Stability: Mean Radiance vs. Variability", fontsize=14)
    ax.axhline(y=0.2, color="green", linestyle=":", alpha=0.5, label="CV=0.2 (stable)")
    ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.5, label="CV=0.5 (erratic)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
        log.info("Saved: %s", output_path)
    plt.close(fig)
