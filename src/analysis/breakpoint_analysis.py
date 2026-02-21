"""
Trend breakpoint detection using piecewise linear regression.

Identifies years where ALAN growth rate changed significantly
(acceleration/deceleration) using AIC-based model selection.

KNOWN FINDING — Universal 2016 Breakpoint:
    In full diagnostic runs against VNL v2.1/v2.2 annual composites for
    Maharashtra (2012–2024), 34 out of 36 districts show a breakpoint in
    2016.  This is driven by TWO confounded factors:

    1. VIIRS product evolution:
       - 2012–2013: vcmcfg (no stray-light correction) — baseline noise
         is lower, broad-area radiance is systematically fainter.
       - 2014+:     vcmslcfg (stray-light corrected) — improved
         calibration and noise floor shift.
       - The 2016 detection is an artifact of the AIC model selecting the
         year that best separates the two radiometric regimes.

    2. Real-world events:
       - India's Deen Dayal Upadhyaya Gram Jyoti Yojana (DDUGJY) rural
         electrification program (2015–2019) and Ujala LED programme
         (2015+) drove genuine step-changes in nighttime radiance.

    IMPLICATION: A single-breakpoint model is too coarse for this data.
    Future work should consider:
       - Multi-breakpoint detection (BIC-penalised to avoid overfitting).
       - Separate trend fitting for pre-2014 vs post-2014 periods.
       - Including VIIRS product version as a covariate in the regression.
"""

from src.logging_config import get_pipeline_logger
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import config

log = get_pipeline_logger(__name__)


def detect_trend_breakpoints(yearly_df, district, entity_col="district"):
    """Detect changepoints in ALAN time series using piecewise linear regression.

    Args:
        yearly_df: DataFrame with [entity_col, year, median_radiance].
        district: Name of district/site to analyse.
        entity_col: Column name for filtering.

    Returns:
        Dict with breakpoint year, segment growth rates, significance.
    """
    sub = yearly_df[yearly_df[entity_col] == district].sort_values("year")
    years = sub["year"].values.astype(float)
    radiance = sub["median_radiance"].values.astype(float)

    if len(years) < 5:
        return {
            entity_col: district, "breakpoint_year": None,
            "growth_rate_before": np.nan, "growth_rate_after": np.nan,
            "p_value": np.nan, "aic_improvement": np.nan,
        }

    log_rad = np.log(radiance + config.LOG_EPSILON)

    # Fit single trend (baseline)
    X_single = sm.add_constant(years)
    model_single = sm.OLS(log_rad, X_single).fit()
    aic_single = model_single.aic

    # Test each potential breakpoint
    best_aic = aic_single
    best_bp = None
    best_result = None

    for bp_year in range(int(years[2]), int(years[-2]) + 1):
        mask_before = years <= bp_year
        mask_after = years > bp_year
        if mask_before.sum() < 2 or mask_after.sum() < 2:
            continue

        # Piecewise: add interaction term
        X_pw = np.column_stack([
            np.ones(len(years)),
            years,
            (years > bp_year).astype(float),
            years * (years > bp_year).astype(float),
        ])

        try:
            model_pw = sm.OLS(log_rad, X_pw).fit()
        except Exception:
            continue

        if model_pw.aic < best_aic:
            best_aic = model_pw.aic
            best_bp = bp_year
            best_result = model_pw

    if best_bp is None:
        beta_single = model_single.params[1]
        return {
            entity_col: district,
            "breakpoint_year": None,
            "growth_rate_before": round((np.exp(beta_single) - 1) * 100, 2),
            "growth_rate_after": round((np.exp(beta_single) - 1) * 100, 2),
            "p_value": np.nan,
            "aic_improvement": 0.0,
        }

    # Extract segment growth rates
    beta_before = best_result.params[1]
    beta_after = best_result.params[1] + best_result.params[3]
    growth_before = (np.exp(beta_before) - 1) * 100
    growth_after = (np.exp(beta_after) - 1) * 100

    # Test if interaction term is significant
    p_value = best_result.pvalues[3] if len(best_result.pvalues) > 3 else np.nan

    return {
        entity_col: district,
        "breakpoint_year": int(best_bp),
        "growth_rate_before": round(growth_before, 2),
        "growth_rate_after": round(growth_after, 2),
        "p_value": round(p_value, 6) if not np.isnan(p_value) else np.nan,
        "aic_improvement": round(aic_single - best_aic, 2),
    }


def analyze_all_breakpoints(yearly_df, entity_col="district", output_csv=None):
    """Run breakpoint detection for all districts/sites."""
    results = []
    for entity in yearly_df[entity_col].unique():
        result = detect_trend_breakpoints(yearly_df, entity, entity_col)
        results.append(result)

    df = pd.DataFrame(results)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info("Saved breakpoints: %s", output_csv)

    return df


def plot_breakpoint_timeline(breakpoint_df, output_path):
    """Timeline showing when districts experienced trend changes."""
    bp_years = breakpoint_df["breakpoint_year"].dropna().astype(int)

    if len(bp_years) == 0:
        log.info("No significant breakpoints detected; skipping timeline plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1.5])

    # Panel 1: Histogram of breakpoint years
    year_range = range(int(bp_years.min()), int(bp_years.max()) + 1)
    ax1.hist(bp_years, bins=len(year_range), color="steelblue", edgecolor="white",
             alpha=0.8, align="mid")
    ax1.set_xlabel("Breakpoint Year", fontsize=12)
    ax1.set_ylabel("Number of Districts", fontsize=12)
    ax1.set_title("Distribution of ALAN Trend Breakpoints", fontsize=14)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: Acceleration vs deceleration
    df_bp = breakpoint_df.dropna(subset=["breakpoint_year"]).copy()
    df_bp["change"] = df_bp["growth_rate_after"] - df_bp["growth_rate_before"]
    df_bp = df_bp.sort_values("change")

    entity_col = "district" if "district" in df_bp.columns else "name"
    colors = ["red" if c > 0 else "blue" for c in df_bp["change"]]
    ax2.barh(range(len(df_bp)), df_bp["change"], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(df_bp)))
    ax2.set_yticklabels(df_bp[entity_col], fontsize=7)
    ax2.set_xlabel("Change in Growth Rate (after - before breakpoint, %/yr)", fontsize=11)
    ax2.set_title("Growth Acceleration/Deceleration at Breakpoint", fontsize=14)
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)
