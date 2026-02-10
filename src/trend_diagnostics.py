"""
Trend model diagnostics: residual analysis, outlier detection, model validation.

Validates log-linear trend model assumptions and identifies anomalous years.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

from src import config

log = logging.getLogger(__name__)


def compute_trend_diagnostics(yearly_df, district, entity_col="district"):
    """Comprehensive diagnostics for log-linear trend model.

    Args:
        yearly_df: DataFrame with yearly radiance data.
        district: District/site name.
        entity_col: Column for filtering.

    Returns:
        Dict with diagnostic metrics.
    """
    sub = yearly_df[yearly_df[entity_col] == district].sort_values("year")
    years = sub["year"].values.astype(float)
    radiance = sub["median_radiance"].values.astype(float)

    if len(years) < 4:
        return {
            entity_col: district,
            "r_squared": np.nan, "adj_r_squared": np.nan,
            "durbin_watson": np.nan, "jarque_bera_p": np.nan,
            "cooks_distance_max": np.nan, "outlier_years": [],
            "model_warnings": ["insufficient data (<4 years)"],
        }

    log_rad = np.log(radiance + config.LOG_EPSILON)
    X = sm.add_constant(years)
    model = sm.OLS(log_rad, X).fit()

    # Residuals
    residuals = model.resid
    std_resid = (residuals - residuals.mean()) / residuals.std()

    # Durbin-Watson (autocorrelation)
    from statsmodels.stats.stattools import durbin_watson
    dw = durbin_watson(residuals)

    # Jarque-Bera (normality)
    try:
        jb_stat, jb_p = scipy_stats.jarque_bera(residuals)
    except Exception:
        jb_p = np.nan

    # Cook's distance
    try:
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        max_cooks = np.max(cooks_d)
    except Exception:
        cooks_d = np.zeros(len(years))
        max_cooks = 0.0

    # Outlier years (|standardized residual| > 2)
    outlier_mask = np.abs(std_resid) > 2
    outlier_years = sub["year"].values[outlier_mask].tolist()

    # Model warnings
    warnings_list = []
    if dw < 1.0 or dw > 3.0:
        warnings_list.append("high autocorrelation (DW={:.2f})".format(dw))
    if jb_p is not None and not np.isnan(jb_p) and jb_p < 0.05:
        warnings_list.append("non-normal residuals (JB p={:.3f})".format(jb_p))
    if max_cooks > 1.0:
        warnings_list.append("high-influence observation (Cook's D={:.2f})".format(max_cooks))
    if model.rsquared < 0.5:
        warnings_list.append("poor model fit (R²={:.3f})".format(model.rsquared))

    return {
        entity_col: district,
        "r_squared": round(model.rsquared, 4),
        "adj_r_squared": round(model.rsquared_adj, 4),
        "durbin_watson": round(dw, 4),
        "jarque_bera_p": round(jb_p, 4) if not np.isnan(jb_p) else np.nan,
        "cooks_distance_max": round(max_cooks, 4),
        "outlier_years": outlier_years,
        "model_warnings": warnings_list,
    }


def compute_all_diagnostics(yearly_df, entity_col="district", output_csv=None):
    """Compute diagnostics for all districts/sites."""
    results = []
    for entity in yearly_df[entity_col].unique():
        diag = compute_trend_diagnostics(yearly_df, entity, entity_col)
        # Flatten lists to strings for CSV
        diag_flat = diag.copy()
        diag_flat["outlier_years"] = "; ".join(str(y) for y in diag["outlier_years"])
        diag_flat["model_warnings"] = "; ".join(diag["model_warnings"])
        results.append(diag_flat)

    df = pd.DataFrame(results)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info("Saved diagnostics: %s", output_csv)

    return df


def plot_diagnostic_panel(yearly_df, district, entity_col="district",
                          output_path=None):
    """4-panel diagnostic plot for one district.

    Panel 1: Time series with fitted trend line
    Panel 2: Residuals vs. fitted
    Panel 3: Q-Q plot
    Panel 4: Cook's distance
    """
    sub = yearly_df[yearly_df[entity_col] == district].sort_values("year")
    years = sub["year"].values.astype(float)
    radiance = sub["median_radiance"].values.astype(float)

    if len(years) < 4:
        return

    log_rad = np.log(radiance + config.LOG_EPSILON)
    X = sm.add_constant(years)
    model = sm.OLS(log_rad, X).fit()
    fitted = model.fittedvalues
    residuals = model.resid

    try:
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
    except Exception:
        cooks_d = np.zeros(len(years))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Time series + trend
    ax = axes[0, 0]
    ax.plot(years, log_rad, "bo-", markersize=5, label="Observed (log)")
    ax.plot(years, fitted, "r--", linewidth=2, label="Fitted trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("log(Radiance)")
    ax.set_title(f"{district}: Time Series + Trend")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Residuals vs fitted
    ax = axes[0, 1]
    ax.scatter(fitted, residuals, c="steelblue", edgecolors="grey", s=40)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    ax.grid(True, alpha=0.3)

    # Panel 3: Q-Q plot
    ax = axes[1, 0]
    scipy_stats.probplot(residuals, plot=ax)
    ax.set_title("Normal Q-Q Plot")
    ax.grid(True, alpha=0.3)

    # Panel 4: Cook's distance
    ax = axes[1, 1]
    ax.bar(years.astype(int), cooks_d, color="steelblue", edgecolor="grey")
    ax.axhline(y=4 / len(years), color="red", linestyle="--", linewidth=1,
               label=f"Threshold (4/n = {4/len(years):.2f})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cook's Distance")
    ax.set_title("Influential Observations")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Trend Diagnostics: {district} (R²={model.rsquared:.3f})",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=config.MAP_DPI, bbox_inches="tight")
        log.info("Saved diagnostics plot: %s", output_path)
    plt.close(fig)
