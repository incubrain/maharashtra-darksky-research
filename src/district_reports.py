"""
District-level deep-dive PDF reports.

Generates a comprehensive 4-page PDF report for each of the 36 districts.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import config

log = logging.getLogger(__name__)


def generate_district_report(district_name, yearly_df, trends_df, stability_df,
                             gdf, output_dir):
    """Generate a multi-page PDF report for one district."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = district_name.replace(" ", "_").replace("/", "_")
    pdf_path = os.path.join(output_dir, f"{safe_name}_report.pdf")

    sub = yearly_df[yearly_df["district"] == district_name].sort_values("year")
    trend_row = trends_df[trends_df["district"] == district_name]
    stab_row = stability_df[stability_df["district"] == district_name] if stability_df is not None else None

    with PdfPages(pdf_path) as pdf:
        # Page 1: Summary
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        lines = [
            f"District Report: {district_name}",
            "=" * 50,
            "",
        ]
        if not trend_row.empty:
            r = trend_row.iloc[0]
            lines += [
                f"Latest Median Radiance: {r.get('median_radiance_latest', 'N/A'):.2f} nW/cm²/sr",
                f"ALAN Classification: {r.get('alan_class', 'N/A')}",
                f"Annual % Change: {r.get('annual_pct_change', np.nan):+.2f}%",
                f"95% CI: [{r.get('ci_low', np.nan):+.2f}%, {r.get('ci_high', np.nan):+.2f}%]",
                f"R²: {r.get('r_squared', np.nan):.3f}",
                f"p-value: {r.get('p_value', np.nan):.2e}",
                f"Years of data: {r.get('n_years', 'N/A')}",
                "",
            ]
        if stab_row is not None and not stab_row.empty:
            s = stab_row.iloc[0]
            lines += [
                "Stability Metrics:",
                f"  Coefficient of Variation: {s.get('coefficient_of_variation', np.nan):.3f}",
                f"  Stability Class: {s.get('stability_class', 'N/A')}",
                f"  Max Year-to-Year Change: {s.get('max_year_to_year_change', np.nan):.3f} nW",
                "",
            ]

        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Time series
        if len(sub) >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            years = sub["year"].values.astype(float)
            radiance = sub["median_radiance"].values

            ax.plot(years, radiance, "bo-", markersize=5, label="Observed")

            # Fitted trend
            log_rad = np.log(radiance + config.LOG_EPSILON)
            X = sm.add_constant(years)
            try:
                model = sm.OLS(log_rad, X).fit()
                fitted = np.exp(model.fittedvalues)
                ax.plot(years, fitted, "r--", linewidth=2, label="Fitted trend")
            except Exception:
                pass

            ax.set_xlabel("Year", fontsize=12)
            ax.set_ylabel("Median Radiance (nW/cm²/sr)", fontsize=12)
            ax.set_title(f"{district_name}: ALAN Time Series (2012-2024)", fontsize=14)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Page 3: Spatial context
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf.plot(ax=ax, color="whitesmoke", edgecolor="grey", linewidth=0.5)
        highlight = gdf[gdf["district"] == district_name]
        if not highlight.empty:
            highlight.plot(ax=ax, color="steelblue", edgecolor="navy", linewidth=1.5)
        for _, row in gdf.iterrows():
            c = row.geometry.centroid
            fontweight = "bold" if row["district"] == district_name else "normal"
            fontsize = 7 if row["district"] == district_name else 5
            ax.annotate(row["district"], xy=(c.x, c.y), fontsize=fontsize,
                        ha="center", va="center", fontweight=fontweight)
        ax.set_title(f"{district_name} within Maharashtra", fontsize=14)
        ax.set_axis_off()
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: Quality timeline
        if len(sub) >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(sub["year"], sub["pixel_count"], color="steelblue", alpha=0.7)
            ax.set_xlabel("Year", fontsize=12)
            ax.set_ylabel("Valid Pixel Count", fontsize=12)
            ax.set_title(f"{district_name}: Valid Pixels per Year", fontsize=14)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    log.info("Generated report: %s", pdf_path)
    return pdf_path


def generate_all_district_reports(yearly_df, trends_df, stability_df, gdf,
                                  output_dir):
    """Generate reports for all districts."""
    report_dir = output_dir
    os.makedirs(report_dir, exist_ok=True)

    districts = trends_df["district"].unique()
    for district in districts:
        generate_district_report(
            district_name=district,
            yearly_df=yearly_df,
            trends_df=trends_df,
            stability_df=stability_df,
            gdf=gdf,
            output_dir=report_dir,
        )
    log.info("Generated %d district reports in %s", len(districts), report_dir)
