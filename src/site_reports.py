"""
Site-level deep-dive PDF reports.

Generates multi-page PDF reports for 5 cities and 11 dark-sky candidate sites.
"""

from src.logging_config import get_pipeline_logger
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from src import config

log = get_pipeline_logger(__name__)


def generate_site_report(site_name, site_type, metrics_df, yearly_df,
                         buffer_comparison_df=None, directional_df=None,
                         proximity_df=None, output_dir="."):
    """Generate multi-page PDF report for one site."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = site_name.replace(" ", "_").replace("/", "_")
    pdf_path = os.path.join(output_dir, f"{safe_name}_report.pdf")

    site_metrics = metrics_df[metrics_df["name"] == site_name] if "name" in metrics_df.columns else pd.DataFrame()
    site_yearly = yearly_df[yearly_df["name"] == site_name].sort_values("year") if "name" in yearly_df.columns else pd.DataFrame()

    with PdfPages(pdf_path) as pdf:
        # Page 1: Summary
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        lines = [
            f"Site Report: {site_name}",
            f"Type: {site_type}",
            "=" * 50,
            "",
        ]
        if not site_metrics.empty:
            r = site_metrics.iloc[-1]
            lines += [
                f"District: {r.get('district', 'N/A')}",
                f"Median Radiance: {r.get('median_radiance', np.nan):.3f} nW/cm²/sr",
                f"Mean Radiance: {r.get('mean_radiance', np.nan):.3f} nW/cm²/sr",
                f"ALAN Class: {r.get('alan_class', 'N/A')}",
                f"Valid Pixels: {r.get('valid_pixels', 'N/A')}",
                f"Quality: {r.get('quality_pct', 'N/A')}%",
                "",
            ]

        # Add proximity info
        if proximity_df is not None and not proximity_df.empty:
            prox = proximity_df[proximity_df["site"] == site_name]
            if not prox.empty:
                p = prox.iloc[0]
                lines += [
                    "Nearest City:",
                    f"  {p.get('nearest_city_name', 'N/A')} ({p.get('distance_km', 'N/A')} km, {p.get('cardinal_direction', '')})",
                    "",
                ]

        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Time series
        if len(site_yearly) >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(site_yearly["year"], site_yearly["median_radiance"], "o-",
                    color="steelblue", markersize=5, linewidth=2)
            if site_type == "site":
                ax.axhline(y=config.ALAN_LOW_THRESHOLD, color="orange",
                           linestyle="--", linewidth=1,
                           label=f"Low-ALAN threshold ({config.ALAN_LOW_THRESHOLD} nW)")
            ax.set_xlabel("Year", fontsize=12)
            ax.set_ylabel("Median Radiance (nW/cm²/sr)", fontsize=12)
            ax.set_title(f"{site_name}: ALAN Time Series", fontsize=14)
            if ax.get_legend_handles_labels()[1]:
                ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Page 3: Buffer comparison (if available)
        if buffer_comparison_df is not None and not buffer_comparison_df.empty:
            buf = buffer_comparison_df[buffer_comparison_df["site"] == site_name]
            if not buf.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                b = buf.iloc[0]
                vals = [b.get("inside_median", 0) or 0, b.get("outside_median", 0) or 0]
                ax.bar(["Inside Buffer", "Outside Buffer"], vals,
                       color=["forestgreen", "orange"], alpha=0.7)
                ax.axhline(y=config.ALAN_LOW_THRESHOLD, color="red",
                           linestyle="--", linewidth=1)
                ax.set_ylabel("Median Radiance (nW/cm²/sr)", fontsize=12)
                ax.set_title(f"{site_name}: Inside vs Outside Buffer", fontsize=14)
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        # Page 4 (dark-sky sites only): Recommendations
        if site_type == "site":
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            rec_lines = [
                f"Dark-Sky Suitability Assessment: {site_name}",
                "=" * 50,
                "",
            ]
            if not site_metrics.empty:
                r = site_metrics.iloc[-1]
                med = r.get("median_radiance", np.nan)
                if not np.isnan(med) and med < config.ALAN_LOW_THRESHOLD:
                    rec_lines.append("SUITABLE: Median radiance below dark-sky threshold")
                    rec_lines.append(f"  Current: {med:.3f} nW/cm²/sr (threshold: {config.ALAN_LOW_THRESHOLD})")
                else:
                    rec_lines.append("NOT SUITABLE: Median radiance above dark-sky threshold")
                    rec_lines.append(f"  Current: {med:.3f} nW/cm²/sr")
                rec_lines.append("")

            # Directional info
            if directional_df is not None and not directional_df.empty:
                d = directional_df[directional_df["site"] == site_name]
                if not d.empty:
                    dd = d.iloc[0]
                    rec_lines.append(f"Dominant light direction: {dd.get('dominant_direction', 'N/A')}")
                    rec_lines.append(f"Directional ratio: {dd.get('max_min_ratio', 'N/A')}")
                    rec_lines.append("")
                    rec_lines.append("Mitigation: Consider shielding/zoning in the "
                                     f"{dd.get('dominant_direction', '')} direction")

            ax.text(0.05, 0.95, "\n".join(rec_lines), transform=ax.transAxes,
                    fontsize=11, verticalalignment="top", fontfamily="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    log.info("Generated site report: %s", pdf_path)
    return pdf_path


def generate_all_site_reports(all_site_data, yearly_df, output_dir,
                              buffer_comparison_df=None, directional_df=None,
                              proximity_df=None):
    """Generate a single consolidated PDF atlas with all site reports."""
    os.makedirs(output_dir, exist_ok=True)
    atlas_path = os.path.join(output_dir, "site_atlas.pdf")

    site_names = sorted(all_site_data["name"].unique())

    with PdfPages(atlas_path) as pdf:
        for site_name in site_names:
            site_row = all_site_data[all_site_data["name"] == site_name].iloc[0]
            _write_site_pages(
                pdf, site_name, site_row.get("type", "site"),
                all_site_data, yearly_df,
                buffer_comparison_df, directional_df, proximity_df,
            )

    log.info("Generated site atlas (%d sites): %s", len(site_names), atlas_path)


def _write_site_pages(pdf, site_name, site_type, metrics_df, yearly_df,
                      buffer_comparison_df=None, directional_df=None,
                      proximity_df=None):
    """Write pages for a single site into the shared PdfPages object."""
    site_metrics = metrics_df[metrics_df["name"] == site_name] if "name" in metrics_df.columns else pd.DataFrame()
    site_yearly = yearly_df[yearly_df["name"] == site_name].sort_values("year") if "name" in yearly_df.columns else pd.DataFrame()

    # Page 1: Summary
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    lines = [
        f"Site Report: {site_name}",
        f"Type: {site_type}",
        "=" * 50,
        "",
    ]
    if not site_metrics.empty:
        r = site_metrics.iloc[-1]
        lines += [
            f"District: {r.get('district', 'N/A')}",
            f"Median Radiance: {r.get('median_radiance', np.nan):.3f} nW/cm\u00b2/sr",
            f"Mean Radiance: {r.get('mean_radiance', np.nan):.3f} nW/cm\u00b2/sr",
            f"ALAN Class: {r.get('alan_class', 'N/A')}",
            f"Valid Pixels: {r.get('valid_pixels', 'N/A')}",
            f"Quality: {r.get('quality_pct', 'N/A')}%",
            "",
        ]

    if proximity_df is not None and not proximity_df.empty:
        prox = proximity_df[proximity_df["site"] == site_name]
        if not prox.empty:
            p = prox.iloc[0]
            lines += [
                "Nearest City:",
                f"  {p.get('nearest_city_name', 'N/A')} ({p.get('distance_km', 'N/A')} km, {p.get('cardinal_direction', '')})",
                "",
            ]

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # Page 2: Time series
    if len(site_yearly) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(site_yearly["year"], site_yearly["median_radiance"], "o-",
                color="steelblue", markersize=5, linewidth=2)
        if site_type == "site":
            ax.axhline(y=config.ALAN_LOW_THRESHOLD, color="orange",
                       linestyle="--", linewidth=1,
                       label=f"Low-ALAN threshold ({config.ALAN_LOW_THRESHOLD} nW)")
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Median Radiance (nW/cm\u00b2/sr)", fontsize=12)
        ax.set_title(f"{site_name}: ALAN Time Series", fontsize=14)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # Page 3: Buffer comparison (if available)
    if buffer_comparison_df is not None and not buffer_comparison_df.empty:
        buf = buffer_comparison_df[buffer_comparison_df["site"] == site_name]
        if not buf.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            b = buf.iloc[0]
            vals = [b.get("inside_median", 0) or 0, b.get("outside_median", 0) or 0]
            ax.bar(["Inside Buffer", "Outside Buffer"], vals,
                   color=["forestgreen", "orange"], alpha=0.7)
            ax.axhline(y=config.ALAN_LOW_THRESHOLD, color="red",
                       linestyle="--", linewidth=1)
            ax.set_ylabel("Median Radiance (nW/cm\u00b2/sr)", fontsize=12)
            ax.set_title(f"{site_name}: Inside vs Outside Buffer", fontsize=14)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    # Page 4 (dark-sky sites only): Recommendations
    if site_type == "site":
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        rec_lines = [
            f"Dark-Sky Suitability Assessment: {site_name}",
            "=" * 50,
            "",
        ]
        if not site_metrics.empty:
            r = site_metrics.iloc[-1]
            med = r.get("median_radiance", np.nan)
            if not np.isnan(med) and med < config.ALAN_LOW_THRESHOLD:
                rec_lines.append("SUITABLE: Median radiance below dark-sky threshold")
                rec_lines.append(f"  Current: {med:.3f} nW/cm\u00b2/sr (threshold: {config.ALAN_LOW_THRESHOLD})")
            else:
                rec_lines.append("NOT SUITABLE: Median radiance above dark-sky threshold")
                rec_lines.append(f"  Current: {med:.3f} nW/cm\u00b2/sr")
            rec_lines.append("")

        if directional_df is not None and not directional_df.empty:
            d = directional_df[directional_df["site"] == site_name]
            if not d.empty:
                dd = d.iloc[0]
                rec_lines.append(f"Dominant light direction: {dd.get('dominant_direction', 'N/A')}")
                rec_lines.append(f"Directional ratio: {dd.get('max_min_ratio', 'N/A')}")
                rec_lines.append("")
                rec_lines.append("Mitigation: Consider shielding/zoning in the "
                                 f"{dd.get('dominant_direction', '')} direction")

        ax.text(0.05, 0.95, "\n".join(rec_lines), transform=ax.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
