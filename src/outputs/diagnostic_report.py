"""
Per-run HTML diagnostic report generator.

Generates a self-contained HTML report summarising pipeline run outcomes,
data quality metrics, trend overview, and anomaly flags. No external JS
dependencies — all plots are embedded as base64 PNGs.

Usage:
    from src.outputs.diagnostic_report import generate_diagnostic_report
    path = generate_diagnostic_report(pipeline_result, yearly_df, trends_df, output_dir)
"""

import base64
import io
import os
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.logging_config import get_pipeline_logger

log = get_pipeline_logger(__name__)


def _fig_to_base64(fig):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _render_step_table(step_results):
    """Render a step results summary as an HTML table."""
    rows = []
    for sr in step_results:
        d = sr if isinstance(sr, dict) else sr.to_dict()
        status = d.get("status", "unknown")
        color = {"success": "#2ecc71", "error": "#e74c3c", "skipped": "#f39c12"}.get(
            status, "#95a5a6"
        )
        timing = d.get("timing_seconds", 0)
        rows.append(
            f"<tr>"
            f'<td>{d.get("step_name", "")}</td>'
            f'<td style="color:{color};font-weight:bold">{status}</td>'
            f"<td>{timing:.1f}s</td>"
            f'<td>{d.get("error", "") or ""}</td>'
            f"</tr>"
        )
    return "\n".join(rows)


def _generate_trend_overview_plot(trends_df):
    """Generate a horizontal bar chart of annual_pct_change by district."""
    if trends_df is None or "annual_pct_change" not in trends_df.columns:
        return None
    if "district" not in trends_df.columns:
        return None

    df = trends_df.dropna(subset=["annual_pct_change"]).sort_values("annual_pct_change")
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.3)))
    colors = ["#2ecc71" if v < 0 else "#e74c3c" for v in df["annual_pct_change"]]
    ax.barh(df["district"], df["annual_pct_change"], color=colors)
    ax.set_xlabel("Annual % Change")
    ax.set_title("District Light Pollution Trends")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)
    return _fig_to_base64(fig)


def _generate_data_quality_plot(yearly_df):
    """Generate a heatmap of NaN counts by district × year."""
    if yearly_df is None:
        return None

    cols_to_check = [c for c in yearly_df.columns if c not in ("district", "year")]
    if not cols_to_check:
        return None

    nan_summary = yearly_df.groupby("district")[cols_to_check].apply(
        lambda g: g.isna().sum()
    )
    if nan_summary.sum().sum() == 0:
        return None  # No NaNs to show

    fig, ax = plt.subplots(figsize=(10, max(4, len(nan_summary) * 0.3)))
    im = ax.imshow(nan_summary.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(nan_summary.columns)))
    ax.set_xticklabels(nan_summary.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(nan_summary.index)))
    ax.set_yticklabels(nan_summary.index, fontsize=8)
    ax.set_title("NaN Count by District × Column")
    fig.colorbar(im, ax=ax, label="NaN count")
    return _fig_to_base64(fig)


def _identify_anomalies(trends_df):
    """Identify districts with unusual patterns."""
    if trends_df is None:
        return []

    flags = []
    for _, row in trends_df.iterrows():
        district = row.get("district", "?")
        reasons = []

        r2 = row.get("r_squared")
        if r2 is not None and not np.isnan(r2) and r2 < 0.3:
            reasons.append(f"Low R² ({r2:.2f})")

        pct = row.get("annual_pct_change")
        if pct is not None and not np.isnan(pct) and abs(pct) > 20:
            reasons.append(f"Extreme trend ({pct:+.1f}%/yr)")

        if reasons:
            flags.append({"district": district, "reasons": "; ".join(reasons)})

    return flags


def generate_diagnostic_report(pipeline_result, yearly_df=None,
                                trends_df=None, output_dir=None):
    """Generate an HTML diagnostic report for a pipeline run.

    Parameters
    ----------
    pipeline_result : PipelineRunResult
        The completed pipeline result.
    yearly_df : pd.DataFrame, optional
        Yearly radiance data for quality analysis.
    trends_df : pd.DataFrame, optional
        Trend results for overview and anomaly detection.
    output_dir : str, optional
        Where to save the report. Defaults to pipeline_result.run_dir.

    Returns
    -------
    str or None
        Path to the generated HTML file, or None on failure.
    """
    if output_dir is None:
        output_dir = pipeline_result.run_dir

    if not output_dir:
        return None

    diagnostics_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)

    # Gather data
    result_dict = (
        pipeline_result if isinstance(pipeline_result, dict)
        else pipeline_result.to_dict()
    )

    steps = result_dict.get("steps", [])
    total_time = result_dict.get("total_time_seconds", 0)
    git_sha = result_dict.get("git_sha", "unknown")
    started_at = result_dict.get("started_at", "")
    entity_type = result_dict.get("entity_type", "unknown")

    n_success = sum(1 for s in steps if s.get("status") == "success")
    n_error = sum(1 for s in steps if s.get("status") == "error")
    n_skipped = sum(1 for s in steps if s.get("status") == "skipped")

    # Generate plots
    trend_plot = _generate_trend_overview_plot(trends_df)
    quality_plot = _generate_data_quality_plot(yearly_df)

    # Anomaly detection
    anomalies = _identify_anomalies(trends_df)
    anomaly_rows = ""
    for a in anomalies:
        anomaly_rows += (
            f'<tr><td>{a["district"]}</td><td>{a["reasons"]}</td></tr>'
        )

    # Step table
    step_table_rows = _render_step_table(steps)

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Pipeline Diagnostic Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
         max-width: 1100px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
  h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #34495e; margin-top: 30px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                   gap: 15px; margin: 20px 0; }}
  .summary-card {{ background: white; border-radius: 8px; padding: 15px;
                   box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .summary-card .label {{ font-size: 0.85em; color: #7f8c8d; }}
  .summary-card .value {{ font-size: 1.4em; font-weight: bold; color: #2c3e50; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white;
           border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  th {{ background: #34495e; color: white; padding: 10px 12px; text-align: left; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #ecf0f1; }}
  tr:hover {{ background: #f8f9fa; }}
  .plot {{ text-align: center; margin: 20px 0; }}
  .plot img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .anomaly {{ background: #ffeaa7; padding: 10px; border-radius: 6px; margin: 5px 0; }}
  .footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #ddd;
             font-size: 0.85em; color: #95a5a6; }}
</style>
</head>
<body>
<h1>Pipeline Diagnostic Report</h1>

<div class="summary-grid">
  <div class="summary-card">
    <div class="label">Entity Type</div>
    <div class="value">{entity_type}</div>
  </div>
  <div class="summary-card">
    <div class="label">Total Time</div>
    <div class="value">{total_time:.1f}s</div>
  </div>
  <div class="summary-card">
    <div class="label">Steps Passed</div>
    <div class="value" style="color:#2ecc71">{n_success}</div>
  </div>
  <div class="summary-card">
    <div class="label">Steps Failed</div>
    <div class="value" style="color:#e74c3c">{n_error}</div>
  </div>
  <div class="summary-card">
    <div class="label">Git SHA</div>
    <div class="value" style="font-size:1em">{git_sha or 'N/A'}</div>
  </div>
  <div class="summary-card">
    <div class="label">Started At</div>
    <div class="value" style="font-size:0.9em">{started_at}</div>
  </div>
</div>

<h2>Step Results</h2>
<table>
<thead><tr><th>Step</th><th>Status</th><th>Time</th><th>Error</th></tr></thead>
<tbody>
{step_table_rows}
</tbody>
</table>
"""

    if trend_plot:
        html += f"""
<h2>Trend Overview</h2>
<div class="plot"><img src="data:image/png;base64,{trend_plot}" alt="Trend Overview"></div>
"""

    if quality_plot:
        html += f"""
<h2>Data Quality — NaN Distribution</h2>
<div class="plot"><img src="data:image/png;base64,{quality_plot}" alt="Data Quality"></div>
"""

    if anomalies:
        html += f"""
<h2>Anomaly Flags ({len(anomalies)} districts)</h2>
<table>
<thead><tr><th>District</th><th>Reason</th></tr></thead>
<tbody>
{anomaly_rows}
</tbody>
</table>
"""
    else:
        html += "<h2>Anomaly Flags</h2><p>No anomalies detected.</p>"

    html += f"""
<div class="footer">
  Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
  | maharashtra-darksky-research diagnostic report
</div>
</body>
</html>"""

    report_path = os.path.join(diagnostics_dir, "run_report.html")
    with open(report_path, "w") as f:
        f.write(html)

    return report_path
