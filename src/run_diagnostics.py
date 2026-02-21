"""
Pipeline run comparison and diff tool.

Compares two pipeline runs to identify changes in step outcomes,
timing regressions, data value shifts, and classification changes.

Usage (CLI):
    python -m src.run_diagnostics --run-a outputs/run_20240115 --run-b outputs/run_20240116

Usage (library):
    from src.run_diagnostics import compare_runs
    report = compare_runs("outputs/run_a", "outputs/run_b")
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.logging_config import get_pipeline_logger
from src.pipeline_types import PipelineRunResult

log = get_pipeline_logger(__name__)


def _load_pipeline_result(run_dir):
    """Load PipelineRunResult from a run directory."""
    result_path = os.path.join(run_dir, "pipeline_run.json")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Pipeline result not found: {result_path}")
    with open(result_path) as f:
        data = json.load(f)
    return PipelineRunResult.from_dict(data)


def compare_runs(run_dir_a, run_dir_b):
    """Compare two pipeline runs and return a structured diff.

    Parameters
    ----------
    run_dir_a : str
        Path to the first (baseline) run directory.
    run_dir_b : str
        Path to the second (comparison) run directory.

    Returns
    -------
    dict
        Comparison report with keys: step_diffs, timing_diffs, csv_diffs.
    """
    result_a = _load_pipeline_result(run_dir_a)
    result_b = _load_pipeline_result(run_dir_b)

    report = {
        "run_a": run_dir_a,
        "run_b": run_dir_b,
        "git_sha_a": result_a.git_sha,
        "git_sha_b": result_b.git_sha,
        "step_diffs": [],
        "timing_diffs": [],
        "csv_diffs": [],
        "summary": {},
    }

    # Compare step outcomes
    steps_a = {s.step_name: s for s in result_a.step_results}
    steps_b = {s.step_name: s for s in result_b.step_results}
    all_steps = sorted(set(steps_a.keys()) | set(steps_b.keys()))

    for name in all_steps:
        sa = steps_a.get(name)
        sb = steps_b.get(name)

        diff = {"step_name": name}
        if sa is None:
            diff["change"] = "added"
            diff["status_b"] = sb.status
        elif sb is None:
            diff["change"] = "removed"
            diff["status_a"] = sa.status
        elif sa.status != sb.status:
            diff["change"] = "status_changed"
            diff["status_a"] = sa.status
            diff["status_b"] = sb.status
        else:
            diff["change"] = "unchanged"
            diff["status"] = sa.status

        # Timing comparison
        if sa and sb:
            t_a = sa.timing_seconds or 0
            t_b = sb.timing_seconds or 0
            diff["timing_a"] = t_a
            diff["timing_b"] = t_b
            if t_a > 0:
                diff["timing_pct_change"] = ((t_b - t_a) / t_a) * 100
            report["timing_diffs"].append({
                "step_name": name,
                "time_a": t_a,
                "time_b": t_b,
            })

        report["step_diffs"].append(diff)

    # Compare CSV outputs
    csv_diffs = compare_csv_outputs(run_dir_a, run_dir_b)
    report["csv_diffs"] = csv_diffs

    # Summary
    n_changed = sum(1 for d in report["step_diffs"] if d["change"] != "unchanged")
    total_a = result_a.total_time_seconds
    total_b = result_b.total_time_seconds

    report["summary"] = {
        "steps_changed": n_changed,
        "total_steps": len(all_steps),
        "total_time_a": total_a,
        "total_time_b": total_b,
        "time_delta_pct": ((total_b - total_a) / total_a * 100) if total_a > 0 else 0,
        "csv_files_diffed": len(csv_diffs),
    }

    return report


def compare_csv_outputs(run_dir_a, run_dir_b, entity_type="district"):
    """Compare CSV output files between two runs.

    Finds matching CSV files in both runs and computes value diffs.

    Parameters
    ----------
    run_dir_a, run_dir_b : str
    entity_type : str

    Returns
    -------
    list[dict]
        Per-file diff summaries.
    """
    csv_dir_a = os.path.join(run_dir_a, entity_type, "csv")
    csv_dir_b = os.path.join(run_dir_b, entity_type, "csv")

    if not os.path.isdir(csv_dir_a) or not os.path.isdir(csv_dir_b):
        return []

    files_a = {f for f in os.listdir(csv_dir_a) if f.endswith(".csv")}
    files_b = {f for f in os.listdir(csv_dir_b) if f.endswith(".csv")}
    common = sorted(files_a & files_b)

    diffs = []
    for filename in common:
        try:
            df_a = pd.read_csv(os.path.join(csv_dir_a, filename))
            df_b = pd.read_csv(os.path.join(csv_dir_b, filename))

            diff = compare_dataframes(df_a, df_b, filename)
            diffs.append(diff)
        except Exception as exc:
            diffs.append({"filename": filename, "error": str(exc)})

    # Report files only in one run
    for f in sorted(files_a - files_b):
        diffs.append({"filename": f, "change": "removed_in_b"})
    for f in sorted(files_b - files_a):
        diffs.append({"filename": f, "change": "added_in_b"})

    return diffs


def compare_dataframes(df_a, df_b, name=""):
    """Compare two DataFrames and return a summary of differences.

    Parameters
    ----------
    df_a, df_b : pd.DataFrame
    name : str

    Returns
    -------
    dict
    """
    result = {
        "filename": name,
        "rows_a": len(df_a),
        "rows_b": len(df_b),
        "cols_a": list(df_a.columns),
        "cols_b": list(df_b.columns),
        "column_changes": [],
        "value_changes": {},
    }

    # Column-level changes
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    result["columns_added"] = sorted(cols_b - cols_a)
    result["columns_removed"] = sorted(cols_a - cols_b)

    # Value-level changes for numeric columns
    common_cols = sorted(cols_a & cols_b)
    for col in common_cols:
        if not pd.api.types.is_numeric_dtype(df_a[col]):
            continue

        # Compare mean, median, std
        stats_a = {
            "mean": float(df_a[col].mean()) if not df_a[col].isna().all() else None,
            "median": float(df_a[col].median()) if not df_a[col].isna().all() else None,
            "nan_count": int(df_a[col].isna().sum()),
        }
        stats_b = {
            "mean": float(df_b[col].mean()) if not df_b[col].isna().all() else None,
            "median": float(df_b[col].median()) if not df_b[col].isna().all() else None,
            "nan_count": int(df_b[col].isna().sum()),
        }

        if stats_a["mean"] is not None and stats_b["mean"] is not None:
            if stats_a["mean"] != 0:
                pct_change = ((stats_b["mean"] - stats_a["mean"]) / abs(stats_a["mean"])) * 100
            else:
                pct_change = 0.0 if stats_b["mean"] == 0 else float("inf")

            result["value_changes"][col] = {
                "mean_a": stats_a["mean"],
                "mean_b": stats_b["mean"],
                "mean_pct_change": pct_change,
                "nan_a": stats_a["nan_count"],
                "nan_b": stats_b["nan_count"],
            }

    return result


def generate_comparison_report(run_dir_a, run_dir_b, output_path=None):
    """Generate a Markdown comparison report between two runs.

    Parameters
    ----------
    run_dir_a, run_dir_b : str
    output_path : str, optional
        If None, prints to stdout.

    Returns
    -------
    str
        Markdown report text.
    """
    report = compare_runs(run_dir_a, run_dir_b)
    lines = [
        "# Pipeline Run Comparison",
        "",
        f"**Run A:** `{report['run_a']}` (git: {report['git_sha_a']})",
        f"**Run B:** `{report['run_b']}` (git: {report['git_sha_b']})",
        "",
        f"## Summary",
        f"- Steps changed: {report['summary']['steps_changed']} / {report['summary']['total_steps']}",
        f"- Total time: {report['summary']['total_time_a']:.1f}s → "
        f"{report['summary']['total_time_b']:.1f}s "
        f"({report['summary']['time_delta_pct']:+.1f}%)",
        f"- CSV files compared: {report['summary']['csv_files_diffed']}",
        "",
        "## Step Results",
        "",
        "| Step | Change | A | B | Time A | Time B |",
        "|------|--------|---|---|--------|--------|",
    ]

    for d in report["step_diffs"]:
        status_a = d.get("status_a", d.get("status", ""))
        status_b = d.get("status_b", d.get("status", ""))
        time_a = f'{d.get("timing_a", 0):.1f}s' if "timing_a" in d else "—"
        time_b = f'{d.get("timing_b", 0):.1f}s' if "timing_b" in d else "—"
        lines.append(
            f"| {d['step_name']} | {d['change']} | {status_a} | {status_b} | {time_a} | {time_b} |"
        )

    if report["csv_diffs"]:
        lines.extend(["", "## CSV Output Diffs", ""])
        for csv_diff in report["csv_diffs"]:
            if "error" in csv_diff:
                lines.append(f"- **{csv_diff['filename']}**: error — {csv_diff['error']}")
                continue
            if csv_diff.get("change"):
                lines.append(f"- **{csv_diff['filename']}**: {csv_diff['change']}")
                continue

            lines.append(f"### {csv_diff['filename']}")
            lines.append(f"Rows: {csv_diff['rows_a']} → {csv_diff['rows_b']}")
            if csv_diff.get("columns_added"):
                lines.append(f"Columns added: {csv_diff['columns_added']}")
            if csv_diff.get("columns_removed"):
                lines.append(f"Columns removed: {csv_diff['columns_removed']}")

            for col, vc in csv_diff.get("value_changes", {}).items():
                lines.append(
                    f"- `{col}`: mean {vc['mean_a']:.4f} → {vc['mean_b']:.4f} "
                    f"({vc['mean_pct_change']:+.2f}%)"
                )
            lines.append("")

    md_text = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(md_text)
        log.info("Comparison report saved: %s", output_path)

    return md_text


# ── CLI entry point ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compare two pipeline runs for regressions and changes"
    )
    parser.add_argument(
        "--run-a", required=True, help="Path to baseline run directory"
    )
    parser.add_argument(
        "--run-b", required=True, help="Path to comparison run directory"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for Markdown report (default: print to stdout)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON diff instead of Markdown",
    )
    args = parser.parse_args()

    if args.json:
        report = compare_runs(args.run_a, args.run_b)
        output = json.dumps(report, indent=2, default=str)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
        else:
            print(output)
    else:
        md = generate_comparison_report(args.run_a, args.run_b, args.output)
        if not args.output:
            print(md)


if __name__ == "__main__":
    main()
