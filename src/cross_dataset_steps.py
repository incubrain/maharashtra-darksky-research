"""
Cross-dataset pipeline steps (16-20).

Five non-critical steps that load external datasets, merge with VNL data,
compute correlations, classify districts into quadrants, and generate
cross-dataset reports.  Boilerplate is handled by ``run_step()``.
"""

import os

import numpy as np
import pandas as pd

from src import config
from src.dataset_aggregator import (
    load_all_datasets,
    merge_trends_with_datasets,
    merge_yearly_with_datasets,
)
from src.logging_config import get_pipeline_logger
from src.step_runner import run_step

log = get_pipeline_logger(__name__)


# ── Step 16: Load datasets ─────────────────────────────────────────────


def step_load_datasets(
    enabled_datasets: list[str],
    args,
    csv_dir: str,
    vnl_district_names: list[str] | None = None,
) -> tuple:
    """Load all enabled external datasets, save CSV checkpoints."""

    def _work():
        all_results = load_all_datasets(
            enabled_datasets, args, vnl_district_names=vnl_district_names
        )
        loaded = {}
        failed = []
        for name, (result, df) in all_results.items():
            if result.ok and df is not None:
                out_path = os.path.join(csv_dir, f"dataset_{name}.csv")
                df.to_csv(out_path, index=False)
                log.info("Saved dataset checkpoint: %s", out_path)
                loaded[name] = df
            else:
                failed.append(name)

        if not loaded:
            raise ValueError(f"No datasets loaded successfully. Failed: {failed}")
        return loaded

    return run_step(
        "load_datasets", _work,
        input_summary={"datasets": enabled_datasets},
        output_summary_fn=lambda d: {"loaded": list(d.keys()), "total_datasets": len(d)},
    )


# ── Step 17: Merge datasets ────────────────────────────────────────────


def step_merge_datasets(
    yearly_df: pd.DataFrame,
    trends_df: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    csv_dir: str,
    suffix: str,
) -> tuple:
    """Merge VNL data with all loaded datasets."""

    def _work():
        merged_yearly = merge_yearly_with_datasets(yearly_df, datasets)
        merged_trends = merge_trends_with_datasets(trends_df, datasets)

        yearly_path = os.path.join(csv_dir, f"merged_yearly{suffix}.csv")
        trends_path = os.path.join(csv_dir, f"merged_trends{suffix}.csv")
        merged_yearly.to_csv(yearly_path, index=False)
        merged_trends.to_csv(trends_path, index=False)
        log.info("Saved merged yearly: %s (%d rows)", yearly_path, len(merged_yearly))
        log.info("Saved merged trends: %s (%d rows)", trends_path, len(merged_trends))

        return {"yearly": merged_yearly, "trends": merged_trends}

    return run_step(
        "merge_datasets", _work,
        output_summary_fn=lambda m: {
            "yearly_rows": len(m["yearly"]),
            "trends_rows": len(m["trends"]),
        },
    )


# ── Step 18: Cross-correlation ──────────────────────────────────────────


def step_cross_correlation(
    merged_trends_df: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    csv_dir: str,
    maps_dir: str,
    suffix: str,
) -> tuple:
    """Compute correlation matrix between VNL metrics and dataset metrics."""
    from src.formulas.correlation import (
        compute_correlation_matrix,
        partial_correlation,
    )

    def _work():
        vnl_cols = [
            c for c in config.VNL_CORRELATION_METRICS if c in merged_trends_df.columns
        ]
        dataset_cols = []
        for name, df in datasets.items():
            from src.datasets import DATASET_REGISTRY
            module = DATASET_REGISTRY.get(name)
            if module:
                prefix = module.get_meta().short_label + "_"
                dataset_cols.extend(
                    c for c in merged_trends_df.columns
                    if c.startswith(prefix) and merged_trends_df[c].notna().sum() >= 3
                )

        if not vnl_cols or not dataset_cols:
            raise ValueError(
                f"Insufficient columns for correlation: vnl={vnl_cols}, dataset={dataset_cols}"
            )

        corr_df = compute_correlation_matrix(
            merged_trends_df, vnl_cols, dataset_cols, method="both"
        )

        pop_col = _find_population_column(merged_trends_df, datasets)
        if pop_col and pop_col in merged_trends_df.columns:
            partial_rs = []
            partial_ps = []
            for _, row in corr_df.iterrows():
                if row["y_col"] == pop_col:
                    partial_rs.append(np.nan)
                    partial_ps.append(np.nan)
                else:
                    pc = partial_correlation(
                        merged_trends_df[row["x_col"]],
                        merged_trends_df[row["y_col"]],
                        merged_trends_df[[pop_col]],
                    )
                    partial_rs.append(pc["r"])
                    partial_ps.append(pc["p_value"])
            corr_df["partial_r"] = partial_rs
            corr_df["partial_p"] = partial_ps

        out_path = os.path.join(csv_dir, f"cross_correlation{suffix}.csv")
        corr_df.to_csv(out_path, index=False)
        log.info("Saved cross-correlation: %s (%d pairs)", out_path, len(corr_df))

        _plot_correlation_heatmap(corr_df, vnl_cols, dataset_cols, maps_dir, suffix)
        _plot_scatter_grid(merged_trends_df, corr_df, maps_dir, suffix)

        return corr_df

    return run_step(
        "cross_correlation", _work,
        output_summary_fn=lambda df: {"correlation_pairs": len(df)},
    )


# ── Step 19: Cross-classification ───────────────────────────────────────


def step_cross_classification(
    merged_trends_df: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    csv_dir: str,
    maps_dir: str,
    suffix: str,
) -> tuple:
    """Classify districts into quadrants based on VNL x dataset metrics."""

    def _work():
        metric_pairs = _get_quadrant_metric_pairs(merged_trends_df, datasets)
        if not metric_pairs:
            raise ValueError("No valid metric pairs for quadrant classification")

        all_rows = []
        for vnl_metric, ds_metric, pair_label in metric_pairs:
            rows = _classify_quadrant(
                merged_trends_df, vnl_metric, ds_metric, pair_label
            )
            all_rows.extend(rows)
            _plot_quadrant(
                merged_trends_df, vnl_metric, ds_metric, pair_label,
                maps_dir, suffix,
            )

        class_df = pd.DataFrame(all_rows)
        out_path = os.path.join(csv_dir, f"cross_classification{suffix}.csv")
        class_df.to_csv(out_path, index=False)
        log.info("Saved cross-classification: %s (%d records)", out_path, len(class_df))
        return class_df

    return run_step(
        "cross_classification", _work,
        output_summary_fn=lambda df: {"records": len(df)},
    )


# ── Step 20: Cross-dataset reports ──────────────────────────────────────


def step_cross_dataset_reports(
    merged_trends_df: pd.DataFrame,
    corr_df: pd.DataFrame | None,
    class_df: pd.DataFrame | None,
    datasets: dict[str, pd.DataFrame],
    reports_dir: str,
    maps_dir: str,
    suffix: str,
) -> tuple:
    """Generate summary report of cross-dataset findings."""

    return run_step(
        "cross_dataset_reports",
        _generate_cross_dataset_summary,
        merged_trends_df, corr_df, class_df, datasets,
        reports_dir, maps_dir, suffix,
    )


# ── Helper functions ────────────────────────────────────────────────────


def _find_population_column(df: pd.DataFrame, datasets: dict) -> str | None:
    """Find the population column for partial correlation control."""
    for col in df.columns:
        if "population" in col.lower() or col.endswith("_TOT_P"):
            return col
    return None


def _get_quadrant_metric_pairs(df, datasets):
    """Determine which VNL x dataset metric pairs to use for quadrants."""
    pairs = []
    vnl_candidates = ["median_radiance", "annual_pct_change"]
    ds_candidates = []
    for name in datasets:
        from src.datasets import DATASET_REGISTRY
        module = DATASET_REGISTRY.get(name)
        if module:
            prefix = module.get_meta().short_label + "_"
            for metric in ["literacy_rate", "urbanization_proxy", "workforce_rate"]:
                col = prefix + metric
                if col in df.columns:
                    ds_candidates.append((col, metric))

    for vnl_col in vnl_candidates:
        if vnl_col not in df.columns:
            continue
        for ds_col, metric_name in ds_candidates:
            label = f"{vnl_col}_vs_{metric_name}"
            pairs.append((vnl_col, ds_col, label))
    return pairs


def _classify_quadrant(df, vnl_col, ds_col, pair_label):
    """Classify districts into HH/HL/LH/LL quadrants."""
    valid = df[[vnl_col, ds_col, "district"]].dropna()
    if valid.empty:
        return []

    vnl_median = valid[vnl_col].median()
    ds_median = valid[ds_col].median()

    rows = []
    for _, row in valid.iterrows():
        v_high = row[vnl_col] >= vnl_median
        d_high = row[ds_col] >= ds_median
        if v_high and d_high:
            quadrant = "HH"
        elif v_high and not d_high:
            quadrant = "HL"
        elif not v_high and d_high:
            quadrant = "LH"
        else:
            quadrant = "LL"

        rows.append({
            "district": row["district"],
            "vnl_metric": vnl_col,
            "dataset_metric": ds_col,
            "vnl_value": row[vnl_col],
            "dataset_value": row[ds_col],
            "quadrant": quadrant,
            "pair_label": pair_label,
        })
    return rows


def _plot_correlation_heatmap(corr_df, vnl_cols, dataset_cols, maps_dir, suffix):
    """Plot correlation matrix heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pivot = corr_df.pivot(index="x_col", columns="y_col", values="pearson_r")

        fig, ax = plt.subplots(figsize=(max(10, len(dataset_cols) * 1.2), max(6, len(vnl_cols) * 1.5)))
        im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(
            [c.split("_", 1)[-1] if "_" in c else c for c in pivot.columns],
            rotation=45, ha="right", fontsize=8,
        )
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)

        p_pivot = corr_df.pivot(index="x_col", columns="y_col", values="pearson_p")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                r_val = pivot.values[i, j]
                if np.isnan(r_val):
                    continue
                p_val = p_pivot.values[i, j]
                stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                color = "white" if abs(r_val) > 0.5 else "black"
                ax.text(j, i, f"{r_val:.2f}{stars}", ha="center", va="center",
                        fontsize=7, color=color)

        plt.colorbar(im, ax=ax, label="Pearson r")
        ax.set_title("Cross-Dataset Correlation Matrix")
        plt.tight_layout()

        out_path = os.path.join(maps_dir, f"correlation_matrix{suffix}.png")
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        log.info("Saved correlation heatmap: %s", out_path)

    except Exception as exc:
        log.warning("Failed to plot correlation heatmap: %s", exc)


def _plot_scatter_grid(merged_df, corr_df, maps_dir, suffix):
    """Plot scatter grid of top 6 strongest correlations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats as sp_stats

        top = corr_df.dropna(subset=["pearson_r"]).copy()
        top["abs_r"] = top["pearson_r"].abs()
        top = top.nlargest(6, "abs_r")

        if top.empty:
            return

        n_plots = len(top)
        ncols = min(3, n_plots)
        nrows = (n_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (_, row) in enumerate(top.iterrows()):
            ax = axes[idx]
            x_col, y_col = row["x_col"], row["y_col"]
            valid = merged_df[[x_col, y_col]].dropna()

            ax.scatter(valid[x_col], valid[y_col], alpha=0.6, s=30)

            if len(valid) >= 3:
                slope, intercept, r, p, se = sp_stats.linregress(valid[x_col], valid[y_col])
                x_range = np.linspace(valid[x_col].min(), valid[x_col].max(), 50)
                ax.plot(x_range, slope * x_range + intercept, "r-", alpha=0.7)
                ax.set_title(f"r={row['pearson_r']:.3f} p={row['pearson_p']:.3g}", fontsize=9)

            x_label = x_col.split("_", 1)[-1] if "_" in x_col else x_col
            y_label = y_col.split("_", 1)[-1] if "_" in y_col else y_col
            ax.set_xlabel(x_label, fontsize=8)
            ax.set_ylabel(y_label, fontsize=8)

        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Top Correlations: VNL vs External Datasets", fontsize=11)
        plt.tight_layout()

        out_path = os.path.join(maps_dir, f"scatter_grid{suffix}.png")
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        log.info("Saved scatter grid: %s", out_path)

    except Exception as exc:
        log.warning("Failed to plot scatter grid: %s", exc)


def _plot_quadrant(df, vnl_col, ds_col, pair_label, maps_dir, suffix):
    """Plot quadrant scatter for a single metric pair."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid = df[["district", vnl_col, ds_col]].dropna()
        if valid.empty:
            return

        vnl_med = valid[vnl_col].median()
        ds_med = valid[ds_col].median()

        fig, ax = plt.subplots(figsize=(8, 6))

        colors = []
        for _, row in valid.iterrows():
            v_high = row[vnl_col] >= vnl_med
            d_high = row[ds_col] >= ds_med
            if v_high and d_high:
                colors.append("#d73027")
            elif v_high and not d_high:
                colors.append("#fc8d59")
            elif not v_high and d_high:
                colors.append("#91bfdb")
            else:
                colors.append("#1a9850")

        ax.scatter(valid[vnl_col], valid[ds_col], c=colors, s=40, alpha=0.7, edgecolors="k", linewidth=0.5)

        ax.axvline(vnl_med, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(ds_med, color="gray", linestyle="--", alpha=0.5)

        for _, row in valid.iterrows():
            ax.annotate(
                row["district"], (row[vnl_col], row[ds_col]),
                fontsize=6, alpha=0.7,
                xytext=(3, 3), textcoords="offset points",
            )

        x_range = ax.get_xlim()
        y_range = ax.get_ylim()
        ax.text(x_range[1] * 0.95, y_range[1] * 0.95, "HH", fontsize=12, ha="right", va="top", color="#d73027", weight="bold")
        ax.text(x_range[1] * 0.95, y_range[0] + (y_range[1] - y_range[0]) * 0.05, "HL", fontsize=12, ha="right", va="bottom", color="#fc8d59", weight="bold")
        ax.text(x_range[0] + (x_range[1] - x_range[0]) * 0.05, y_range[1] * 0.95, "LH", fontsize=12, ha="left", va="top", color="#91bfdb", weight="bold")
        ax.text(x_range[0] + (x_range[1] - x_range[0]) * 0.05, y_range[0] + (y_range[1] - y_range[0]) * 0.05, "LL", fontsize=12, ha="left", va="bottom", color="#1a9850", weight="bold")

        ax.set_xlabel(vnl_col)
        ax.set_ylabel(ds_col.split("_", 1)[-1] if "_" in ds_col else ds_col)
        ax.set_title(f"Quadrant Classification: {pair_label}")
        plt.tight_layout()

        out_path = os.path.join(maps_dir, f"quadrant_{pair_label}{suffix}.png")
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        log.info("Saved quadrant plot: %s", out_path)

    except Exception as exc:
        log.warning("Failed to plot quadrant %s: %s", pair_label, exc)


def _generate_cross_dataset_summary(
    merged_trends_df, corr_df, class_df, datasets,
    reports_dir, maps_dir, suffix,
):
    """Generate a summary CSV report of cross-dataset findings."""
    os.makedirs(reports_dir, exist_ok=True)

    if corr_df is not None and not corr_df.empty:
        summary_rows = []
        top_corr = corr_df.dropna(subset=["pearson_r"]).copy()
        top_corr["abs_r"] = top_corr["pearson_r"].abs()
        top_corr = top_corr.nlargest(10, "abs_r")

        for _, row in top_corr.iterrows():
            summary_rows.append({
                "type": "correlation",
                "vnl_metric": row["x_col"],
                "dataset_metric": row["y_col"],
                "pearson_r": row["pearson_r"],
                "pearson_p": row["pearson_p"],
                "spearman_r": row.get("spearman_r", np.nan),
                "partial_r": row.get("partial_r", np.nan),
                "n": row.get("n", 0),
            })

        summary_df = pd.DataFrame(summary_rows)
        out_path = os.path.join(reports_dir, f"cross_dataset_summary{suffix}.csv")
        summary_df.to_csv(out_path, index=False)
        log.info("Saved cross-dataset summary: %s", out_path)

    if class_df is not None and not class_df.empty:
        quad_summary = class_df.groupby(["pair_label", "quadrant"]).size().reset_index(name="count")
        out_path = os.path.join(reports_dir, f"quadrant_distribution{suffix}.csv")
        quad_summary.to_csv(out_path, index=False)
        log.info("Saved quadrant distribution: %s", out_path)

    if merged_trends_df is not None:
        ranking = _compute_district_ranking(merged_trends_df, datasets)
        if ranking is not None:
            out_path = os.path.join(reports_dir, f"district_ranking{suffix}.csv")
            ranking.to_csv(out_path, index=False)
            log.info("Saved district ranking: %s", out_path)

    return None


def _compute_district_ranking(merged_trends_df, datasets):
    """Rank districts by composite VNL + dataset metrics."""
    if "district" not in merged_trends_df.columns:
        return None

    ranking = merged_trends_df[["district"]].copy()

    for col in ["median_radiance", "annual_pct_change"]:
        if col in merged_trends_df.columns:
            ranking[col] = merged_trends_df[col]

    for name in datasets:
        from src.datasets import DATASET_REGISTRY
        module = DATASET_REGISTRY.get(name)
        if module:
            prefix = module.get_meta().short_label + "_"
            for metric in ["literacy_rate", "urbanization_proxy", "workforce_rate"]:
                col = prefix + metric
                if col in merged_trends_df.columns:
                    ranking[col] = merged_trends_df[col]

    if "median_radiance" in ranking.columns:
        ranking = ranking.sort_values("median_radiance", ascending=True)

    return ranking
