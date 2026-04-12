"""Web export step: merge district CSVs + boundary GeoJSON into web-ready artifacts.

This step runs after the district pipeline and produces a self-contained bundle
(``web-export/``) that the public-facing frontend consumes as static assets.

Pure merge + reshape — no new analysis is performed here. All metrics come from
existing CSVs produced by earlier pipeline steps; classifications are re-used
from ``src.formulas.classification``.

Outputs
-------
- ``districts.geojson``   — FeatureCollection, polygon per district with flat metric props
- ``leaderboard.json``    — pre-sorted best/worst rankings
- ``districts/<slug>.json`` — per-district detail (13-year series, trend, rankings, neighbors)
- ``meta.json``           — data vintage, pipeline version, methodology summary, citation

Usage
-----
Invoked by ``src.pipeline_runner`` when the ``--export-web`` flag is set.
Can also be run standalone::

    python3 -m src.outputs.web_export \\
        --csv-dir outputs/<run>/district/csv \\
        --shapefile-path data/shapefiles/maharashtra_district.geojson \\
        --out-dir web-export
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src import config
from src.logging_config import get_pipeline_logger


def classify_alan(radiance: float) -> str:
    """Classify ALAN level from a radiance value (nW/cm²/sr).

    Mirrors ``src.formulas.classification.classify_alan``. Inlined here so the
    web export (and its tests) avoid importing ``src.formulas``, whose package
    ``__init__`` pulls in heavy scientific deps (statsmodels, rasterio) that
    aren't needed for pure reshape work.
    """
    if radiance < config.ALAN_LOW_THRESHOLD:
        return "low"
    if radiance < config.ALAN_MEDIUM_THRESHOLD:
        return "medium"
    return "high"

log = get_pipeline_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────


def slugify(name: str) -> str:
    """Convert a district name to a URL-safe slug (``"Mumbai Suburban"`` → ``"mumbai-suburban"``)."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def round_coords(coords: Any, n: int = 5) -> Any:
    """Recursively round GeoJSON coordinate arrays to ``n`` decimal places."""
    if isinstance(coords, (int, float)):
        return round(float(coords), n)
    return [round_coords(c, n) for c in coords]


def _git_sha(repo_dir: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _percentile_tier(radiance: float, all_values: list[float]) -> str:
    """Return percentile-tier label for a single radiance value against the population."""
    if not all_values:
        return "Unknown"
    sorted_vals = sorted(all_values)
    n = len(sorted_vals)
    rank = sum(1 for v in sorted_vals if v <= radiance)
    pct = rank / n
    if pct <= 0.20:
        return "Pristine"
    if pct <= 0.40:
        return "Low"
    if pct <= 0.60:
        return "Medium"
    if pct <= 0.80:
        return "High"
    return "Very High"


# ── Core export ───────────────────────────────────────────────────────────


def _ci_bounds(row) -> tuple[float | None, float | None]:
    """Pull CI bounds from a trends row, tolerating either naming convention.

    The research pipeline emits ``ci_low`` / ``ci_high`` (used in the CSV
    written by ``src.pipeline_steps.step_fit_trends``). Older or alternate
    callers may supply ``ci_lower`` / ``ci_upper``. Accept both.
    """
    lo = row.get("ci_low") if hasattr(row, "get") else None
    hi = row.get("ci_high") if hasattr(row, "get") else None
    if lo is None or (hasattr(pd, "isna") and pd.isna(lo)):
        lo = row.get("ci_lower") if hasattr(row, "get") else None
    if hi is None or (hasattr(pd, "isna") and pd.isna(hi)):
        hi = row.get("ci_upper") if hasattr(row, "get") else None
    return _safe_float(lo), _safe_float(hi)


def build_district_properties(
    trends_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
) -> dict[str, dict]:
    """Assemble the flat property dict for each district (keyed by district name)."""
    baseline_year = int(yearly_df["year"].min())
    latest_year = int(yearly_df["year"].max())

    latest_vals = (
        yearly_df[yearly_df["year"] == latest_year]
        .set_index("district")["mean_radiance"]
        .to_dict()
    )
    baseline_vals = (
        yearly_df[yearly_df["year"] == baseline_year]
        .set_index("district")["mean_radiance"]
        .to_dict()
    )

    latest_population = list(latest_vals.values())

    # Rankings
    trends_sorted_growth = trends_df.sort_values(
        "annual_pct_change", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    rank_growth = {
        row["district"]: i + 1 for i, row in trends_sorted_growth.iterrows()
    }

    latest_ranked = sorted(
        latest_vals.items(), key=lambda kv: -kv[1]
    )
    rank_latest = {name: i + 1 for i, (name, _v) in enumerate(latest_ranked)}

    props: dict[str, dict] = {}
    for _, row in trends_df.iterrows():
        name = row["district"]
        latest = latest_vals.get(name)
        baseline = baseline_vals.get(name)
        if latest is None or baseline is None:
            log.warning("Missing yearly data for district %s — skipping", name)
            continue
        pct_total = (
            round(((latest - baseline) / baseline) * 100, 2) if baseline else None
        )
        ci_lower, ci_upper = _ci_bounds(row)
        annual_pct = float(row["annual_pct_change"])
        # Composite "conservation score": radiance projected 10 years forward
        # at the current growth rate. Captures BOTH current brightness and
        # growth trajectory — a low-radiance district growing fast can still
        # rank worse than an already-bright district that's saturating.
        # Units: nW/cm²/sr (projected in latest_year + 10).
        try:
            projected = float(latest) * ((1.0 + annual_pct / 100.0) ** 10.0)
        except (OverflowError, ValueError):
            projected = float(latest)
        props[name] = {
            "district": name,
            "slug": slugify(name),
            "latest_year": latest_year,
            "latest_radiance": round(float(latest), 4),
            "baseline_year": baseline_year,
            "baseline_radiance": round(float(baseline), 4),
            "pct_change_total": pct_total,
            "annual_pct_change": round(annual_pct, 3),
            "annual_pct_change_ci_lower": ci_lower,
            "annual_pct_change_ci_upper": ci_upper,
            "alan_class": classify_alan(float(latest)),
            "percentile_tier": _percentile_tier(float(latest), latest_population),
            "rank_annual_growth": rank_growth.get(name),
            "rank_latest_radiance": rank_latest.get(name),
            "projected_radiance_10yr": round(projected, 3),
        }
    # Second pass: rank by projected radiance (ascending = best conserving).
    ordered = sorted(props.values(), key=lambda p: p["projected_radiance_10yr"])
    for i, p in enumerate(ordered):
        p["rank_conservation"] = i + 1
    return props


def _safe_float(v: Any) -> float | None:
    if v is None or pd.isna(v):
        return None
    return round(float(v), 3)


def write_districts_geojson(
    boundary_geojson: dict,
    props_by_district: dict[str, dict],
    out_path: str,
    coord_precision: int = 5,
) -> int:
    """Write merged districts.geojson. Returns feature count."""
    out_features = []
    for feat in boundary_geojson["features"]:
        name = feat["properties"].get("district")
        p = props_by_district.get(name)
        if p is None:
            log.warning("No metrics for boundary district %s — skipping feature", name)
            continue
        geom = dict(feat["geometry"])
        geom["coordinates"] = round_coords(geom["coordinates"], coord_precision)
        out_features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": p,
        })

    fc = {"type": "FeatureCollection", "features": out_features}
    with open(out_path, "w") as f:
        json.dump(fc, f, separators=(",", ":"))
    return len(out_features)


def write_leaderboard(
    props_by_district: dict[str, dict],
    out_path: str,
    data_years: tuple[int, int],
) -> None:
    """Write leaderboard.json with two orderings.

    Ranking uses a composite **conservation score** = radiance projected 10
    years forward at the current growth rate. A district that is already very
    bright (e.g. Mumbai Suburban at ~27 nW/cm²/sr) cannot be "best" just
    because its % growth is modest — its absolute light pollution already
    dwarfs rural districts. Sorting by projected future radiance captures
    both the current state and the trajectory in a single metric.

    Two slices are emitted so the frontend can present them diplomatically:
    - ``best_conserving``: lowest projected 10-year radiance (rural, dark,
      growing slowly) — these are the districts still worth protecting.
    - ``fastest_changing``: highest *annual %* growth (places where night-sky
      loss is accelerating most quickly right now) — presented as a call to
      attention rather than an accusation.
    """
    rows = list(props_by_district.values())

    def lb_row(p: dict, rank: int) -> dict:
        return {
            "rank": rank,
            "district": p["district"],
            "slug": p["slug"],
            "latest_radiance": p["latest_radiance"],
            "annual_pct_change": p["annual_pct_change"],
            "annual_pct_change_ci_lower": p["annual_pct_change_ci_lower"],
            "annual_pct_change_ci_upper": p["annual_pct_change_ci_upper"],
            "alan_class": p["alan_class"],
            "percentile_tier": p["percentile_tier"],
            "projected_radiance_10yr": p["projected_radiance_10yr"],
        }

    # Ascending projected radiance → best conservation
    best = sorted(rows, key=lambda r: r["projected_radiance_10yr"])
    # Descending annual growth → fastest-accelerating light pollution
    fastest = sorted(rows, key=lambda r: -r["annual_pct_change"])

    payload = {
        "ranking_basis": {
            "best_conserving": {
                "metric": "projected_radiance_10yr",
                "formula": "latest_radiance × (1 + annual_pct_change/100)^10",
                "order": "ascending",
                "explanation": (
                    "Radiance projected ten years forward at the current growth "
                    "rate. Rewards districts that are both dark and not brightening "
                    "rapidly. A dim district growing fast, or a bright district "
                    "growing slowly, can both still project poorly."
                ),
            },
            "fastest_changing": {
                "metric": "annual_pct_change",
                "order": "descending",
                "explanation": (
                    "Districts where annual light-pollution growth is highest, "
                    "signalling where new lighting infrastructure is being added "
                    "fastest."
                ),
            },
        },
        "best_conserving": [lb_row(p, i + 1) for i, p in enumerate(best)],
        "fastest_changing": [lb_row(p, i + 1) for i, p in enumerate(fastest)],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_years": list(data_years),
        "total_districts": len(rows),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def write_district_details(
    props_by_district: dict[str, dict],
    yearly_df: pd.DataFrame,
    trends_df: pd.DataFrame,
    out_dir: str,
    boundary_geojson: dict | None = None,
) -> int:
    """Write per-district JSONs. Returns count of files written."""
    os.makedirs(out_dir, exist_ok=True)

    # neighbor map from boundary geojson via shared-edge adjacency — approx:
    # for MVP we just pick up to 3 districts with smallest annual_pct_change delta.
    # A real adjacency check can be added later via geopandas .touches().
    trend_by_name = trends_df.set_index("district")["annual_pct_change"].to_dict()

    count = 0
    for name, p in props_by_district.items():
        series = (
            yearly_df[yearly_df["district"] == name]
            .sort_values("year")
            .to_dict(orient="records")
        )
        trend_row = trends_df[trends_df["district"] == name]
        if trend_row.empty:
            log.warning("Missing trend row for %s", name)
            continue
        tr = trend_row.iloc[0]
        ci_lower, ci_upper = _ci_bounds(tr)

        # nearest-by-growth-rate proxy for "neighbors" until true adjacency lands
        others = sorted(
            [(n, v) for n, v in trend_by_name.items() if n != name],
            key=lambda nv: abs(nv[1] - p["annual_pct_change"]),
        )[:3]
        neighbors = [{"district": n, "annual_pct_change": round(float(v), 3)} for n, v in others]

        detail = {
            "district": name,
            "slug": p["slug"],
            "annual_series": [
                {
                    "year": int(r["year"]),
                    "mean_radiance": round(float(r.get("mean_radiance", 0.0)), 4),
                    "median_radiance": round(float(r.get("median_radiance", 0.0)), 4),
                    "min_radiance": round(float(r.get("min_radiance", 0.0)), 4),
                    "max_radiance": round(float(r.get("max_radiance", 0.0)), 4),
                    "std_radiance": round(float(r.get("std_radiance", 0.0)), 4),
                    "pixel_count": int(r.get("pixel_count", 0)),
                }
                for r in series
            ],
            "trend": {
                "annual_pct_change": round(float(tr["annual_pct_change"]), 3),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "r_squared": _safe_float(tr.get("r_squared")),
                "baseline_year": p["baseline_year"],
                "latest_year": p["latest_year"],
                "baseline_radiance": p["baseline_radiance"],
                "latest_radiance": p["latest_radiance"],
                "pct_change_total": p["pct_change_total"],
            },
            "classification": {
                "alan_class": p["alan_class"],
                "percentile_tier": p["percentile_tier"],
            },
            "neighbors": neighbors,
            "projected_radiance_10yr": p["projected_radiance_10yr"],
            "rankings": {
                "conservation": p["rank_conservation"],
                "annual_growth": p["rank_annual_growth"],
                "latest_radiance": p["rank_latest_radiance"],
                "total_districts": len(props_by_district),
            },
        }
        with open(os.path.join(out_dir, f"{p['slug']}.json"), "w") as f:
            json.dump(detail, f, indent=2)
        count += 1
    return count


def write_meta(
    out_path: str,
    data_years: tuple[int, int],
    repo_dir: str,
    total_districts: int,
) -> None:
    meta = {
        "is_fixture": False,
        "state": "Maharashtra",
        "data_source": "NOAA EOG VIIRS DNB annual composites (v21 for 2012–2013, v22 for 2014+)",
        "year_range": list(data_years),
        "total_districts": total_districts,
        "pipeline_version": _git_sha(repo_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology_summary": (
            "Annual VIIRS DNB radiance is aggregated over each district's boundary. "
            "Per-district log-linear trends fit over the full year range yield annual % "
            "change with bootstrap 95% CIs. ALAN classification uses nW/cm²/sr cutoffs "
            "(low<1, medium 1–5, high>5). Percentile tiers are derived from latest-year "
            "radiance distribution across all districts."
        ),
        "citation": "Maharashtra Dark-Sky Research, AstronEra. Based on NOAA EOG VIIRS DNB.",
    }
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)


# ── Entry point ───────────────────────────────────────────────────────────


def export_web_bundle(
    csv_dir: str,
    shapefile_path: str,
    out_dir: str,
    repo_dir: str | None = None,
) -> dict:
    """Run the full web export. Returns a summary dict of counts and paths."""
    repo_dir = repo_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "districts"), exist_ok=True)

    yearly_path = os.path.join(csv_dir, "districts_yearly_radiance.csv")
    trends_path = os.path.join(csv_dir, "districts_trends.csv")
    if not os.path.exists(yearly_path):
        raise FileNotFoundError(f"Yearly CSV not found: {yearly_path}")
    if not os.path.exists(trends_path):
        raise FileNotFoundError(f"Trends CSV not found: {trends_path}")
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Boundary GeoJSON not found: {shapefile_path}")

    yearly_df = pd.read_csv(yearly_path)
    trends_df = pd.read_csv(trends_path)
    with open(shapefile_path) as f:
        boundary = json.load(f)

    props = build_district_properties(trends_df, yearly_df)

    geo_count = write_districts_geojson(
        boundary, props, os.path.join(out_dir, "districts.geojson")
    )
    write_leaderboard(
        props,
        os.path.join(out_dir, "leaderboard.json"),
        data_years=(int(yearly_df["year"].min()), int(yearly_df["year"].max())),
    )
    detail_count = write_district_details(
        props, yearly_df, trends_df, os.path.join(out_dir, "districts"), boundary
    )
    write_meta(
        os.path.join(out_dir, "meta.json"),
        data_years=(int(yearly_df["year"].min()), int(yearly_df["year"].max())),
        repo_dir=repo_dir,
        total_districts=len(props),
    )

    summary = {
        "districts_geojson_features": geo_count,
        "district_detail_files": detail_count,
        "out_dir": os.path.abspath(out_dir),
    }
    log.info("Web export complete: %s", summary)
    return summary


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Export web-ready bundle from pipeline outputs.")
    parser.add_argument("--csv-dir", required=True, help="Path to outputs/<run>/district/csv/")
    parser.add_argument("--shapefile-path", default=config.DEFAULT_SHAPEFILE_PATH)
    parser.add_argument("--out-dir", default="web-export")
    args = parser.parse_args()
    export_web_bundle(args.csv_dir, args.shapefile_path, args.out_dir)


if __name__ == "__main__":
    _cli()
