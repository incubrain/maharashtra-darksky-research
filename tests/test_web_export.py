"""Unit tests for src.outputs.web_export.

Runs entirely against synthetic DataFrames — no VIIRS rasters, no pipeline
execution. Safe to run anytime.
"""

from __future__ import annotations

import json
import os
import tempfile

import pandas as pd
import pytest

from src.outputs.web_export import (
    build_district_properties,
    export_web_bundle,
    round_coords,
    slugify,
    write_districts_geojson,
    write_leaderboard,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _synthetic_yearly() -> pd.DataFrame:
    rows = []
    for name, baseline, latest in [
        ("Alpha", 10.0, 20.0),
        ("Beta", 1.0, 1.5),
        ("Gamma", 0.1, 0.12),
    ]:
        for i, year in enumerate(range(2012, 2025)):
            # linear interpolation between baseline/latest for simplicity
            frac = i / 12
            v = baseline + (latest - baseline) * frac
            rows.append({
                "district": name,
                "year": year,
                "mean_radiance": v,
                "median_radiance": v * 0.95,
                "min_radiance": v * 0.1,
                "max_radiance": v * 3.0,
                "std_radiance": v * 0.5,
                "pixel_count": 1000 + i * 10,
            })
    return pd.DataFrame(rows)


def _synthetic_trends() -> pd.DataFrame:
    return pd.DataFrame([
        {"district": "Alpha", "annual_pct_change": 5.9, "ci_lower": 5.0, "ci_upper": 6.8, "r_squared": 0.98},
        {"district": "Beta",  "annual_pct_change": 3.4, "ci_lower": 2.5, "ci_upper": 4.3, "r_squared": 0.92},
        {"district": "Gamma", "annual_pct_change": 1.5, "ci_lower": 0.8, "ci_upper": 2.2, "r_squared": 0.81},
    ])


def _synthetic_trends_pipeline_names() -> pd.DataFrame:
    """Mirrors the actual research pipeline's column names (ci_low/ci_high)."""
    return pd.DataFrame([
        {"district": "Alpha", "annual_pct_change": 5.9, "ci_low": 5.0, "ci_high": 6.8, "r_squared": 0.98},
        {"district": "Beta",  "annual_pct_change": 3.4, "ci_low": 2.5, "ci_high": 4.3, "r_squared": 0.92},
        {"district": "Gamma", "annual_pct_change": 1.5, "ci_low": 0.8, "ci_high": 2.2, "r_squared": 0.81},
    ])


def _synthetic_boundary() -> dict:
    def _poly(name, cx, cy):
        ring = [[cx, cy], [cx + 0.1, cy], [cx + 0.1, cy + 0.1], [cx, cy + 0.1], [cx, cy]]
        return {
            "type": "Feature",
            "properties": {"district": name},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }
    return {
        "type": "FeatureCollection",
        "features": [_poly("Alpha", 72.0, 19.0), _poly("Beta", 73.0, 19.0), _poly("Gamma", 80.0, 21.0)],
    }


# ── Helper tests ──────────────────────────────────────────────────────────


def test_slugify_handles_spaces_and_case():
    assert slugify("Mumbai Suburban") == "mumbai-suburban"
    assert slugify("Pune") == "pune"
    assert slugify("  A / B  ") == "a-b"


def test_round_coords_rounds_nested_arrays():
    coords = [[72.123456789, 19.987654321], [73.0, 19.0]]
    assert round_coords(coords, 4) == [[72.1235, 19.9877], [73.0, 19.0]]


# ── Properties build ──────────────────────────────────────────────────────


def test_build_properties_has_required_fields():
    yearly = _synthetic_yearly()
    trends = _synthetic_trends()
    props = build_district_properties(trends, yearly)

    assert set(props) == {"Alpha", "Beta", "Gamma"}
    alpha = props["Alpha"]
    for key in [
        "district", "slug", "latest_year", "latest_radiance",
        "baseline_year", "baseline_radiance", "pct_change_total",
        "annual_pct_change", "annual_pct_change_ci_lower",
        "annual_pct_change_ci_upper", "alan_class", "percentile_tier",
        "rank_annual_growth", "rank_latest_radiance",
    ]:
        assert key in alpha, f"missing {key}"
    assert alpha["latest_year"] == 2024
    assert alpha["baseline_year"] == 2012


def test_rankings_are_dense_and_unique():
    props = build_district_properties(_synthetic_trends(), _synthetic_yearly())
    growth_ranks = sorted(p["rank_annual_growth"] for p in props.values())
    latest_ranks = sorted(p["rank_latest_radiance"] for p in props.values())
    assert growth_ranks == [1, 2, 3]
    assert latest_ranks == [1, 2, 3]


def test_leaderboard_orderings():
    """Best-conserving uses the projected-10yr composite; fastest-changing uses raw growth %."""
    props = build_district_properties(_synthetic_trends(), _synthetic_yearly())
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "leaderboard.json")
        write_leaderboard(props, out, data_years=(2012, 2024))
        with open(out) as f:
            lb = json.load(f)

    # Schema: both slices present, explanation metadata exposed
    assert "best_conserving" in lb and "fastest_changing" in lb
    assert "ranking_basis" in lb

    # Fastest changing = strict descending annual_pct_change
    fastest_pct = [row["annual_pct_change"] for row in lb["fastest_changing"]]
    assert fastest_pct == sorted(fastest_pct, reverse=True)

    # Best conserving = strict ascending projected 10-year radiance
    best_proj = [row["projected_radiance_10yr"] for row in lb["best_conserving"]]
    assert best_proj == sorted(best_proj)


def test_composite_ranking_outranks_pure_growth_sort():
    """Mumbai-Suburban analogue must NOT win best-conserving purely on a low growth %.

    A high-radiance / low-growth district can project worse than a low-radiance /
    high-growth district, so the composite must demote it below the rural one.
    """
    yearly_rows = []
    for name, baseline, latest in [
        ("Bright-slow", 10.0, 12.0),   # high radiance, modest growth
        ("Dim-fast",    0.5,  1.5),    # low radiance but tripling
    ]:
        for i, year in enumerate(range(2012, 2025)):
            v = baseline + (latest - baseline) * (i / 12)
            yearly_rows.append({
                "district": name, "year": year,
                "mean_radiance": v, "median_radiance": v * 0.95,
                "min_radiance": v * 0.1, "max_radiance": v * 3,
                "std_radiance": v * 0.5, "pixel_count": 1000 + i,
            })
    yearly = pd.DataFrame(yearly_rows)
    trends = pd.DataFrame([
        {"district": "Bright-slow", "annual_pct_change": 1.5, "ci_lower": 1.0, "ci_upper": 2.0, "r_squared": 0.95},
        {"district": "Dim-fast",    "annual_pct_change": 9.5, "ci_lower": 8.0, "ci_upper": 11.0, "r_squared": 0.98},
    ])
    props = build_district_properties(trends, yearly)
    # If we sorted by growth only, Bright-slow would be #1 (lowest growth).
    # Composite should put Dim-fast first because 1.5 × 1.095^10 ≈ 3.7 vs
    # 12 × 1.015^10 ≈ 13.9 — projected future radiance matters.
    assert props["Dim-fast"]["rank_conservation"] == 1
    assert props["Bright-slow"]["rank_conservation"] == 2


def test_ci_bounds_accepts_pipeline_column_names():
    """Real pipeline writes ``ci_low`` / ``ci_high``; old callers may use ``ci_lower`` / ``ci_upper``."""
    props_a = build_district_properties(_synthetic_trends(), _synthetic_yearly())
    props_b = build_district_properties(_synthetic_trends_pipeline_names(), _synthetic_yearly())

    for name in ["Alpha", "Beta", "Gamma"]:
        a = props_a[name]
        b = props_b[name]
        assert a["annual_pct_change_ci_lower"] is not None
        assert b["annual_pct_change_ci_lower"] is not None, (
            f"{name}: ci_low/ci_high not being read — the pipeline's CSV would "
            f"produce None CI bounds if this isn't handled"
        )
        assert a["annual_pct_change_ci_lower"] == b["annual_pct_change_ci_lower"]
        assert a["annual_pct_change_ci_upper"] == b["annual_pct_change_ci_upper"]


def test_alan_classification_uses_thresholds():
    """Latest radiance 20 → 'high' (>5); 1.5 → 'medium' (1–5); 0.12 → 'low' (<1)."""
    props = build_district_properties(_synthetic_trends(), _synthetic_yearly())
    assert props["Alpha"]["alan_class"] == "high"
    assert props["Beta"]["alan_class"] == "medium"
    assert props["Gamma"]["alan_class"] == "low"


# ── GeoJSON writer ────────────────────────────────────────────────────────


def test_write_districts_geojson_feature_count_and_props():
    props = build_district_properties(_synthetic_trends(), _synthetic_yearly())
    boundary = _synthetic_boundary()
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "districts.geojson")
        count = write_districts_geojson(boundary, props, out, coord_precision=4)
        assert count == 3
        with open(out) as f:
            fc = json.load(f)
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 3
        props_alpha = next(ft["properties"] for ft in fc["features"]
                           if ft["properties"]["district"] == "Alpha")
        assert props_alpha["slug"] == "alpha"
        # coords rounded to 4 decimals
        coord = fc["features"][0]["geometry"]["coordinates"][0][0][0]
        assert len(str(coord).split(".")[-1]) <= 4


# ── End-to-end against a temp scenario ────────────────────────────────────


def test_export_web_bundle_end_to_end(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    _synthetic_yearly().to_csv(csv_dir / "districts_yearly_radiance.csv", index=False)
    _synthetic_trends().to_csv(csv_dir / "districts_trends.csv", index=False)

    shp = tmp_path / "boundary.geojson"
    with open(shp, "w") as f:
        json.dump(_synthetic_boundary(), f)

    out_dir = tmp_path / "web-export"
    summary = export_web_bundle(str(csv_dir), str(shp), str(out_dir))

    assert summary["districts_geojson_features"] == 3
    assert summary["district_detail_files"] == 3
    assert (out_dir / "districts.geojson").exists()
    assert (out_dir / "leaderboard.json").exists()
    assert (out_dir / "meta.json").exists()
    assert (out_dir / "districts" / "alpha.json").exists()

    with open(out_dir / "districts" / "alpha.json") as f:
        detail = json.load(f)
    assert len(detail["annual_series"]) == 13
    assert detail["trend"]["annual_pct_change"] == 5.9
    assert set(detail["rankings"]) == {"conservation", "annual_growth", "latest_radiance", "total_districts"}

    with open(out_dir / "meta.json") as f:
        meta = json.load(f)
    assert meta["is_fixture"] is False
    assert meta["year_range"] == [2012, 2024]
