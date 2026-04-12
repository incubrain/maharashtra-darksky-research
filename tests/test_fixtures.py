"""Validate that the hand-crafted web-export fixture matches the schema the frontend expects."""

from __future__ import annotations

import json
import os

import pytest

FIXTURE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "src", "outputs", "fixtures", "web_export_sample"
)


def _load(name: str) -> dict:
    with open(os.path.join(FIXTURE_DIR, name)) as f:
        return json.load(f)


def test_districts_geojson_is_valid_feature_collection():
    gj = _load("districts.geojson")
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 3
    for feat in gj["features"]:
        assert feat["type"] == "Feature"
        assert feat["geometry"]["type"] in ("Polygon", "MultiPolygon")


@pytest.mark.parametrize("field", [
    "district", "slug", "latest_year", "latest_radiance",
    "baseline_year", "baseline_radiance", "pct_change_total",
    "annual_pct_change", "annual_pct_change_ci_lower",
    "annual_pct_change_ci_upper", "alan_class", "percentile_tier",
    "rank_annual_growth", "rank_latest_radiance",
])
def test_geojson_features_have_required_props(field):
    gj = _load("districts.geojson")
    for feat in gj["features"]:
        assert field in feat["properties"], f"missing {field} in {feat['properties'].get('district')}"


def test_leaderboard_has_best_and_worst_arrays():
    """Fixture still uses the legacy ``worst_conserving`` key; real exports now emit ``fastest_changing``."""
    lb = _load("leaderboard.json")
    assert "best_conserving" in lb
    # Accept either legacy fixture key or the new key
    counter_key = "fastest_changing" if "fastest_changing" in lb else "worst_conserving"
    assert counter_key in lb
    assert len(lb["best_conserving"]) == 3
    assert len(lb[counter_key]) == 3
    assert [r["rank"] for r in lb["best_conserving"]] == [1, 2, 3]
    assert [r["rank"] for r in lb[counter_key]] == [1, 2, 3]


def test_leaderboard_best_and_fastest_cover_same_set():
    lb = _load("leaderboard.json")
    counter_key = "fastest_changing" if "fastest_changing" in lb else "worst_conserving"
    best = {r["district"] for r in lb["best_conserving"]}
    counter = {r["district"] for r in lb[counter_key]}
    assert best == counter  # same districts, different ordering


def test_meta_flags_fixture():
    meta = _load("meta.json")
    assert meta["is_fixture"] is True
    assert "note" in meta
    assert meta["year_range"] == [2012, 2024]


@pytest.mark.parametrize("slug", ["mumbai-suburban", "pune", "gadchiroli"])
def test_per_district_detail_shape(slug):
    detail = _load(f"districts/{slug}.json")
    assert detail["slug"] == slug
    assert len(detail["annual_series"]) == 13
    years = [row["year"] for row in detail["annual_series"]]
    assert years == list(range(2012, 2025))
    assert "trend" in detail and "classification" in detail and "rankings" in detail
    for tr_field in ["annual_pct_change", "ci_lower", "ci_upper", "r_squared",
                     "baseline_year", "latest_year", "baseline_radiance",
                     "latest_radiance", "pct_change_total"]:
        assert tr_field in detail["trend"], f"missing trend.{tr_field}"
