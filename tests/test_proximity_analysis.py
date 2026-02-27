"""
Tests for src/analysis/proximity_analysis.py.

Verifies haversine distance, bearing calculations, cardinal direction mapping,
and the compute_nearest_city_distances() aggregation function.

CRITICAL TESTS: These functions underpin dark-sky site proximity metrics
used in the research paper. An error here corrupts distance-based regression.
"""

import math
import os

import numpy as np
import pandas as pd
import pytest

from src.analysis.proximity_analysis import (
    _bearing_deg,
    _bearing_to_cardinal,
    _haversine_km,
    compute_nearest_city_distances,
)
from src.formulas.spatial import EARTH_RADIUS_KM


class TestHaversine:
    """Haversine distance: correctness against known geodetic references."""

    def test_same_point_returns_zero(self):
        """Distance from a point to itself must be exactly zero."""
        assert _haversine_km(19.0, 73.0, 19.0, 73.0) == 0.0

    def test_known_distance_mumbai_pune(self):
        """Mumbai (19.076, 72.878) → Pune (18.520, 73.857): ~120 km.
        Verified against Google Maps geodesic distance."""
        dist = _haversine_km(19.076, 72.878, 18.520, 73.857)
        assert 115 < dist < 130, f"Mumbai-Pune should be ~120 km, got {dist}"

    def test_known_distance_equator_one_degree_longitude(self):
        """At equator, 1° longitude ≈ 111.32 km (WGS84 reference)."""
        dist = _haversine_km(0, 0, 0, 1)
        assert abs(dist - 111.195) < 0.5, f"1° lon at equator should be ~111 km, got {dist}"

    def test_known_distance_one_degree_latitude(self):
        """1° latitude ≈ 111.0 km everywhere (great circle on meridian)."""
        dist = _haversine_km(0, 0, 1, 0)
        assert abs(dist - 111.195) < 0.5, f"1° lat should be ~111 km, got {dist}"

    def test_antipodal_points(self):
        """Opposite poles: half Earth circumference ≈ pi * R ≈ 20015 km."""
        dist = _haversine_km(90, 0, -90, 0)
        expected = math.pi * EARTH_RADIUS_KM
        assert abs(dist - expected) < 1.0, f"Pole-to-pole should be ~{expected}, got {dist}"

    def test_symmetry(self):
        """d(A,B) must equal d(B,A)."""
        d1 = _haversine_km(19.0, 73.0, 21.0, 79.0)
        d2 = _haversine_km(21.0, 79.0, 19.0, 73.0)
        assert d1 == pytest.approx(d2, abs=1e-10)

    def test_triangle_inequality(self):
        """d(A,C) <= d(A,B) + d(B,C) for any three points."""
        a = (19.0, 73.0)  # near Mumbai
        b = (20.0, 76.0)  # mid-Maharashtra
        c = (21.0, 79.0)  # near Nagpur
        dab = _haversine_km(*a, *b)
        dbc = _haversine_km(*b, *c)
        dac = _haversine_km(*a, *c)
        assert dac <= dab + dbc + 1e-6

    def test_negative_coordinates(self):
        """Should work with southern/western hemisphere coordinates."""
        dist = _haversine_km(-33.87, 151.21, -37.81, 144.96)  # Sydney→Melbourne
        assert 700 < dist < 900

    def test_longitude_wraparound(self):
        """Points near the date line (e.g., 179° and -179°) should be ~222 km apart at equator."""
        dist = _haversine_km(0, 179, 0, -179)
        expected = _haversine_km(0, 0, 0, 2)  # 2° at equator
        assert abs(dist - expected) < 0.1

    def test_non_negative(self):
        """Distance should never be negative."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            lat1, lat2 = rng.uniform(-90, 90, 2)
            lon1, lon2 = rng.uniform(-180, 180, 2)
            assert _haversine_km(lat1, lon1, lat2, lon2) >= 0


class TestBearing:
    """Initial bearing calculations: compass direction from point A to point B."""

    def test_due_north(self):
        """Going straight north: bearing should be 0°."""
        b = _bearing_deg(19.0, 73.0, 20.0, 73.0)
        assert abs(b) < 0.1 or abs(b - 360) < 0.1

    def test_due_south(self):
        """Going straight south: bearing should be 180°."""
        b = _bearing_deg(20.0, 73.0, 19.0, 73.0)
        assert abs(b - 180) < 0.1

    def test_due_east_at_equator(self):
        """Going east at the equator: bearing should be 90°."""
        b = _bearing_deg(0, 0, 0, 1)
        assert abs(b - 90) < 0.1

    def test_due_west_at_equator(self):
        """Going west at the equator: bearing should be 270°."""
        b = _bearing_deg(0, 1, 0, 0)
        assert abs(b - 270) < 0.1

    def test_bearing_range_0_to_360(self):
        """Bearing must always be in [0, 360)."""
        rng = np.random.default_rng(99)
        for _ in range(200):
            lat1, lat2 = rng.uniform(-89, 89, 2)
            lon1, lon2 = rng.uniform(-180, 180, 2)
            b = _bearing_deg(lat1, lon1, lat2, lon2)
            assert 0 <= b < 360, f"Bearing {b} out of range"

    def test_reverse_bearing_approximately_opposite(self):
        """Bearing A→B and B→A should differ by ~180° (not exact due to great circle)."""
        b_fwd = _bearing_deg(19.0, 73.0, 21.0, 79.0)
        b_rev = _bearing_deg(21.0, 79.0, 19.0, 73.0)
        diff = abs(b_fwd - b_rev)
        # Should be near 180, but allow some curvature deviation
        assert 170 < diff < 190 or 170 < (360 - diff) < 190


class TestBearingToCardinal:
    """8-point compass conversion from bearing degrees."""

    @pytest.mark.parametrize("bearing,expected", [
        (0, "N"), (22, "N"), (23, "NE"), (45, "NE"),
        (67, "NE"), (68, "E"), (90, "E"), (112, "E"),
        (135, "SE"), (180, "S"), (225, "SW"),
        (270, "W"), (315, "NW"), (337, "NW"),
        (338, "N"), (359.9, "N"),
    ])
    def test_boundary_directions(self, bearing, expected):
        assert _bearing_to_cardinal(bearing) == expected

    def test_exact_boundary_at_360(self):
        """360° should wrap to N (same as 0°)."""
        assert _bearing_to_cardinal(360) == "N"

    def test_all_eight_directions_reachable(self):
        """Each of the 8 cardinal directions should be reachable."""
        results = set()
        for bearing in range(0, 360, 1):
            results.add(_bearing_to_cardinal(bearing))
        assert results == {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}


class TestComputeNearestCityDistances:
    """Integration tests for the full nearest-city computation."""

    def test_single_site_single_city(self):
        """Simplest case: one site, one city."""
        sites = {"SiteA": {"lat": 19.0, "lon": 73.0}}
        cities = {"CityA": {"lat": 19.5, "lon": 73.5}}
        df = compute_nearest_city_distances(sites, cities)
        assert len(df) == 1
        assert df.iloc[0]["site"] == "SiteA"
        assert df.iloc[0]["nearest_city_name"] == "CityA"
        assert df.iloc[0]["distance_km"] > 0

    def test_selects_nearest_from_multiple_cities(self):
        """With 3 cities, should always pick the closest one."""
        sites = {"TestSite": {"lat": 19.0, "lon": 73.0}}
        cities = {
            "FarCity": {"lat": 21.0, "lon": 79.0},
            "NearCity": {"lat": 19.1, "lon": 73.1},
            "MidCity": {"lat": 20.0, "lon": 75.0},
        }
        df = compute_nearest_city_distances(sites, cities)
        assert df.iloc[0]["nearest_city_name"] == "NearCity"

    def test_output_columns(self):
        """Result should have exactly the documented columns."""
        sites = {"S": {"lat": 19.0, "lon": 73.0}}
        cities = {"C": {"lat": 20.0, "lon": 74.0}}
        df = compute_nearest_city_distances(sites, cities)
        expected_cols = {"site", "nearest_city_name", "distance_km",
                         "cardinal_direction", "bearing_degrees"}
        assert set(df.columns) == expected_cols

    def test_empty_cities_raises_or_inf(self):
        """If no cities exist, distance should be inf or function handles gracefully."""
        sites = {"S": {"lat": 19.0, "lon": 73.0}}
        df = compute_nearest_city_distances(sites, {})
        assert len(df) == 1
        # With no cities to compare, nearest_city_name should be None
        assert df.iloc[0]["nearest_city_name"] is None

    def test_csv_output(self, tmp_dir):
        """Output CSV should be written and readable."""
        sites = {"S": {"lat": 19.0, "lon": 73.0}}
        cities = {"C": {"lat": 20.0, "lon": 74.0}}
        csv_path = os.path.join(tmp_dir, "proximity.csv")
        df = compute_nearest_city_distances(sites, cities, output_csv=csv_path)
        assert os.path.exists(csv_path)
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == len(df)

    def test_cardinal_direction_consistency(self):
        """If city is due east, cardinal should be E."""
        sites = {"S": {"lat": 0.0, "lon": 0.0}}
        cities = {"C": {"lat": 0.0, "lon": 5.0}}
        df = compute_nearest_city_distances(sites, cities)
        assert df.iloc[0]["cardinal_direction"] == "E"

    def test_distance_reasonableness_with_config_sites(self):
        """Distances between real config sites and cities should be 0 < d < 1000 km
        (all within Maharashtra, which is ~700 km across)."""
        from src import config
        df = compute_nearest_city_distances(config.DARKSKY_SITES, config.URBAN_CITIES)
        assert len(df) == len(config.DARKSKY_SITES)
        assert (df["distance_km"] > 0).all()
        assert (df["distance_km"] < 1000).all()

    def test_no_site_matches_itself_as_nearest(self):
        """Sites and cities are distinct sets; a site should not match a city at 0 km
        unless co-located."""
        sites = {"S": {"lat": 19.0, "lon": 73.0}}
        cities = {"C": {"lat": 19.0, "lon": 73.0}}
        df = compute_nearest_city_distances(sites, cities)
        # Co-located: distance should be 0
        assert df.iloc[0]["distance_km"] == 0.0


class TestHaversineEdgeCases:
    """Edge cases that could silently corrupt distance data."""

    def test_very_close_points(self):
        """Two points ~1 meter apart: should not return 0 due to float truncation."""
        # ~0.00001° ≈ 1.1 m at these latitudes
        dist = _haversine_km(19.0, 73.0, 19.00001, 73.00001)
        assert dist > 0, "Very close points should not collapse to zero"
        assert dist < 0.01, "Should be less than 10 meters"

    def test_poles(self):
        """Distance from north pole (90,0) should work without math domain errors."""
        dist = _haversine_km(90, 0, 89, 0)
        assert 100 < dist < 120  # ~111 km for 1° latitude
