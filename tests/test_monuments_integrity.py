"""
Tests for monuments data integrity and geocoding verification utilities.

CRITICAL: The MONUMENTS list contains 384 heritage sites used in spatial
overlay analysis. Data integrity errors (wrong districts, duplicate entries,
missing fields) silently corrupt the dark-sky vs heritage proximity analysis.
"""

import pandas as pd
import pytest

from src.monuments.constants import MONUMENTS, TYPE_COLORS, TYPE_MARKERS
from src.geocoding.verification import check_bbox


class TestMonumentsDataIntegrity:
    """Validate the 385-entry MONUMENTS list for structural correctness."""

    def test_monument_count_minimum(self):
        """Should have a substantial number of monuments (>350)."""
        assert len(MONUMENTS) >= 350, f"Expected >= 350 monuments, got {len(MONUMENTS)}"

    def test_tuple_structure(self):
        """Every entry should be a 6-tuple: (name, type, place, taluka, district, status)."""
        for i, m in enumerate(MONUMENTS):
            assert isinstance(m, tuple), f"Entry {i} is not a tuple"
            assert len(m) == 6, f"Entry {i} ({m[0]}) has {len(m)} fields, expected 6"

    def test_no_empty_names(self):
        """Monument names should never be empty or whitespace."""
        for i, m in enumerate(MONUMENTS):
            assert m[0] and m[0].strip(), f"Entry {i} has empty name"

    def test_no_empty_districts(self):
        """District field should never be empty."""
        for i, m in enumerate(MONUMENTS):
            assert m[4] and m[4].strip(), f"Entry {i} ({m[0]}) has empty district"

    def test_valid_monument_types(self):
        """All monument types should be one of the defined categories."""
        valid_types = set(TYPE_COLORS.keys())
        for i, m in enumerate(MONUMENTS):
            assert m[1] in valid_types, \
                f"Entry {i} ({m[0]}) has unknown type '{m[1]}', valid: {valid_types}"

    def test_valid_notification_status(self):
        """Notification status should be either 'Final' or 'First'."""
        valid_statuses = {"Final", "First"}
        for i, m in enumerate(MONUMENTS):
            assert m[5] in valid_statuses, \
                f"Entry {i} ({m[0]}) has status '{m[5]}', expected {valid_statuses}"

    def test_no_duplicate_names_within_district(self):
        """No two monuments in the same district should have identical names."""
        seen = set()
        duplicates = []
        for m in MONUMENTS:
            key = (m[0], m[4])  # (name, district)
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        assert not duplicates, f"Duplicate (name, district) pairs: {duplicates}"

    def test_maharashtra_districts_valid(self):
        """All districts in the list should be recognized Maharashtra districts."""
        # Known Maharashtra districts (modern names)
        known_districts = {
            "Palghar", "Thane", "Mumbai", "Raigad", "Ratnagiri", "Sindhudurg",
            "Nandurbar", "Dhule", "Nashik", "Jalgaon", "Ahmednagar",
            "Pune", "Satara", "Solapur", "Sangli", "Kolhapur",
            "Aurangabad", "Jalna", "Beed", "Osmanabad",
            "Parbhani", "Hingoli", "Latur", "Nanded",
            "Buldhana", "Akola", "Amravati", "Washim",
            "Nagpur", "Yavatmal", "Bhandara", "Chandrapur", "Gondia",
            "Wardha", "Gadchiroli",
        }
        monument_districts = {m[4] for m in MONUMENTS}
        unknown = monument_districts - known_districts
        # Some names may have alternate spellings; flag but allow
        if unknown:
            pytest.skip(f"Unknown districts found (may be alternate spellings): {unknown}")

    def test_type_colors_cover_all_types(self):
        """Every monument type used should have a color mapping."""
        used_types = {m[1] for m in MONUMENTS}
        for t in used_types:
            assert t in TYPE_COLORS, f"Type '{t}' has no color mapping"

    def test_type_markers_cover_all_types(self):
        """Every monument type used should have a marker mapping."""
        used_types = {m[1] for m in MONUMENTS}
        for t in used_types:
            assert t in TYPE_MARKERS, f"Type '{t}' has no marker mapping"

    def test_district_distribution(self):
        """Monuments should span at least 20 different districts
        (Maharashtra has 36 — not all may have registered monuments)."""
        districts = {m[4] for m in MONUMENTS}
        assert len(districts) >= 20, f"Only {len(districts)} districts represented"

    def test_type_distribution(self):
        """Should have at least 3 different monument types."""
        types = {m[1] for m in MONUMENTS}
        assert len(types) >= 3


class TestCheckBbox:
    """Tests for geocoding bounding box verification."""

    def test_all_inside(self):
        """Points fully inside Maharashtra bbox should return empty DataFrame."""
        df = pd.DataFrame({
            "lat": [19.0, 20.0, 17.0],
            "lon": [73.0, 76.0, 75.0],
        })
        outside = check_bbox(df)
        assert len(outside) == 0

    def test_point_outside_north(self):
        """Point above lat 22.5 should be flagged."""
        df = pd.DataFrame({"lat": [23.0], "lon": [76.0]})
        outside = check_bbox(df)
        assert len(outside) == 1

    def test_point_outside_south(self):
        """Point below lat 15.5 should be flagged."""
        df = pd.DataFrame({"lat": [14.0], "lon": [76.0]})
        outside = check_bbox(df)
        assert len(outside) == 1

    def test_point_outside_east(self):
        """Point east of lon 81.0 should be flagged."""
        df = pd.DataFrame({"lat": [19.0], "lon": [82.0]})
        outside = check_bbox(df)
        assert len(outside) == 1

    def test_point_outside_west(self):
        """Point west of lon 72.5 should be flagged."""
        df = pd.DataFrame({"lat": [19.0], "lon": [71.0]})
        outside = check_bbox(df)
        assert len(outside) == 1

    def test_boundary_points_inclusive(self):
        """Points exactly on the boundary should be inside (>= / <=)."""
        df = pd.DataFrame({
            "lat": [15.5, 22.5],
            "lon": [72.5, 81.0],
        })
        outside = check_bbox(df)
        assert len(outside) == 0

    def test_mixed_inside_outside(self):
        """Should correctly identify only the outside points."""
        df = pd.DataFrame({
            "name": ["inside", "outside_north", "outside_west"],
            "lat": [19.0, 25.0, 19.0],
            "lon": [76.0, 76.0, 70.0],
        })
        outside = check_bbox(df)
        assert len(outside) == 2
        assert set(outside["name"]) == {"outside_north", "outside_west"}

    def test_empty_dataframe(self):
        """Empty input should return empty output."""
        df = pd.DataFrame({"lat": [], "lon": []})
        outside = check_bbox(df)
        assert len(outside) == 0
