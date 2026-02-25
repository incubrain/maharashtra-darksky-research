#!/usr/bin/env python3
"""
Build per-district monument CSV files from the raw monument data.

Source: Maharashtra State Archaeology Department official monument lists
(Ratnagiri, Nashik, Pune, Aurangabad, Nanded, Nagpur divisions).

Output: data/monuments/<district>.csv for each district.

Usage:
    python scripts/build_monument_csvs.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.monuments.data_builder import build_monument_csvs

if __name__ == "__main__":
    build_monument_csvs()
