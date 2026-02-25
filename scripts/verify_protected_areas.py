#!/usr/bin/env python3
"""
Verify geocoded protected areas by plotting a random sample on a map.

Usage:
    python scripts/verify_protected_areas.py
    python scripts/verify_protected_areas.py --seed 42
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.protected_areas.verification import verify_sites


def main():
    parser = argparse.ArgumentParser(description="Verify protected area geocoding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    verify_sites(seed=args.seed)


if __name__ == "__main__":
    main()
