#!/usr/bin/env python3
"""
Verify geocoded monuments by plotting a random sample on a map.

Usage:
    python scripts/verify_monuments.py
    python scripts/verify_monuments.py --seed 42
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.monuments.verification import verify_monuments


def main():
    parser = argparse.ArgumentParser(description="Verify monument geocoding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    verify_monuments(seed=args.seed)


if __name__ == "__main__":
    main()
