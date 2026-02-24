#!/usr/bin/env python3
"""
Verify geocoded protected areas by plotting a random sample of 7 sites
on a map of Maharashtra district boundaries.

Produces a PNG with:
- Maharashtra district outlines
- All 79 protected area points (faded)
- 7 randomly sampled sites highlighted with labels

Usage:
    python scripts/verify_protected_areas.py
    python scripts/verify_protected_areas.py --seed 42
"""

import argparse
import glob
import os
import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = "data/protected_areas"
SHAPEFILE_PATH = "data/shapefiles/maharashtra_district.geojson"
OUTPUT_DIR = "data/protected_areas"

CATEGORY_COLORS = {
    "national_park": "#e74c3c",
    "tiger_reserve": "#e67e22",
    "wildlife_sanctuary": "#2ecc71",
    "conservation_reserve": "#3498db",
}

CATEGORY_MARKERS = {
    "national_park": "^",
    "tiger_reserve": "D",
    "wildlife_sanctuary": "o",
    "conservation_reserve": "s",
}


def load_all_sites() -> pd.DataFrame:
    """Load all protected area CSVs into a single DataFrame."""
    frames = []
    for csv_path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        df = pd.read_csv(csv_path)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    # Drop any without coordinates
    combined = combined.dropna(subset=["lat", "lon"])
    return combined


def plot_verification(sites: pd.DataFrame, sample: pd.DataFrame, seed: int):
    """Create verification map."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))

    # Load Maharashtra boundaries if available
    if os.path.exists(SHAPEFILE_PATH):
        gdf = gpd.read_file(SHAPEFILE_PATH)
        gdf.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#999999", linewidth=0.5)
    else:
        print(f"  WARNING: {SHAPEFILE_PATH} not found, plotting without boundaries")

    # Plot all sites (faded)
    for cat, color in CATEGORY_COLORS.items():
        subset = sites[sites["category"] == cat]
        if len(subset) > 0:
            ax.scatter(
                subset["lon"], subset["lat"],
                c=color, marker=CATEGORY_MARKERS[cat],
                s=30, alpha=0.3, zorder=2,
                label=f"{cat.replace('_', ' ').title()} (n={len(subset)})",
            )

    # Plot sample sites (highlighted)
    for _, row in sample.iterrows():
        cat = row["category"]
        color = CATEGORY_COLORS.get(cat, "#333333")
        marker = CATEGORY_MARKERS.get(cat, "o")
        ax.scatter(
            row["lon"], row["lat"],
            c=color, marker=marker,
            s=150, edgecolors="black", linewidths=1.5,
            zorder=5,
        )
        # Label with name and district
        label = row["name"]
        if len(label) > 35:
            label = label[:32] + "..."
        district = row.get("district", "")
        if pd.notna(district) and district:
            label += f"\n({district})"
        ax.annotate(
            label,
            (row["lon"], row["lat"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=7,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.9),
            zorder=6,
        )

    ax.set_title(
        f"Maharashtra Protected Areas — Geocode Verification\n"
        f"7 random samples (seed={seed}) from {len(sites)} total sites",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    # Set bounds to Maharashtra + margin
    ax.set_xlim(72.0, 81.5)
    ax.set_ylim(15.2, 22.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    output_path = os.path.join(OUTPUT_DIR, "verification_map.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Verify protected area geocoding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    print("Protected Area Verification")
    print("=" * 60)

    print("\n1. Loading all geocoded sites...")
    sites = load_all_sites()
    print(f"   {len(sites)} sites with coordinates")

    print(f"\n2. Sampling 7 random sites (seed={args.seed})...")
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(sites), size=7, replace=False)
    sample = sites.iloc[sample_idx].copy()

    print("\n   Selected sites for verification:")
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        status = row.get("geocode_status", "unknown")
        print(f"   {i}. {row['name']}")
        print(f"      District: {row.get('district', 'N/A')}")
        print(f"      Coords: {row['lat']:.4f}°N, {row['lon']:.4f}°E")
        print(f"      Status: {status}")

    print("\n3. Generating verification map...")
    plot_verification(sites, sample, args.seed)

    print("\n4. Coordinate plausibility checks:")
    # Check all coords are within Maharashtra bbox
    bbox_ok = (
        (sites["lat"] >= 15.5) & (sites["lat"] <= 22.5)
        & (sites["lon"] >= 72.5) & (sites["lon"] <= 81.0)
    )
    oob = sites[~bbox_ok]
    if len(oob) > 0:
        print(f"   WARNING: {len(oob)} sites outside Maharashtra bbox:")
        for _, row in oob.iterrows():
            print(f"     {row['name']}: {row['lat']:.4f}°N, {row['lon']:.4f}°E")
    else:
        print(f"   All {len(sites)} sites within Maharashtra bounding box.")

    print("\nDone!")


if __name__ == "__main__":
    main()
