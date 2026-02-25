"""
Shared verification-map generation for geocoded sites.

Both the monument and protected-area verify scripts delegate to this
module for the common load → sample → plot → bbox-check flow.
"""

import glob
import os

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SHAPEFILE_PATH = "data/shapefiles/maharashtra_district.geojson"


def load_all_csvs(data_dir: str) -> pd.DataFrame:
    """Load every CSV in *data_dir* into one DataFrame, dropping rows without coords."""
    frames = []
    for csv_path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        frames.append(pd.read_csv(csv_path))
    combined = pd.concat(frames, ignore_index=True)
    return combined.dropna(subset=["lat", "lon"])


def plot_verification_map(
    sites: pd.DataFrame,
    sample: pd.DataFrame,
    seed: int,
    *,
    title_prefix: str,
    category_col: str,
    colors: dict[str, str],
    markers: dict[str, str],
    output_path: str,
    label_col: str = "name",
    district_col: str = "district",
) -> str:
    """Generate a verification map PNG and return its path."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))

    if os.path.exists(SHAPEFILE_PATH):
        gdf = gpd.read_file(SHAPEFILE_PATH)
        gdf.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#999999", linewidth=0.5)
    else:
        print(f"  WARNING: {SHAPEFILE_PATH} not found, plotting without boundaries")

    # All sites (faded)
    for cat, color in colors.items():
        subset = sites[sites[category_col] == cat]
        if len(subset) == 0:
            continue
        marker = markers.get(cat, "o")
        display = cat.replace("_", " ").title()
        ax.scatter(
            subset["lon"], subset["lat"],
            c=color, marker=marker,
            s=20, alpha=0.25, zorder=2,
            label=f"{display} (n={len(subset)})",
        )

    # Highlighted sample
    for _, row in sample.iterrows():
        cat = row[category_col]
        color = colors.get(cat, "#333333")
        marker = markers.get(cat, "o")
        ax.scatter(
            row["lon"], row["lat"],
            c=color, marker=marker,
            s=150, edgecolors="black", linewidths=1.5, zorder=5,
        )
        label = str(row[label_col])
        if len(label) > 30:
            label = label[:27] + "..."
        district = row.get(district_col, "")
        if pd.notna(district) and district:
            label += f"\n({district})"
        ax.annotate(
            label,
            (row["lon"], row["lat"]),
            textcoords="offset points", xytext=(8, 8),
            fontsize=7, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, alpha=0.9),
            zorder=6,
        )

    ax.set_title(
        f"{title_prefix} — Geocode Verification\n"
        f"7 random samples (seed={seed}) from {len(sites)} total sites",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_xlim(72.0, 81.5)
    ax.set_ylim(15.2, 22.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return output_path


def check_bbox(sites: pd.DataFrame) -> pd.DataFrame:
    """Return rows outside the Maharashtra bounding box."""
    outside = ~(
        (sites["lat"] >= 15.5) & (sites["lat"] <= 22.5)
        & (sites["lon"] >= 72.5) & (sites["lon"] <= 81.0)
    )
    return sites[outside]
