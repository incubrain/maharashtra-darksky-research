"""
Migration pathway visualization.

Produces static matplotlib maps showing bird migration patterns through
Maharashtra and neighboring states. Consistent with the project's dark-theme
aesthetic and MAP_DPI = 300 convention.

Map types:
  A. Monthly migration heatmaps (KDE density per IUCN category)
  B. Centroid migration trails (monthly centroid paths)
  C. Entry/exit flow arrows (border crossing directions)
  D. Seasonal summary (district-level choropleth)
"""

import os

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

from src.migration.constants import (
    IUCN_CATEGORIES,
    MIGRATION_OUTPUT_DIR,
    MIGRATION_SEASONS,
    NEIGHBORING_STATES,
    REGION_SHAPEFILE,
    THREATENED_CODES,
)
from src import config

MONTH_NAMES = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# Map extent covering Maharashtra + neighbors.
REGION_EXTENT = {
    "west": 72.0, "east": 81.5,
    "south": 14.5, "north": 23.0,
}


def _load_shapefiles():
    """Load Maharashtra districts and regional state boundaries."""
    maha_gdf = gpd.read_file(config.DEFAULT_SHAPEFILE_PATH)

    region_path = REGION_SHAPEFILE
    if os.path.isfile(region_path):
        region_gdf = gpd.read_file(region_path)
    else:
        region_gdf = None

    return maha_gdf, region_gdf


def _setup_map(ax, maha_gdf, region_gdf=None, title=""):
    """Set up a dark-theme map with boundaries."""
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(REGION_EXTENT["west"], REGION_EXTENT["east"])
    ax.set_ylim(REGION_EXTENT["south"], REGION_EXTENT["north"])
    ax.set_aspect("equal")

    # Draw neighboring state boundaries (faded)
    if region_gdf is not None:
        region_gdf.plot(ax=ax, facecolor="#2a2a3e", edgecolor="#555555",
                        linewidth=0.5, alpha=0.8)

    # Draw Maharashtra districts (more prominent)
    maha_gdf.plot(ax=ax, facecolor="none", edgecolor="#aaaaaa",
                  linewidth=0.6, alpha=0.9)

    # Draw Maharashtra outer boundary (bold)
    maha_gdf.dissolve().boundary.plot(ax=ax, color="#ffffff", linewidth=1.5)

    if title:
        ax.set_title(title, fontsize=11, color="white", pad=8)

    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")


def plot_monthly_heatmaps(
    profiles_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    iucn_code: str,
    output_dir: str = MIGRATION_OUTPUT_DIR,
):
    """Generate 12-panel monthly KDE heatmap for one IUCN category.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        Monthly species profiles (speciesKey, month, obs_count, mean_lat, mean_lon).
    classification_df : pd.DataFrame
        Species classification with iucn_code column.
    iucn_code : str
        IUCN category code (CR, EN, VU, NT, LC).
    output_dir : str
        Directory for output PNG.
    """
    cat_info = IUCN_CATEGORIES.get(iucn_code, {})
    label = cat_info.get("label", iucn_code)
    color = cat_info.get("color", "#ffffff")

    # Filter to species in this IUCN category
    species_in_cat = set(
        classification_df[classification_df["iucn_code"] == iucn_code]["speciesKey"]
        .astype(str)
    )
    cat_profiles = profiles_df[profiles_df["speciesKey"].astype(str).isin(species_in_cat)]

    if len(cat_profiles) == 0:
        print(f"  No data for {iucn_code} ({label}) — skipping heatmap")
        return

    n_species = cat_profiles["speciesKey"].nunique()
    maha_gdf, region_gdf = _load_shapefiles()

    fig, axes = plt.subplots(3, 4, figsize=(20, 16))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle(f"Monthly Bird Density: {label} Species (n={n_species})",
                 fontsize=16, color="white", y=0.98)

    for month_idx in range(12):
        month = month_idx + 1
        ax = axes[month_idx // 4, month_idx % 4]

        _setup_map(ax, maha_gdf, region_gdf, title=MONTH_NAMES[month])

        month_data = cat_profiles[cat_profiles["month"] == month]
        if len(month_data) < 3:
            ax.text(
                0.5, 0.5, "< 3 obs", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#666666",
            )
            continue

        lons = month_data["mean_lon"].values
        lats = month_data["mean_lat"].values
        weights = month_data["obs_count"].values.astype(float)

        # KDE on species centroids weighted by observation count
        try:
            coords = np.vstack([lons, lats])
            kde = gaussian_kde(coords, weights=weights, bw_method=0.3)

            # Evaluate on grid
            xi = np.linspace(REGION_EXTENT["west"], REGION_EXTENT["east"], 200)
            yi = np.linspace(REGION_EXTENT["south"], REGION_EXTENT["north"], 200)
            xx, yy = np.meshgrid(xi, yi)
            zi = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

            ax.contourf(
                xx, yy, zi, levels=15, cmap="plasma", alpha=0.7,
                extent=[REGION_EXTENT["west"], REGION_EXTENT["east"],
                        REGION_EXTENT["south"], REGION_EXTENT["north"]],
            )
        except np.linalg.LinAlgError:
            # Singular matrix — all points at same location
            ax.scatter(lons, lats, s=10, c=color, alpha=0.5, zorder=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"migration_heatmap_{iucn_code}.png")
    fig.savefig(out_path, dpi=config.MAP_DPI, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_centroid_trails(
    profiles_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    iucn_code: str,
    output_dir: str = MIGRATION_OUTPUT_DIR,
):
    """Plot monthly centroid migration trails for migratory species.

    Shows the weighted centroid position for each month connected by arrows.
    Color encodes month (blue=Jan → red=Dec).
    """
    cat_info = IUCN_CATEGORIES.get(iucn_code, {})
    label = cat_info.get("label", iucn_code)

    # Filter to migratory species in this category
    migratory_in_cat = classification_df[
        (classification_df["iucn_code"] == iucn_code) &
        (classification_df["migration_class"].isin(["migratory", "passage"]))
    ]
    species_keys = set(migratory_in_cat["speciesKey"].astype(str))
    cat_profiles = profiles_df[profiles_df["speciesKey"].astype(str).isin(species_keys)]

    if len(cat_profiles) == 0:
        print(f"  No migratory species for {iucn_code} — skipping trails")
        return

    maha_gdf, region_gdf = _load_shapefiles()

    fig, ax = plt.subplots(figsize=(14, 16))
    fig.patch.set_facecolor("#0d0d1a")
    _setup_map(ax, maha_gdf, region_gdf,
               title=f"Migration Centroid Trails: {label}")

    # Month colormap (blue → red)
    cmap = plt.cm.coolwarm
    norm = Normalize(vmin=1, vmax=12)

    for sk in species_keys:
        sp_data = cat_profiles[cat_profiles["speciesKey"].astype(str) == sk].sort_values("month")
        if len(sp_data) < 2:
            continue

        months = sp_data["month"].values
        lons = sp_data["mean_lon"].values
        lats = sp_data["mean_lat"].values
        sizes = np.clip(sp_data["obs_count"].values / 10, 5, 50)

        # Plot points
        for i in range(len(months)):
            ax.scatter(lons[i], lats[i], s=sizes[i], c=[cmap(norm(months[i]))],
                       alpha=0.4, zorder=3, edgecolors="none")

        # Draw arrows between consecutive months
        for i in range(len(months) - 1):
            dx = lons[i+1] - lons[i]
            dy = lats[i+1] - lats[i]
            ax.annotate(
                "", xy=(lons[i+1], lats[i+1]), xytext=(lons[i], lats[i]),
                arrowprops=dict(
                    arrowstyle="->", color=cmap(norm(months[i])),
                    alpha=0.3, lw=0.8,
                ),
                zorder=2,
            )

    # Colorbar for months
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.02)
    cbar.set_ticks(range(1, 13))
    cbar.set_ticklabels(MONTH_NAMES[1:])
    cbar.ax.tick_params(colors="white", labelsize=8)
    cbar.set_label("Month", color="white", fontsize=10)

    # Legend
    n_species = len(species_keys)
    ax.text(
        0.02, 0.02,
        f"{n_species} migratory species\nArrows: monthly centroid movement",
        transform=ax.transAxes, fontsize=9, color="#aaaaaa",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#555555"),
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"migration_trails_{iucn_code}.png")
    fig.savefig(out_path, dpi=config.MAP_DPI, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_entry_exit(
    profiles_df: pd.DataFrame,
    neighbor_profiles_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    iucn_code: str,
    output_dir: str = MIGRATION_OUTPUT_DIR,
):
    """Plot entry/exit flow arrows at Maharashtra borders.

    For each migratory species, detects which neighboring state shows earlier
    presence (entry direction) and later presence (exit direction).
    """
    cat_info = IUCN_CATEGORIES.get(iucn_code, {})
    label = cat_info.get("label", iucn_code)

    migratory_in_cat = classification_df[
        (classification_df["iucn_code"] == iucn_code) &
        (classification_df["migration_class"].isin(["migratory", "passage"]))
    ]
    species_keys = set(migratory_in_cat["speciesKey"].astype(str))

    if len(species_keys) == 0 or neighbor_profiles_df is None or len(neighbor_profiles_df) == 0:
        print(f"  No entry/exit data for {iucn_code} — skipping")
        return

    # Determine entry/exit directions
    # Direction centroids for arrow placement
    direction_coords = {
        "NW": (72.8, 21.5), "N": (77.0, 22.0), "NE": (80.0, 21.0),
        "E": (80.5, 19.0), "SE": (79.0, 16.5), "S": (76.0, 15.5),
        "SW": (73.5, 16.0),
    }

    entry_counts = {}  # direction → count of species entering from there
    exit_counts = {}

    for sk in species_keys:
        sk_str = str(sk)
        # Maharashtra arrival/departure months
        sp_class = migratory_in_cat[migratory_in_cat["speciesKey"].astype(str) == sk_str]
        if len(sp_class) == 0:
            continue
        mh_arrival = sp_class.iloc[0].get("arrival_month")
        mh_departure = sp_class.iloc[0].get("departure_month")
        if pd.isna(mh_arrival) or pd.isna(mh_departure):
            continue
        mh_arrival = int(mh_arrival)
        mh_departure = int(mh_departure)

        # Check each neighbor for earlier/later presence
        sp_neighbors = neighbor_profiles_df[
            neighbor_profiles_df["speciesKey"].astype(str) == sk_str
        ]
        for _, nrow in sp_neighbors.iterrows():
            direction = nrow.get("direction", "")
            n_month = int(nrow["month"])

            # Entry: neighbor has species 1-2 months before Maharashtra arrival
            diff = (mh_arrival - n_month) % 12
            if 1 <= diff <= 2:
                entry_counts[direction] = entry_counts.get(direction, 0) + 1

            # Exit: neighbor has species 1-2 months after Maharashtra departure
            diff = (n_month - mh_departure) % 12
            if 1 <= diff <= 2:
                exit_counts[direction] = exit_counts.get(direction, 0) + 1

    if not entry_counts and not exit_counts:
        print(f"  No directional data for {iucn_code} — skipping")
        return

    maha_gdf, region_gdf = _load_shapefiles()

    fig, ax = plt.subplots(figsize=(14, 16))
    fig.patch.set_facecolor("#0d0d1a")
    _setup_map(ax, maha_gdf, region_gdf,
               title=f"Migration Entry/Exit: {label}")

    # Maharashtra centroid for arrow targets
    mh_centroid = maha_gdf.dissolve().centroid.iloc[0]
    cx, cy = mh_centroid.x, mh_centroid.y

    max_count = max(
        max(entry_counts.values(), default=0),
        max(exit_counts.values(), default=0),
        1,
    )

    # Entry arrows (green, pointing inward)
    for direction, count in entry_counts.items():
        if direction not in direction_coords:
            continue
        ox, oy = direction_coords[direction]
        width = 1 + 4 * (count / max_count)
        ax.annotate(
            "", xy=(cx, cy), xytext=(ox, oy),
            arrowprops=dict(
                arrowstyle="-|>", color="#2ecc71",
                lw=width, alpha=0.7,
            ),
            zorder=4,
        )
        ax.text(ox, oy, f"IN: {count}", fontsize=8, color="#2ecc71",
                ha="center", va="center",
                bbox=dict(facecolor="#1a1a2e", edgecolor="#2ecc71",
                          boxstyle="round,pad=0.3", alpha=0.9),
                zorder=5)

    # Exit arrows (orange, pointing outward)
    for direction, count in exit_counts.items():
        if direction not in direction_coords:
            continue
        ox, oy = direction_coords[direction]
        width = 1 + 4 * (count / max_count)
        # Offset slightly to avoid overlap with entry arrows
        offset_x = (ox - cx) * 0.05
        offset_y = (oy - cy) * 0.05
        ax.annotate(
            "", xy=(ox + offset_x, oy + offset_y),
            xytext=(cx + offset_x, cy + offset_y),
            arrowprops=dict(
                arrowstyle="-|>", color="#e67e22",
                lw=width, alpha=0.7,
            ),
            zorder=4,
        )
        ax.text(ox + offset_x, oy - 0.4, f"OUT: {count}", fontsize=8,
                color="#e67e22", ha="center", va="center",
                bbox=dict(facecolor="#1a1a2e", edgecolor="#e67e22",
                          boxstyle="round,pad=0.3", alpha=0.9),
                zorder=5)

    # Legend
    entry_patch = mpatches.Patch(color="#2ecc71", label="Entry (from neighbor)")
    exit_patch = mpatches.Patch(color="#e67e22", label="Exit (to neighbor)")
    ax.legend(handles=[entry_patch, exit_patch], loc="lower right",
              facecolor="#1a1a2e", edgecolor="#555555", labelcolor="white",
              fontsize=9)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"migration_entry_exit_{iucn_code}.png")
    fig.savefig(out_path, dpi=config.MAP_DPI, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_seasonal_summary(
    classification_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    output_dir: str = MIGRATION_OUTPUT_DIR,
):
    """Generate 4-panel seasonal summary figure.

    Panels:
    1. Migratory species count by IUCN category (bar chart)
    2. Peak arrival months (histogram)
    3. Peak departure months (histogram)
    4. Migration class pie chart (resident/migratory/passage)
    """
    maha_gdf, region_gdf = _load_shapefiles()

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle("Maharashtra Bird Migration: Seasonal Summary",
                 fontsize=16, color="white", y=0.98)

    migratory = classification_df[
        classification_df["migration_class"].isin(["migratory", "passage"])
    ]

    # Panel 1: Species count by IUCN category
    ax = axes[0, 0]
    ax.set_facecolor("#1a1a2e")
    if "iucn_code" in migratory.columns:
        cat_counts = migratory["iucn_code"].value_counts()
        codes = ["CR", "EN", "VU", "NT", "LC", "DD", "NE"]
        counts_ordered = [cat_counts.get(c, 0) for c in codes]
        colors = [IUCN_CATEGORIES[c]["color"] for c in codes]
        labels = [IUCN_CATEGORIES[c]["label"] for c in codes]

        bars = ax.barh(range(len(codes)), counts_ordered, color=colors, alpha=0.85)
        ax.set_yticks(range(len(codes)))
        ax.set_yticklabels(labels, fontsize=9, color="white")
        ax.set_xlabel("Number of migratory species", fontsize=10, color="white")
        ax.tick_params(colors="#888888")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        # Add count labels
        for bar, count in zip(bars, counts_ordered):
            if count > 0:
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        str(count), va="center", fontsize=9, color="white")

    ax.set_title("Migratory Species by IUCN Category", fontsize=11, color="white")

    # Panel 2: Arrival month histogram
    ax = axes[0, 1]
    ax.set_facecolor("#1a1a2e")
    arrival_months = migratory["arrival_month"].dropna().astype(int)
    if len(arrival_months) > 0:
        ax.hist(arrival_months, bins=range(1, 14), color="#2ecc71",
                alpha=0.8, edgecolor="#1a1a2e", align="left")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_NAMES[1:], fontsize=8, color="white")
    ax.set_ylabel("Species count", fontsize=10, color="white")
    ax.set_title("Arrival Month Distribution", fontsize=11, color="white")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    # Panel 3: Departure month histogram
    ax = axes[1, 0]
    ax.set_facecolor("#1a1a2e")
    departure_months = migratory["departure_month"].dropna().astype(int)
    if len(departure_months) > 0:
        ax.hist(departure_months, bins=range(1, 14), color="#e67e22",
                alpha=0.8, edgecolor="#1a1a2e", align="left")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_NAMES[1:], fontsize=8, color="white")
    ax.set_ylabel("Species count", fontsize=10, color="white")
    ax.set_title("Departure Month Distribution", fontsize=11, color="white")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    # Panel 4: Migration class breakdown (pie chart)
    ax = axes[1, 1]
    ax.set_facecolor("#1a1a2e")
    class_counts = classification_df["migration_class"].value_counts()
    class_colors = {"resident": "#3498db", "migratory": "#2ecc71", "passage": "#e67e22"}
    pie_colors = [class_colors.get(c, "#999999") for c in class_counts.index]
    wedges, texts, autotexts = ax.pie(
        class_counts.values, labels=class_counts.index,
        colors=pie_colors, autopct="%1.0f%%",
        textprops={"color": "white", "fontsize": 10},
    )
    for autotext in autotexts:
        autotext.set_color("white")
    ax.set_title("Species Classification", fontsize=11, color="white")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "migration_seasonal_summary.png")
    fig.savefig(out_path, dpi=config.MAP_DPI, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def run_all_visualizations(
    profiles_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    neighbor_profiles_df: pd.DataFrame | None = None,
    output_dir: str = MIGRATION_OUTPUT_DIR,
    categories: list[str] | None = None,
):
    """Run all visualization types.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        Monthly species profiles.
    classification_df : pd.DataFrame
        Species classification with iucn_code.
    neighbor_profiles_df : pd.DataFrame or None
        Neighboring state profiles (for entry/exit maps).
    output_dir : str
        Output directory for PNG files.
    categories : list of str or None
        IUCN codes to generate maps for. Defaults to all with data.
    """
    if categories is None:
        if "iucn_code" in classification_df.columns:
            categories = sorted(
                classification_df["iucn_code"].dropna().unique(),
                key=lambda c: IUCN_CATEGORIES.get(c, {}).get("priority", 99),
            )
        else:
            categories = []

    print(f"\nGenerating visualizations for categories: {categories}")

    # Map A: Monthly heatmaps
    print("\n=== Map A: Monthly Migration Heatmaps ===")
    for code in categories:
        plot_monthly_heatmaps(profiles_df, classification_df, code, output_dir)

    # Map B: Centroid trails
    print("\n=== Map B: Centroid Migration Trails ===")
    for code in categories:
        plot_centroid_trails(profiles_df, classification_df, code, output_dir)

    # Map C: Entry/exit arrows
    if neighbor_profiles_df is not None and len(neighbor_profiles_df) > 0:
        print("\n=== Map C: Entry/Exit Flow ===")
        for code in categories:
            plot_entry_exit(
                profiles_df, neighbor_profiles_df,
                classification_df, code, output_dir,
            )
    else:
        print("\n=== Map C: Skipped (no neighbor data) ===")

    # Map D: Seasonal summary
    print("\n=== Map D: Seasonal Summary ===")
    plot_seasonal_summary(classification_df, profiles_df, output_dir)
