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
    OBSERVATION_POINTS_CSV,
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
    obs_points_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    iucn_code: str,
    output_dir: str = MIGRATION_OUTPUT_DIR,
):
    """Generate 12-panel monthly KDE heatmap for one IUCN category.

    Uses individual observation lat/lon points (not species centroids) so the
    heatmap reflects actual spatial distribution across Maharashtra, not just
    the mean location of each species.

    Parameters
    ----------
    obs_points_df : pd.DataFrame
        Individual observation points (speciesKey, month, lat, lon).
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
    cat_obs = obs_points_df[obs_points_df["speciesKey"].astype(str).isin(species_in_cat)]

    if len(cat_obs) == 0:
        print(f"  No data for {iucn_code} ({label}) — skipping heatmap")
        return

    n_species = cat_obs["speciesKey"].nunique()
    n_obs = len(cat_obs)
    maha_gdf, region_gdf = _load_shapefiles()

    fig, axes = plt.subplots(3, 4, figsize=(20, 16))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle(
        f"Monthly Bird Density: {label} Species "
        f"(n={n_species} species, {n_obs:,} observations)",
        fontsize=14, color="white", y=0.98,
    )

    for month_idx in range(12):
        month = month_idx + 1
        ax = axes[month_idx // 4, month_idx % 4]

        _setup_map(ax, maha_gdf, region_gdf, title=MONTH_NAMES[month])

        month_data = cat_obs[cat_obs["month"] == month]
        if len(month_data) < 10:
            ax.text(
                0.5, 0.5, f"n={len(month_data)}", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#666666",
            )
            continue

        lons = month_data["lon"].values
        lats = month_data["lat"].values

        # KDE on actual observation locations
        try:
            coords = np.vstack([lons, lats])
            kde = gaussian_kde(coords, bw_method=0.15)

            # Evaluate on grid
            xi = np.linspace(REGION_EXTENT["west"], REGION_EXTENT["east"], 250)
            yi = np.linspace(REGION_EXTENT["south"], REGION_EXTENT["north"], 250)
            xx, yy = np.meshgrid(xi, yi)
            zi = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

            ax.contourf(
                xx, yy, zi, levels=20, cmap="plasma", alpha=0.7,
            )
        except np.linalg.LinAlgError:
            # Singular matrix — all points at same location
            ax.scatter(lons, lats, s=5, c=color, alpha=0.3, zorder=3)

        # Add observation count annotation
        ax.text(
            0.98, 0.02, f"n={len(month_data):,}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="#aaaaaa",
        )

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


def _analyze_corridors(
    migratory_in_cat: pd.DataFrame,
    neighbor_profiles_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
) -> dict:
    """Analyze migration corridors between Maharashtra and neighboring states.

    For each species, determines:
    - Which neighboring states the species is observed in
    - Whether the neighbor presence precedes (entry) or follows (exit) Maharashtra
    - The observation strength (total obs count) in each neighbor

    Returns a dict keyed by state name with entry/exit species lists and counts.
    """
    species_keys = set(migratory_in_cat["speciesKey"].astype(str))
    state_names = {
        "NW": "Gujarat", "N": "Madhya Pradesh", "NE": "Chhattisgarh",
        "E": "Telangana", "SE": "Andhra Pradesh", "S": "Karnataka",
        "SW": "Goa",
    }

    corridors = {}
    for direction, state_label in state_names.items():
        corridors[direction] = {
            "state": state_label,
            "entry_species": [],   # species arriving FROM this neighbor
            "exit_species": [],    # species departing TO this neighbor
            "total_obs": 0,        # total neighbor observation count
        }

    for sk in species_keys:
        sk_str = str(sk)
        sp_class = migratory_in_cat[migratory_in_cat["speciesKey"].astype(str) == sk_str]
        if len(sp_class) == 0:
            continue
        row = sp_class.iloc[0]
        mh_arrival = row.get("arrival_month")
        mh_departure = row.get("departure_month")
        species_name = row.get("species_name", row.get("species", sk_str))
        if pd.isna(mh_arrival) or pd.isna(mh_departure):
            continue
        mh_arrival = int(mh_arrival)
        mh_departure = int(mh_departure)

        # Maharashtra presence months
        sp_mh_months = set(
            profiles_df[profiles_df["speciesKey"].astype(str) == sk_str]["month"].astype(int)
        )

        # Check each direction
        sp_neighbors = neighbor_profiles_df[
            neighbor_profiles_df["speciesKey"].astype(str) == sk_str
        ]
        for direction in corridors:
            dir_data = sp_neighbors[sp_neighbors["direction"] == direction]
            if len(dir_data) == 0:
                continue

            dir_months = set(dir_data["month"].astype(int))
            dir_obs = int(dir_data["obs_count"].sum())
            corridors[direction]["total_obs"] += dir_obs

            # Entry: neighbor has species before Maharashtra arrival
            # Check if neighbor months precede Maharashtra arrival
            has_pre_arrival = False
            for nm in dir_months:
                diff = (mh_arrival - nm) % 12
                if 1 <= diff <= 3:
                    has_pre_arrival = True
                    break

            # Exit: neighbor has species after Maharashtra departure
            has_post_departure = False
            for nm in dir_months:
                diff = (nm - mh_departure) % 12
                if 1 <= diff <= 3:
                    has_post_departure = True
                    break

            if has_pre_arrival:
                corridors[direction]["entry_species"].append(
                    (species_name, dir_obs)
                )
            if has_post_departure:
                corridors[direction]["exit_species"].append(
                    (species_name, dir_obs)
                )

    return corridors


def plot_entry_exit(
    profiles_df: pd.DataFrame,
    neighbor_profiles_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    iucn_code: str,
    output_dir: str = MIGRATION_OUTPUT_DIR,
):
    """Plot migration corridor diagram showing flows between Maharashtra and neighbors.

    Two-panel layout:
    - Left: Map with flow arrows from neighboring states into/out of Maharashtra,
      thickness proportional to species count, labeled with state names.
    - Right: Corridor summary table showing species counts and key species per border.
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

    corridors = _analyze_corridors(migratory_in_cat, neighbor_profiles_df, profiles_df)

    # Check if any corridors have data
    has_data = any(
        c["entry_species"] or c["exit_species"]
        for c in corridors.values()
    )
    if not has_data:
        print(f"  No corridor data for {iucn_code} — skipping")
        return

    maha_gdf, region_gdf = _load_shapefiles()

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle(
        f"Migration Corridors: {label} ({len(species_keys)} migratory species)",
        fontsize=15, color="white", y=0.97,
    )

    # Left panel: map with flow arrows
    ax_map = fig.add_axes([0.02, 0.05, 0.58, 0.88])
    _setup_map(ax_map, maha_gdf, region_gdf, title="")

    # Points where arrows start (outside Maharashtra) and end (at border)
    # Outer points sit in the neighboring state; border points sit on the edge
    arrow_geometry = {
        "NW": {"outer": (72.2, 22.0), "border": (73.3, 21.0), "label_pos": (72.0, 22.3)},
        "N":  {"outer": (77.0, 23.0), "border": (77.0, 21.8), "label_pos": (77.0, 23.3)},
        "NE": {"outer": (80.5, 21.8), "border": (79.8, 20.8), "label_pos": (80.8, 22.1)},
        "E":  {"outer": (80.8, 19.0), "border": (79.8, 19.3), "label_pos": (81.2, 19.0)},
        "SE": {"outer": (79.5, 16.3), "border": (78.2, 17.2), "label_pos": (79.8, 16.0)},
        "S":  {"outer": (75.5, 15.0), "border": (75.5, 16.2), "label_pos": (75.5, 14.7)},
        "SW": {"outer": (73.3, 15.2), "border": (73.8, 16.3), "label_pos": (73.0, 14.9)},
    }

    max_species = max(
        max(len(c["entry_species"]) for c in corridors.values()),
        max(len(c["exit_species"]) for c in corridors.values()),
        1,
    )

    for direction, corridor in corridors.items():
        geom = arrow_geometry.get(direction)
        if geom is None:
            continue

        n_entry = len(corridor["entry_species"])
        n_exit = len(corridor["exit_species"])
        state = corridor["state"]
        ox, oy = geom["outer"]
        bx, by = geom["border"]
        lx, ly = geom["label_pos"]

        # State name label
        ax_map.text(
            lx, ly, state, fontsize=9, color="#cccccc",
            ha="center", va="center", fontweight="bold",
            bbox=dict(facecolor="#1a1a2e", edgecolor="#666666",
                      boxstyle="round,pad=0.4", alpha=0.95),
            zorder=6,
        )

        # Entry arrow (green): outer → border
        if n_entry > 0:
            width = 1.0 + 4.0 * (n_entry / max_species)
            # Offset entry arrow slightly to one side
            perp_x = -(by - oy) * 0.03
            perp_y = (bx - ox) * 0.03
            ax_map.annotate(
                "", xy=(bx + perp_x, by + perp_y),
                xytext=(ox + perp_x, oy + perp_y),
                arrowprops=dict(
                    arrowstyle="-|>", color="#2ecc71",
                    lw=width, alpha=0.8,
                    mutation_scale=15,
                ),
                zorder=4,
            )
            # Count badge near the border end
            mid_x = (ox + bx) / 2 + perp_x * 3
            mid_y = (oy + by) / 2 + perp_y * 3
            ax_map.text(
                mid_x, mid_y, str(n_entry), fontsize=8, color="#2ecc71",
                ha="center", va="center", fontweight="bold",
                bbox=dict(facecolor="#0d0d1a", edgecolor="#2ecc71",
                          boxstyle="circle,pad=0.3", alpha=0.95),
                zorder=5,
            )

        # Exit arrow (orange): border → outer
        if n_exit > 0:
            width = 1.0 + 4.0 * (n_exit / max_species)
            perp_x = (by - oy) * 0.03
            perp_y = -(bx - ox) * 0.03
            ax_map.annotate(
                "", xy=(ox + perp_x, oy + perp_y),
                xytext=(bx + perp_x, by + perp_y),
                arrowprops=dict(
                    arrowstyle="-|>", color="#e67e22",
                    lw=width, alpha=0.8,
                    mutation_scale=15,
                ),
                zorder=4,
            )
            mid_x = (ox + bx) / 2 + perp_x * 3
            mid_y = (oy + by) / 2 + perp_y * 3
            ax_map.text(
                mid_x, mid_y, str(n_exit), fontsize=8, color="#e67e22",
                ha="center", va="center", fontweight="bold",
                bbox=dict(facecolor="#0d0d1a", edgecolor="#e67e22",
                          boxstyle="circle,pad=0.3", alpha=0.95),
                zorder=5,
            )

    # Map legend
    entry_patch = mpatches.Patch(color="#2ecc71", label="Entry (from neighbor)")
    exit_patch = mpatches.Patch(color="#e67e22", label="Exit (to neighbor)")
    ax_map.legend(
        handles=[entry_patch, exit_patch], loc="lower left",
        facecolor="#1a1a2e", edgecolor="#555555", labelcolor="white",
        fontsize=9,
    )

    # Right panel: corridor summary table
    ax_table = fig.add_axes([0.62, 0.05, 0.36, 0.88])
    ax_table.set_facecolor("#0d0d1a")
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.axis("off")

    ax_table.text(
        0.5, 0.97, "Corridor Details", fontsize=13, color="white",
        ha="center", va="top", fontweight="bold",
    )

    # Sort directions by total activity (entry + exit species)
    sorted_dirs = sorted(
        corridors.items(),
        key=lambda kv: len(kv[1]["entry_species"]) + len(kv[1]["exit_species"]),
        reverse=True,
    )

    y = 0.92
    for direction, corridor in sorted_dirs:
        n_entry = len(corridor["entry_species"])
        n_exit = len(corridor["exit_species"])
        if n_entry == 0 and n_exit == 0:
            continue

        state = corridor["state"]
        total_obs = corridor["total_obs"]

        # State header
        ax_table.text(
            0.05, y, f"{state} ({direction})", fontsize=11, color="white",
            fontweight="bold", va="top",
        )
        ax_table.text(
            0.95, y, f"{total_obs:,} obs", fontsize=9, color="#888888",
            ha="right", va="top",
        )
        y -= 0.025

        # Entry species
        if n_entry > 0:
            ax_table.text(
                0.08, y, f"Entry: {n_entry} species", fontsize=9,
                color="#2ecc71", va="top",
            )
            y -= 0.02
            # Top 3 species by observation count
            top_entry = sorted(corridor["entry_species"], key=lambda x: x[1], reverse=True)[:3]
            for name, obs in top_entry:
                display_name = name if len(name) <= 30 else name[:28] + "..."
                ax_table.text(
                    0.12, y, f"• {display_name} ({obs:,})", fontsize=8,
                    color="#aaaaaa", va="top", style="italic",
                )
                y -= 0.018

        # Exit species
        if n_exit > 0:
            ax_table.text(
                0.08, y, f"Exit: {n_exit} species", fontsize=9,
                color="#e67e22", va="top",
            )
            y -= 0.02
            top_exit = sorted(corridor["exit_species"], key=lambda x: x[1], reverse=True)[:3]
            for name, obs in top_exit:
                display_name = name if len(name) <= 30 else name[:28] + "..."
                ax_table.text(
                    0.12, y, f"• {display_name} ({obs:,})", fontsize=8,
                    color="#aaaaaa", va="top", style="italic",
                )
                y -= 0.018

        y -= 0.015  # gap between states

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
    obs_points_df: pd.DataFrame | None = None,
    output_dir: str = MIGRATION_OUTPUT_DIR,
    categories: list[str] | None = None,
):
    """Run all visualization types.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        Monthly species profiles (centroids).
    classification_df : pd.DataFrame
        Species classification with iucn_code.
    neighbor_profiles_df : pd.DataFrame or None
        Neighboring state profiles (for entry/exit maps).
    obs_points_df : pd.DataFrame or None
        Individual observation points (for KDE heatmaps). If None, loads
        from OBSERVATION_POINTS_CSV.
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

    # Load observation points for heatmaps
    if obs_points_df is None:
        obs_path = OBSERVATION_POINTS_CSV
        if os.path.isfile(obs_path):
            print(f"Loading observation points from {obs_path}...")
            obs_points_df = pd.read_csv(
                obs_path, compression="gzip", dtype={"speciesKey": str},
            )
            print(f"  {len(obs_points_df):,} observation points loaded")
        else:
            print(f"  WARNING: {obs_path} not found — heatmaps will use centroids")

    print(f"\nGenerating visualizations for categories: {categories}")

    # Map A: Monthly heatmaps (from observation points)
    print("\n=== Map A: Monthly Migration Heatmaps ===")
    if obs_points_df is not None:
        for code in categories:
            plot_monthly_heatmaps(obs_points_df, classification_df, code, output_dir)
    else:
        print("  Skipped — no observation points available")

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
