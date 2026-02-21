import os
import logging
import numpy as np
import rasterio
import rasterio.mask
import rasterio.transform
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import mapping
from scipy import stats
from src import config
from src import viirs_utils

log = logging.getLogger(__name__)

def _setup_plot(district_gdf):
    """Helper to setup a clean map figure."""
    fig, ax = plt.subplots(figsize=(14, 11))
    district_gdf.boundary.plot(ax=ax, edgecolor="white", linewidth=0.3, alpha=0.5)
    ax.set_axis_off()
    return fig, ax

def _add_annotation(ax, text):
    """Helper to add bottom-right annotation."""
    ax.text(0.98, 0.02, text, transform=ax.transAxes,
            fontsize=14, fontweight="bold", color="white",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc="black", alpha=0.7, ec="white"))

def _load_raster(output_dir, year):
    """
    Helper to load a specific year's median raster.
    Applies Dynamic Background Subtraction (DBS) using the 1st percentile (P01)
    as the year-specific noise floor.
    """
    subset_dir = os.path.join(output_dir, "subsets", str(year))
    median_path = os.path.join(subset_dir, f"maharashtra_median_{year}.tif")
    if not os.path.exists(median_path):
        return None, None

    with rasterio.open(median_path) as src:
        data = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    # Dynamic Background Subtraction using central utility
    data = viirs_utils.apply_dynamic_background_subtraction(data, year=year)

    return data, extent

def generate_sprawl_frames(years, output_dir, district_gdf,
                           threshold_nw=config.SPRAWL_THRESHOLD_NW,
                           maps_output_dir=None):
    """
    Generate frames showing the 'Sprawl' (binary lit vs unlit).

    Parameters
    ----------
    years : list[int]
        Years to generate frames for.
    output_dir : str
        Run-level directory containing subsets/.
    district_gdf : gpd.GeoDataFrame
        District boundaries.
    threshold_nw : float
        Radiance threshold to consider 'lit' (default 1.5 nW).
    maps_output_dir : str, optional
        Directory for map outputs. Defaults to output_dir/maps/.
    """
    if maps_output_dir is None:
        maps_output_dir = os.path.join(output_dir, "maps")
    frame_dir = os.path.join(maps_output_dir, "frames", "sprawl")
    os.makedirs(frame_dir, exist_ok=True)
    log.info("Generating Sprawl frames in %s...", frame_dir)

    for year in years:
        data, extent = _load_raster(output_dir, year)
        if data is None: continue

        # Binary mask
        lit_mask = (data >= threshold_nw)
        total_pixels = np.sum(np.isfinite(data))
        lit_pixels = np.sum(lit_mask)
        lit_pct = (lit_pixels / total_pixels * 100) if total_pixels > 0 else 0

        fig, ax = _setup_plot(district_gdf)

        # Create an RGBA image
        rows, cols = data.shape
        img = np.zeros((rows, cols, 4))
        img[lit_mask] = [1, 0.9, 0.2, 1] # Yellow for lit

        ax.imshow(img, extent=extent, origin="upper", zorder=1)

        fig.patch.set_facecolor('black')
        ax.set_title(f"Urban Sprawl (Lit > {threshold_nw} nW) - {year}", fontsize=16, color='white')

        text = (
            f"Year: {year}\n"
            f"Lit Area Index: {lit_pct:.1f}%\n"
            f"(Threshold: {threshold_nw} nW/cm²/sr)"
        )
        _add_annotation(ax, text)

        path = os.path.join(frame_dir, f"sprawl_{year}.png")
        fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight", facecolor='black')
        plt.close(fig)
        log.info("Generated: %s", path)

def generate_differential_frames(years, output_dir, district_gdf,
                                 maps_output_dir=None):
    """
    Generate frames showing (Current Year - Baseline).

    Parameters
    ----------
    years : list[int]
        Years to generate frames for.
    output_dir : str
        Run-level directory containing subsets/.
    district_gdf : gpd.GeoDataFrame
        District boundaries.
    maps_output_dir : str, optional
        Directory for map outputs. Defaults to output_dir/maps/.
    """
    if maps_output_dir is None:
        maps_output_dir = os.path.join(output_dir, "maps")
    frame_dir = os.path.join(maps_output_dir, "frames", "differential")
    os.makedirs(frame_dir, exist_ok=True)
    log.info("Generating Differential frames in %s...", frame_dir)

    baseline_year = min(years)
    base_data, extent = _load_raster(output_dir, baseline_year)
    if base_data is None:
        log.error("Baseline year %s data missing.", baseline_year)
        return

    for year in years:
        data, _ = _load_raster(output_dir, year)
        if data is None: continue

        diff = data - base_data

        # Determine stats
        net_increase = np.nanmean(diff)

        fig, ax = _setup_plot(district_gdf)
        fig.patch.set_facecolor('black')

        im = ax.imshow(diff, extent=extent, cmap="RdBu_r", vmin=-5, vmax=5,
                       origin="upper", aspect="auto")

        text = (
            f"Year: {year}\n"
            f"New Light vs {baseline_year}\n"
            f"Net Change: {net_increase:+.2f} nW"
        )
        _add_annotation(ax, text)

        cbar = plt.colorbar(im, ax=ax, shrink=0.6, label="Difference (nW/cm²/sr)")
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.set_label("Change vs 2012 (nW/cm²/sr)", color='white')

        ax.set_title(f"New Light: {year} vs {baseline_year}", fontsize=16, color='white')

        path = os.path.join(frame_dir, f"diff_{year}.png")
        fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight", facecolor='black')
        plt.close(fig)
        log.info("Generated: %s", path)

def generate_darkness_frames(years, output_dir, district_gdf,
                             threshold_nw=config.DARKNESS_THRESHOLD_NW,
                             maps_output_dir=None):
    """
    Generate frames showing 'Erosion of Darkness' (pixels < threshold).

    Parameters
    ----------
    years : list[int]
        Years to generate frames for.
    output_dir : str
        Run-level directory containing subsets/.
    district_gdf : gpd.GeoDataFrame
        District boundaries.
    threshold_nw : float
        Dark sky threshold (default 0.25 nW - 'Pristine').
    maps_output_dir : str, optional
        Directory for map outputs. Defaults to output_dir/maps/.
    """
    if maps_output_dir is None:
        maps_output_dir = os.path.join(output_dir, "maps")
    frame_dir = os.path.join(maps_output_dir, "frames", "darkness")
    os.makedirs(frame_dir, exist_ok=True)
    log.info("Generating Darkness frames in %s...", frame_dir)

    for year in years:
        data, extent = _load_raster(output_dir, year)
        if data is None: continue

        # Dark mask
        dark_mask = (data < threshold_nw) & (data > 0)

        total_pixels = np.sum(np.isfinite(data))
        dark_pixels = np.sum(dark_mask)
        dark_pct = (dark_pixels / total_pixels * 100) if total_pixels > 0 else 0

        fig, ax = _setup_plot(district_gdf)
        fig.patch.set_facecolor('black')

        img = np.zeros((data.shape[0], data.shape[1], 4))
        img[dark_mask] = [0.0, 0.8, 0.6, 1.0]
        lit_mask = (data >= threshold_nw)
        img[lit_mask] = [0.2, 0.2, 0.2, 1.0]

        ax.imshow(img, extent=extent, origin="upper")

        text = (
            f"Year: {year}\n"
            f"Dark Reservoirs: {dark_pct:.1f}%\n"
            f"(< {threshold_nw} nW/cm²/sr)"
        )
        _add_annotation(ax, text)
        ax.set_title(f"Erosion of Darkness ({year})", fontsize=16, color='white')

        path = os.path.join(frame_dir, f"darkness_{year}.png")
        fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight", facecolor='black')
        plt.close(fig)
        log.info("Generated: %s", path)

def generate_trend_map(years, output_dir, district_gdf, maps_output_dir=None):
    """
    Generate pixel-wise linear trend map (slope of radiance over years).

    Parameters
    ----------
    years : list[int]
        Years to compute trend across.
    output_dir : str
        Run-level directory containing subsets/.
    district_gdf : gpd.GeoDataFrame
        District boundaries.
    maps_output_dir : str, optional
        Directory for map outputs. Defaults to output_dir/maps/.
    """
    if maps_output_dir is None:
        maps_output_dir = os.path.join(output_dir, "maps")
    os.makedirs(maps_output_dir, exist_ok=True)
    log.info("Generating Trend Map...")

    # Load all years into a stack
    stack = []
    valid_years = []

    ref_raster, extent = _load_raster(output_dir, years[0])
    if ref_raster is None: return

    for year in years:
        data, _ = _load_raster(output_dir, year)
        if data is not None:
            stack.append(data)
            valid_years.append(year)

    if not stack:
        return

    stack_arr = np.array(stack) # (T, H, W)

    x = np.array(valid_years)
    x = x - x[0] # Time delta

    log.info("Computing regression slope for %s pixels...", stack_arr.shape)

    T, H, W = stack_arr.shape
    reshaped = stack_arr.reshape(T, -1)

    # Fast vectorized slope: slope = cov(x, y) / var(x)
    x_mean = np.mean(x)
    y_mean = np.mean(reshaped, axis=0)
    numerator = np.sum((x[:, None] - x_mean) * (reshaped - y_mean), axis=0)
    denominator = np.sum((x[:, None] - x_mean)**2, axis=0)
    slope = numerator / denominator
    slope_map = slope.reshape(H, W)

    fig, ax = _setup_plot(district_gdf)
    fig.patch.set_facecolor('black')

    im = ax.imshow(slope_map, extent=extent, cmap="coolwarm", vmin=-2, vmax=2,
                   origin="upper", aspect="auto")

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, label="Rate of Change (nW/year)")
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    cbar.set_label("Trend Slope (nW/year)", color='white')

    ax.set_title(f"Light Pollution Growth Trend ({years[0]}-{years[-1]})", fontsize=16, color='white')

    path = os.path.join(maps_output_dir, "alan_trend_map.png")
    fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight", facecolor='black')
    plt.close(fig)
    log.info("Saved Trend Map: %s", path)


def generate_per_district_radiance_maps(output_dir, year, district_gdf,
                                        maps_output_dir=None):
    """Generate zoomed-in radiance raster maps clipped to each district.

    For each district, clips the median raster to the district boundary and
    renders a zoomed-in map showing actual raster radiance values.

    Parameters
    ----------
    output_dir : str
        Run-level directory containing subsets/.
    year : int
        Year to render (typically the latest).
    district_gdf : gpd.GeoDataFrame
        District boundaries with 'district' column.
    maps_output_dir : str, optional
        Where to write maps. Defaults to output_dir/maps/.

    Returns
    -------
    int
        Number of district maps generated.
    """
    if maps_output_dir is None:
        maps_output_dir = os.path.join(output_dir, "maps")

    district_maps_dir = os.path.join(maps_output_dir, "districts")
    os.makedirs(district_maps_dir, exist_ok=True)

    subset_dir = os.path.join(output_dir, "subsets", str(year))
    median_path = os.path.join(subset_dir, f"maharashtra_median_{year}.tif")

    if not os.path.exists(median_path):
        log.warning("No raster data for year %d: %s", year, median_path)
        return 0

    count = 0
    for _, row in district_gdf.iterrows():
        district_name = row["district"]
        try:
            # Clip raster to district boundary
            with rasterio.open(median_path) as src:
                geom = [mapping(row.geometry)]
                clipped, clipped_transform = rasterio.mask.mask(
                    src, geom, crop=True, filled=True, nodata=np.nan
                )
                clipped_data = clipped[0]
                clipped_bounds = rasterio.transform.array_bounds(
                    clipped_data.shape[0], clipped_data.shape[1], clipped_transform
                )

            # Apply DBS
            clipped_data = viirs_utils.apply_dynamic_background_subtraction(
                clipped_data, year=year
            )

            clipped_extent = [
                clipped_bounds[0], clipped_bounds[2],
                clipped_bounds[1], clipped_bounds[3],
            ]

            fig, ax = plt.subplots(figsize=(10, 9))
            fig.patch.set_facecolor('black')

            # Log-scale for visibility
            display_data = np.log10(np.clip(clipped_data, 0.01, None))
            display_data = np.where(np.isfinite(display_data), display_data, np.nan)

            im = ax.imshow(
                display_data, extent=clipped_extent, cmap="magma",
                vmin=-2, vmax=2, origin="upper", aspect="auto"
            )

            # Overlay district boundary
            single_gdf = district_gdf[
                district_gdf["district"] == district_name
            ]
            single_gdf.boundary.plot(
                ax=ax, edgecolor="white", linewidth=1.0, alpha=0.8
            )

            cbar = plt.colorbar(im, ax=ax, shrink=0.6)
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            cbar.set_label("log\u2081\u2080(Radiance nW/cm\u00b2/sr)", color='white')

            ax.set_title(
                f"{district_name}: Nighttime Radiance ({year})",
                fontsize=14, color='white'
            )
            ax.set_axis_off()
            plt.tight_layout()

            safe_name = district_name.lower().replace(" ", "_")
            path = os.path.join(
                district_maps_dir, f"{safe_name}_radiance.png"
            )
            fig.savefig(
                path, dpi=config.MAP_DPI, bbox_inches="tight",
                facecolor='black'
            )
            plt.close(fig)
            count += 1

        except Exception as exc:
            log.warning(
                "Failed to generate radiance map for %s: %s",
                district_name, exc
            )
            plt.close("all")
            continue

    log.info(
        "Generated %d per-district radiance maps in %s",
        count, district_maps_dir
    )
    return count
