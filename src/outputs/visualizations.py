"""
Visualization frame generators for Maharashtra VIIRS ALAN analysis.

Background correction uses a **dark-reference-area** approach following
Coesfeld et al. (2020):  for each year the median radiance is sampled
inside 10 km buffers around three protected-area dark-sky sites —
Pench Tiger Reserve, Tadoba Tiger Reserve, and Yawal Wildlife Sanctuary.
The median of the three site medians is subtracted from the state raster
as the year-specific natural background estimate.  This is more robust
than the original per-year P01 DBS because the reference sites are
physically meaningful (genuinely unlit areas) rather than a statistical
artefact of the raster's own noise distribution.

Citation:
    Coesfeld, J., Kuester, T., Kuechly, H.U. & Kyba, C.C.M. (2020).
    Reducing Variability and Removing Natural Light from Nighttime
    Satellite Imagery.  Sensors, 20(11), 3287.
    https://doi.org/10.3390/s20113287
"""

import os
import logging
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rasterio.transform
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, mapping
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from scipy import stats
from scipy.ndimage import gaussian_filter, zoom
from src import config

log = logging.getLogger(__name__)

# ── Dark-reference sites for background calibration ─────────────────
# Selected from config.DARKSKY_SITES based on: (1) lowest mean radiance,
# (2) highest temporal stability (lowest CV), (3) adequate pixel count,
# (4) non-coastal.  See CHANGELOG.md for full selection methodology.
DARK_REFERENCE_SITES = {
    "Pench Tiger Reserve":      config.DARKSKY_SITES["Pench Tiger Reserve"],
    "Tadoba Tiger Reserve":     config.DARKSKY_SITES["Tadoba Tiger Reserve"],
    "Yawal Wildlife Sanctuary": config.DARKSKY_SITES["Yawal Wildlife Sanctuary"],
}
DARK_REF_BUFFER_KM = config.SITE_BUFFER_RADIUS_KM  # 10 km

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

def _make_clip_patch(gdf_subset, ax):
    """Create a matplotlib clip path from a GeoDataFrame's geometry."""
    verts = []
    codes = []
    for geom_part in gdf_subset.geometry.values:
        polys = (geom_part.geoms if hasattr(geom_part, 'geoms')
                 else [geom_part])
        for poly in polys:
            ring = np.array(poly.exterior.coords)
            verts.extend(ring.tolist())
            codes.extend(
                [MplPath.MOVETO]
                + [MplPath.LINETO] * (len(ring) - 2)
                + [MplPath.CLOSEPOLY]
            )
    patch = PathPatch(
        MplPath(verts, codes), transform=ax.transData,
        facecolor='none', edgecolor='none'
    )
    ax.add_patch(patch)
    return patch


def _load_raster(output_dir, year, background=None):
    """
    Load a year's median raster with optional background subtraction.

    Parameters
    ----------
    output_dir : str
        Run-level directory containing ``subsets/<year>/``.
    year : int
        Year to load.
    background : float or None
        Dark-reference background value to subtract (nW/cm²/sr).
        When provided, ``max(0, data - background)`` is applied.
        When ``None``, raw radiance is returned unchanged.

    Returns
    -------
    tuple[np.ndarray | None, list | None]
        ``(data, extent)`` or ``(None, None)`` if the file is missing.
    """
    subset_dir = os.path.join(output_dir, "subsets", str(year))
    median_path = os.path.join(subset_dir, f"maharashtra_median_{year}.tif")
    if not os.path.exists(median_path):
        return None, None

    with rasterio.open(median_path) as src:
        data = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    if background is not None:
        data = np.maximum(0, data - background)

    return data, extent


def compute_dark_reference_backgrounds(output_dir, years):
    """Sample dark-reference sites to estimate per-year natural background.

    For each year, clips the state raster to each dark-reference site
    buffer, computes the median radiance of valid pixels inside the
    buffer, then takes the median across the three sites as the final
    background estimate.

    Parameters
    ----------
    output_dir : str
        Run-level directory containing ``subsets/<year>/``.
    years : list[int]
        Years to compute backgrounds for.

    Returns
    -------
    dict[int, float]
        ``{year: background_nw}`` mapping.
    list[dict]
        Per-site per-year audit records (for CSV export).
    """
    # Build site buffers in WGS84
    buffers = {}
    for name, info in DARK_REFERENCE_SITES.items():
        pt = Point(info["lon"], info["lat"])
        # Buffer in UTM for metric accuracy, then back to WGS84
        pt_gdf = gpd.GeoDataFrame(
            [{"name": name}], geometry=[pt], crs="EPSG:4326"
        )
        pt_utm = pt_gdf.to_crs(epsg=config.MAHARASHTRA_UTM_EPSG)
        pt_utm["geometry"] = pt_utm.geometry.buffer(DARK_REF_BUFFER_KM * 1000)
        buf_wgs = pt_utm.to_crs("EPSG:4326")
        buffers[name] = buf_wgs.geometry.values[0]

    backgrounds = {}
    audit_rows = []

    for year in years:
        subset_dir = os.path.join(output_dir, "subsets", str(year))
        median_path = os.path.join(subset_dir, f"maharashtra_median_{year}.tif")
        if not os.path.exists(median_path):
            log.warning("Dark-ref: no raster for %d, skipping", year)
            continue

        site_medians = []
        with rasterio.open(median_path) as src:
            for name, buf_geom in buffers.items():
                try:
                    clipped, _ = rasterio.mask.mask(
                        src, [mapping(buf_geom)],
                        crop=True, filled=True, nodata=np.nan,
                    )
                    pixels = clipped[0]
                    valid = pixels[np.isfinite(pixels) & (pixels > 0)]
                    n_valid = len(valid)
                    if n_valid > 0:
                        site_med = float(np.median(valid))
                        site_mean = float(np.mean(valid))
                        site_p95 = float(np.percentile(valid, 95))
                    else:
                        site_med = 0.0
                        site_mean = 0.0
                        site_p95 = 0.0
                    site_medians.append(site_med)
                    audit_rows.append({
                        "year": year,
                        "site": name,
                        "median_nw": round(site_med, 6),
                        "mean_nw": round(site_mean, 6),
                        "p95_nw": round(site_p95, 6),
                        "valid_pixels": n_valid,
                    })
                except Exception as exc:
                    log.warning("Dark-ref clip failed for %s/%d: %s",
                                name, year, exc)
                    audit_rows.append({
                        "year": year, "site": name,
                        "median_nw": np.nan, "mean_nw": np.nan,
                        "p95_nw": np.nan, "valid_pixels": 0,
                    })

        if site_medians:
            bg = float(np.median(site_medians))
        else:
            bg = 0.0
        backgrounds[year] = bg
        log.info(
            "Dark-ref background %d: %.4f nW  [%s]",
            year, bg,
            ", ".join(f"{name}: {m:.4f}" for name, m
                      in zip(buffers.keys(), site_medians)),
        )

    return backgrounds, audit_rows


def save_dark_reference_audit(audit_rows, output_dir):
    """Write the per-site per-year dark-reference audit to CSV.

    Parameters
    ----------
    audit_rows : list[dict]
        Records from ``compute_dark_reference_backgrounds()``.
    output_dir : str
        Run-level directory; CSV written to ``diagnostics/``.
    """
    if not audit_rows:
        return
    diag_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    path = os.path.join(diag_dir, "dark_reference_backgrounds.csv")
    pd.DataFrame(audit_rows).to_csv(path, index=False)
    log.info("Dark-reference audit saved: %s", path)

def generate_sprawl_frames(years, output_dir, district_gdf,
                           threshold_nw=config.SPRAWL_THRESHOLD_NW,
                           maps_output_dir=None, backgrounds=None):
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
    backgrounds = backgrounds or {}
    if maps_output_dir is None:
        maps_output_dir = os.path.join(output_dir, "maps")
    frame_dir = os.path.join(maps_output_dir, "frames", "sprawl")
    os.makedirs(frame_dir, exist_ok=True)
    log.info("Generating Sprawl frames in %s...", frame_dir)

    for year in years:
        data, extent = _load_raster(output_dir, year, background=backgrounds.get(year))
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
                                 maps_output_dir=None, backgrounds=None):
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
    backgrounds = backgrounds or {}
    if maps_output_dir is None:
        maps_output_dir = os.path.join(output_dir, "maps")
    frame_dir = os.path.join(maps_output_dir, "frames", "differential")
    os.makedirs(frame_dir, exist_ok=True)
    log.info("Generating Differential frames in %s...", frame_dir)

    baseline_year = min(years)
    base_data, extent = _load_raster(output_dir, baseline_year,
                                     background=backgrounds.get(baseline_year))
    if base_data is None:
        log.error("Baseline year %s data missing.", baseline_year)
        return

    for year in years:
        data, _ = _load_raster(output_dir, year, background=backgrounds.get(year))
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
                             maps_output_dir=None, backgrounds=None):
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
    backgrounds = backgrounds or {}
    if maps_output_dir is None:
        maps_output_dir = os.path.join(output_dir, "maps")
    frame_dir = os.path.join(maps_output_dir, "frames", "darkness")
    os.makedirs(frame_dir, exist_ok=True)
    log.info("Generating Darkness frames in %s...", frame_dir)

    for year in years:
        data, extent = _load_raster(output_dir, year, background=backgrounds.get(year))
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

def generate_trend_map(years, output_dir, district_gdf, maps_output_dir=None,
                       backgrounds=None):
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
    backgrounds = backgrounds or {}
    if maps_output_dir is None:
        maps_output_dir = os.path.join(output_dir, "maps")
    os.makedirs(maps_output_dir, exist_ok=True)
    log.info("Generating Trend Map...")

    # Load all years into a stack
    stack = []
    valid_years = []

    ref_raster, extent = _load_raster(output_dir, years[0],
                                      background=backgrounds.get(years[0]))
    if ref_raster is None: return

    for year in years:
        data, _ = _load_raster(output_dir, year, background=backgrounds.get(year))
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


def generate_light_increase_frames(years, output_dir, district_gdf,
                                   maps_output_dir=None, backgrounds=None):
    """Generate per-year state-level radiance heatmap frames.

    Shows absolute nighttime radiance intensity across the entire state for
    each year, with year-over-year change annotation.  Designed as GIF frames
    for the "Light Increase" animation.

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
    backgrounds = backgrounds or {}
    if maps_output_dir is None:
        maps_output_dir = os.path.join(output_dir, "maps")
    frame_dir = os.path.join(maps_output_dir, "frames", "light_increase")
    os.makedirs(frame_dir, exist_ok=True)
    log.info("Generating Light Increase frames in %s...", frame_dir)

    baseline_year = min(years)
    baseline_data, _ = _load_raster(output_dir, baseline_year,
                                    background=backgrounds.get(baseline_year))
    baseline_mean = (float(np.nanmean(baseline_data))
                     if baseline_data is not None else None)

    for year in years:
        data, extent = _load_raster(output_dir, year,
                                    background=backgrounds.get(year))
        if data is None:
            continue

        fig, ax = _setup_plot(district_gdf)
        fig.patch.set_facecolor('black')

        # Log-scale for visibility (same as per-district maps)
        display_data = np.log10(np.clip(data, 0.01, None))
        display_data = np.where(np.isfinite(display_data), display_data, np.nan)

        im = ax.imshow(display_data, extent=extent, cmap="magma",
                       vmin=-2, vmax=2, origin="upper", aspect="auto",
                       interpolation="bilinear")

        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.set_label("log\u2081\u2080(Radiance nW/cm\u00b2/sr)", color='white')

        # Compute statistics
        current_mean = float(np.nanmean(data))

        baseline_text = ""
        if baseline_mean and baseline_mean > 0 and year != baseline_year:
            abs_change = current_mean - baseline_mean
            pct_change = (abs_change / baseline_mean) * 100
            baseline_text = (
                f"\nvs {baseline_year}: {abs_change:+.3f} nW ({pct_change:+.1f}%)"
            )

        text = (
            f"Year: {year}\n"
            f"Mean Radiance: {current_mean:.3f} nW"
            f"{baseline_text}"
        )
        _add_annotation(ax, text)

        ax.set_title(f"Maharashtra: Light Pollution ({year})",
                     fontsize=16, color='white')

        path = os.path.join(frame_dir, f"light_increase_{year}.png")
        fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight",
                    facecolor='black')
        plt.close(fig)
        log.info("Generated: %s", path)


def generate_per_district_radiance_frames(years, output_dir, district_gdf,
                                          maps_output_dir=None,
                                          backgrounds=None):
    """Generate per-district zoomed radiance frames for every year.

    For each district × year combination, clips the median raster to the
    district boundary and renders a zoomed-in map.  Output is organised as
    ``frames/districts/<district_name>/radiance_<year>.png`` so each district
    directory can be independently assembled into a GIF.

    Parameters
    ----------
    years : list[int]
        Years to generate frames for.
    output_dir : str
        Run-level directory containing subsets/.
    district_gdf : gpd.GeoDataFrame
        District boundaries with 'district' column.
    maps_output_dir : str, optional
        Where to write maps. Defaults to output_dir/maps/.

    Returns
    -------
    int
        Total number of frames generated.
    """
    backgrounds = backgrounds or {}
    if maps_output_dir is None:
        maps_output_dir = os.path.join(output_dir, "maps")

    frames_base = os.path.join(maps_output_dir, "frames", "districts")
    os.makedirs(frames_base, exist_ok=True)
    log.info("Generating per-district radiance frames in %s...", frames_base)

    baseline_year = min(years)
    total_count = 0
    for _, row in district_gdf.iterrows():
        district_name = row["district"]
        safe_name = district_name.lower().replace(" ", "_")
        district_frame_dir = os.path.join(frames_base, safe_name)
        os.makedirs(district_frame_dir, exist_ok=True)

        baseline_mean = None

        for year in years:
            subset_dir = os.path.join(output_dir, "subsets", str(year))
            median_path = os.path.join(subset_dir,
                                       f"maharashtra_median_{year}.tif")
            if not os.path.exists(median_path):
                continue

            try:
                with rasterio.open(median_path) as src:
                    geom = [mapping(row.geometry)]
                    clipped, clipped_transform = rasterio.mask.mask(
                        src, geom, crop=True, filled=True, nodata=np.nan
                    )
                    clipped_data = clipped[0]
                    clipped_bounds = rasterio.transform.array_bounds(
                        clipped_data.shape[0], clipped_data.shape[1],
                        clipped_transform
                    )

                # Dark-reference background subtraction
                bg = backgrounds.get(year, 0.0) if backgrounds else 0.0
                if bg > 0:
                    clipped_data = np.maximum(0, clipped_data - bg)

                clipped_extent = [
                    clipped_bounds[0], clipped_bounds[2],
                    clipped_bounds[1], clipped_bounds[3],
                ]

                fig, ax = plt.subplots(figsize=(10, 9))
                fig.patch.set_facecolor('black')

                # Fill NaN (outside boundary) with 0 before log-scale
                data_filled = np.where(np.isnan(clipped_data),
                                       0.0, clipped_data)
                display_data = np.log10(np.clip(data_filled, 0.01, None))

                # Gaussian smooth + upsample for clean rendering at
                # district zoom level (raw VIIRS ~450m pixels are blocky)
                display_smooth = gaussian_filter(display_data, sigma=2.0)
                display_up = zoom(display_smooth, 3, order=1)

                im = ax.imshow(
                    display_up, extent=clipped_extent, cmap="magma",
                    vmin=-2, vmax=2, origin="upper", aspect="auto",
                )

                # Clip raster to district polygon (hides exterior fill)
                single_gdf = district_gdf[
                    district_gdf["district"] == district_name
                ]
                clip_path = _make_clip_patch(single_gdf, ax)
                im.set_clip_path(clip_path)

                single_gdf.boundary.plot(
                    ax=ax, edgecolor="white", linewidth=1.0, alpha=0.8
                )

                cbar = plt.colorbar(im, ax=ax, shrink=0.6)
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
                cbar.set_label("log\u2081\u2080(Radiance nW/cm\u00b2/sr)",
                               color='white')

                # Stats annotation
                current_mean = float(np.nanmean(clipped_data))
                if baseline_mean is None:
                    baseline_mean = current_mean

                baseline_text = ""
                if baseline_mean > 0 and year != baseline_year:
                    abs_change = current_mean - baseline_mean
                    pct_change = (abs_change / baseline_mean) * 100
                    baseline_text = (
                        f"\nvs {baseline_year}: "
                        f"{abs_change:+.3f} nW ({pct_change:+.1f}%)"
                    )

                text = (
                    f"Year: {year}\n"
                    f"Mean: {current_mean:.3f} nW"
                    f"{baseline_text}"
                )
                ax.text(0.98, 0.02, text, transform=ax.transAxes,
                        fontsize=12, fontweight="bold", color="white",
                        ha="right", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.5", fc="black",
                                  alpha=0.7, ec="white"))

                ax.set_title(
                    f"{district_name}: Nighttime Radiance ({year})",
                    fontsize=14, color='white'
                )
                ax.set_axis_off()
                plt.tight_layout()

                path = os.path.join(district_frame_dir,
                                    f"radiance_{year}.png")
                fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight",
                            facecolor='black')
                plt.close(fig)
                total_count += 1

            except Exception as exc:
                log.warning(
                    "Failed to generate radiance frame for %s/%d: %s",
                    district_name, year, exc
                )
                plt.close("all")
                continue

        log.info("Generated frames for %s", district_name)

    log.info("Generated %d total per-district radiance frames in %s",
             total_count, frames_base)
    return total_count


def generate_per_district_radiance_maps(output_dir, year, district_gdf,
                                        maps_output_dir=None,
                                        backgrounds=None):
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
    backgrounds = backgrounds or {}
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

            # Dark-reference background subtraction
            bg = backgrounds.get(year, 0.0) if backgrounds else 0.0
            if bg > 0:
                clipped_data = np.maximum(0, clipped_data - bg)

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
