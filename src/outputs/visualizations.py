import os
import logging
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
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

def generate_sprawl_frames(years, output_dir, district_gdf, threshold_nw=config.SPRAWL_THRESHOLD_NW):
    """
    Generate frames showing the 'Sprawl' (binary lit vs unlit).
    threshold_nw: Radiance threshold to consider 'lit' (default 1.5 nW).
    """
    frame_dir = os.path.join(output_dir, "maps", "frames", "sprawl")
    os.makedirs(frame_dir, exist_ok=True)
    log.info("Generating Sprawl frames in %s...", frame_dir)

    for year in years:
        data, extent = _load_raster(output_dir, year)
        if data is None: continue

        # Binary mask
        lit_mask = (data >= threshold_nw)
        # Calculate area stats (approximate, assuming ~500m pixels depending on projection, 
        # but pure pixel count is fine for relative story or we use metadata if strictly needed.
        # VNP46A4 is ~15 arc-seconds ~460m. 
        # Let's just report pixel count or % of state for now to be safe/simple.)
        total_pixels = np.sum(np.isfinite(data))
        lit_pixels = np.sum(lit_mask)
        lit_pct = (lit_pixels / total_pixels * 100) if total_pixels > 0 else 0

        fig, ax = _setup_plot(district_gdf)
        
        # Plot: 0 is black/transparent, 1 is Yellow
        # We allow the district boundary to show the shape
        # imshow with transparency for unlit
        
        # Create an RGBA image
        rows, cols = data.shape
        img = np.zeros((rows, cols, 4))
        img[lit_mask] = [1, 0.9, 0.2, 1] # Yellow for lit
        # Unlit stays transparent [0,0,0,0]
        
        ax.imshow(img, extent=extent, origin="upper", zorder=1)
        ax.set_title(f"Urban Sprawl (Lit Area > {threshold_nw} nW)", fontsize=16, color='black') # title might need contrasting color depending on bg. 
        # Actually standard mpl bg is white, but these frames usually look good with dark theme?
        # Let's stick to standard map style but maybe dark background for impact?
        # User liked previous frames which were black bg.
        
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

def generate_differential_frames(years, output_dir, district_gdf):
    """
    Generate frames showing (Current Year - Baseline 2012).
    """
    frame_dir = os.path.join(output_dir, "maps", "frames", "differential")
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
        
        # Plot difference. 
        # vmin/vmax symmetric to show decrease too
        # Use simple limits like -20 to +20 nW or dynamic?
        # Light pollution can grow a lot. Let's try log scale difference? 
        # Or simple linear with SymLogNorm.
        # For visuals, linear clipped usually looks like "fire".
        
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

def generate_darkness_frames(years, output_dir, district_gdf, threshold_nw=config.DARKNESS_THRESHOLD_NW):
    """
    Generate frames showing 'Erosion of Darkness' (pixels < threshold).
    threshold_nw: Dark sky threshold (default 0.25 nW - 'Pristine').
    """
    frame_dir = os.path.join(output_dir, "maps", "frames", "darkness")
    os.makedirs(frame_dir, exist_ok=True)
    log.info("Generating Darkness frames in %s...", frame_dir)

    for year in years:
        data, extent = _load_raster(output_dir, year)
        if data is None: continue

        # Dark mask
        dark_mask = (data < threshold_nw) & (data > 0) # Exclude nodata(0 or -999) if any, assuming >0 is valid data
        
        total_pixels = np.sum(np.isfinite(data))
        dark_pixels = np.sum(dark_mask)
        dark_pct = (dark_pixels / total_pixels * 100) if total_pixels > 0 else 0

        fig, ax = _setup_plot(district_gdf)
        fig.patch.set_facecolor('black')
        
        # Color dark areas Green/Blue, rest is black (transparent) or dark grey
        img = np.zeros((data.shape[0], data.shape[1], 4))
        # Dark areas = Greenish Blue
        img[dark_mask] = [0.0, 0.8, 0.6, 1.0] 
        # Lit areas = slightly visible grey to show context? Or just black?
        # Let's make lit areas dark grey
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

def generate_trend_map(years, output_dir, district_gdf):
    """
    Generate pixel-wise linear trend map (slope of radiance over years).
    """
    maps_dir = os.path.join(output_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)
    log.info("Generating Trend Map...")

    # Load all years into a stack
    stack = []
    valid_years = []
    
    # Needs valid extent from first frame
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
    
    # Calculate slope per pixel
    # Vectorized regression is memory intensive for large images.
    # Simple approach: polyfit along axis 0
    x = np.array(valid_years)
    x = x - x[0] # Time delta
    
    # Mask nan
    # We'll compute slope only where we have data
    # (Ignoring complex nan handling for speed, assuming stacks are consistent)
    
    # numpy polyfit is fast enough for standard images (e.g. 2k x 2k)
    # slope is index 0 of result
    log.info("Computing regression slope for %s pixels...", stack_arr.shape)
    
    # Reshape to (T, N)
    T, H, W = stack_arr.shape
    reshaped = stack_arr.reshape(T, -1)
    
    # Check for NaNs
    # Just fill NaNs with 0 for trend calculation or handle properly?
    # Better to mask. For map, we want simple visual.
    
    # Fast vectorized slope:
    # slope = cov(x, y) / var(x)
    x_mean = np.mean(x)
    y_mean = np.mean(reshaped, axis=0)
    numerator = np.sum((x[:, None] - x_mean) * (reshaped - y_mean), axis=0)
    denominator = np.sum((x[:, None] - x_mean)**2, axis=0)
    slope = numerator / denominator
    slope_map = slope.reshape(H, W)
    
    fig, ax = _setup_plot(district_gdf)
    fig.patch.set_facecolor('black')
    
    # Plot slope
    # Positive (Red) = Growing
    # Negative (Blue) = Shrinking
    
    im = ax.imshow(slope_map, extent=extent, cmap="coolwarm", vmin=-2, vmax=2,
                   origin="upper", aspect="auto")
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, label="Rate of Change (nW/year)")
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    cbar.set_label("Trend Slope (nW/year)", color='white')
    
    ax.set_title(f"Light Pollution Growth Trend ({years[0]}-{years[-1]})", fontsize=16, color='white')
    
    path = os.path.join(maps_dir, "alan_trend_map.png")
    fig.savefig(path, dpi=config.MAP_DPI, bbox_inches="tight", facecolor='black')
    plt.close(fig)
    log.info("Saved Trend Map: %s", path)
