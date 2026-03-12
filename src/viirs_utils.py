import logging
import numpy as np

log = logging.getLogger(__name__)

def apply_dynamic_background_subtraction(raster, year=None, percentile=1.0):
    """
    Per-year P01 background subtraction (legacy — used only in diagnostics).

    Computes the 1st-percentile of valid (finite & > 0) pixels in *raster*
    and subtracts it uniformly.  This was the original background correction
    but has a **known limitation**: the P01 floor rises over the study
    period (0.10 nW in 2012 → 0.49 nW in 2024) because later NOAA EOG
    composites retain more dim pixels, not because of real radiance change.
    Subtracting a larger floor from later years systematically compresses
    the observed growth trend.

    This function is retained for single-year diagnostic plots (gradient
    analysis, quality diagnostics, ecological overlay) where cross-year
    consistency is not required.  Time-series visualization frames use
    raw VNL V2.2 radiance without additional background subtraction,
    since the annual composite product already handles background zeroing.

    Args:
        raster (np.ndarray): The raw radiance values.
        year (int, optional): The year being processed (for logging).
        percentile (float): The percentile to use as the floor (default 1.0).

    Returns:
        np.ndarray: The baseline-corrected raster (all values >= 0).
    """
    # Create mask for valid background pixels
    # We exclude 0 and infinites to find the true "noise floor" of the data areas
    bg_mask = np.isfinite(raster) & (raster > 0)
    
    if bg_mask.sum() > 0:
        floor = np.percentile(raster[bg_mask], percentile)
        year_str = f"Year {year} " if year else ""
        log.debug("%sBackground Floor (P%.1f): %.4f nW/cm²/sr",
                  year_str, percentile, floor)
        
        # Apply correction and clip at zero
        corrected = np.maximum(0, raster - floor)
        return corrected
    else:
        if year:
            log.warning("Year %d: No valid data for DBS calculation.", year)
        return np.maximum(0, raster)
