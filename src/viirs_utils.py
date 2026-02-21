import logging
import numpy as np

log = logging.getLogger(__name__)

def apply_dynamic_background_subtraction(raster, year=None, percentile=1.0):
    """
    Applies Dynamic Background Subtraction (DBS) to a radiance raster.

    Identifies the noise floor using a low percentile (default 1%) of valid
    (finite and >0) pixels and subtracts it from the entire array.

    KNOWN BEHAVIOR — Rising Background Floor:
        The P1.0 background floor increases over the study period
        (e.g., 0.10 nW in 2012 → 0.49 nW in 2024 for Maharashtra).
        This is NOT a pipeline bug but a characteristic of the VIIRS VNL
        annual composite product, caused by:

        1. NOAA EOG background masking evolution — later composites
           retain more dim/background pixels that were previously zeroed.
        2. Stray-light correction changes: vcmcfg (2012-2013) vs
           vcmslcfg (2014+) have different noise characteristics.
        3. NPP→NOAA-20 satellite transition (2018+) introduces
           cross-sensor calibration offsets.

        The DBS function is used ONLY in visualization/quality diagnostics
        (gradient analysis, annual map frames), not in the main zonal
        statistics path where raw radiance is preserved.

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
        log.info("%sBackground Floor (P%.1f): %.4f nW/cm²/sr", 
                 year_str, percentile, floor)
        
        # Apply correction and clip at zero
        corrected = np.maximum(0, raster - floor)
        return corrected
    else:
        if year:
            log.warning("Year %d: No valid data for DBS calculation.", year)
        return np.maximum(0, raster)
