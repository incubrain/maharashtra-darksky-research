"""
Quality filtering constants for VIIRS raster processing.
"""

# Lit mask threshold: pixels with radiance below this in the lit_mask
# layer are classified as background (water, desert, uninhabited).
LIT_MASK_THRESHOLD = 0.5

# Valid range for cloud-free coverage observations (days per year).
CF_CVG_VALID_RANGE = (0, 365)
