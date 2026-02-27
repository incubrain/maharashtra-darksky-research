"""
Quality filtering constants for VIIRS raster processing.
"""

# Valid range for cloud-free coverage observations (days per year).
CF_CVG_VALID_RANGE = (0, 365)

# NOTE (finding E2, review 2026-02-27):
# A previous LIT_MASK_THRESHOLD = 0.5 constant existed here but was dead
# code — the actual lit_mask filter in viirs_process.apply_quality_filters()
# uses ``lit_data > 0`` (binary mask). The unused constant has been removed
# to avoid confusion.
# Ref: Elvidge et al. (2017, 2021) — VIIRS lit_mask is a binary layer
# where 0 = background and any positive value = lit pixel.
