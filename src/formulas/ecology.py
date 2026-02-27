"""
Ecological overlay constants for land cover cross-tabulation.
"""

# Land cover class codes → human-readable names.
LAND_COVER_CLASSES = {
    1: "Forest",
    2: "Shrubland",
    3: "Grassland",
    4: "Cropland",
    5: "Urban/Built-up",
    6: "Water",
    7: "Wetland",
    8: "Barren",
}

# Ecological sensitivity weights per land cover type (0-1 scale).
# Higher values indicate greater sensitivity to artificial light.
#
# PROVENANCE (finding B2, review 2026-02-27):
# These weights are project-specific heuristic estimates, NOT derived from
# a published empirical study. They encode a qualitative ordering based on
# the ecological ALAN literature consensus:
#   - Forest/wetland ecosystems are most disrupted by ALAN (melatonin
#     suppression, foraging disruption, reproductive timing shifts).
#     Ref: Bennie, J. et al. (2014). Scientific Reports, 4, 3789.
#     Ref: Gaston, K.J. et al. (2013). Biological Reviews, 88(4), 912-927.
#   - Urban/barren habitats are least affected (already adapted).
# The absolute values (0.1-0.9) are ordinal placeholders. Impact scores
# using these weights should be interpreted as RELATIVE rankings, not
# calibrated physical quantities.
ECOLOGICAL_SENSITIVITY = {
    "Forest": 0.9,
    "Shrubland": 0.7,
    "Grassland": 0.6,
    "Cropland": 0.4,
    "Urban/Built-up": 0.1,
    "Water": 0.5,
    "Wetland": 0.8,
    "Barren": 0.2,
}
