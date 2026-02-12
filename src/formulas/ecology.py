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
