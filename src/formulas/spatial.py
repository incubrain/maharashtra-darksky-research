"""
Spatial analysis constants.
"""

import numpy as np

# Mean Earth radius for Haversine distance calculation.
# Standard geodetic value (IUGG).
EARTH_RADIUS_KM = 6371.0

# Compass direction definitions for directional brightness analysis.
# Each direction: start/end bearing (degrees clockwise from north)
# and radian angle for polar plotting.
DIRECTION_DEFINITIONS = {
    "north": {"start_angle": 315, "end_angle": 45, "radian": 0},
    "east": {"start_angle": 45, "end_angle": 135, "radian": np.pi / 2},
    "south": {"start_angle": 135, "end_angle": 225, "radian": np.pi},
    "west": {"start_angle": 225, "end_angle": 315, "radian": 3 * np.pi / 2},
}
