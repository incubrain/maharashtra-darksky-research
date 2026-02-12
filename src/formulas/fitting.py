"""
Light dome exponential decay model fitting parameters.
"""

import numpy as np

# scipy.optimize.curve_fit bounds for exponential decay model:
#   radiance(d) = peak * exp(-decay_rate * d) + background
# Parameters: [peak, decay_rate, background]
EXP_DECAY_BOUNDS = ([0, 0, 0], [np.inf, 10, np.inf])

# Maximum function evaluations for curve_fit.
EXP_DECAY_MAXFEV = 5000

# Background radiance threshold (nW/cm²/sr) defining the light dome edge.
LIGHT_DOME_BACKGROUND_THRESHOLD = 0.5
