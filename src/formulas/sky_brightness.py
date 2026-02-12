"""
Sky brightness constants for VIIRS radiance → mag/arcsec² conversion.

Citation: Falchi, F. et al. (2016). The new world atlas of artificial
night sky brightness. Science Advances, 2(6), e1600377.
"""

import numpy as np

# Natural sky background brightness in mag/arcsec².
# Falchi et al. (2016): "The natural sky background at zenith in a
# moonless, cloudless night is approximately 21.6 mag/arcsec²."
NATURAL_SKY_BRIGHTNESS = 21.6

# Empirical conversion factor: nW/cm²/sr → mcd/m².
# Falchi et al. (2016), Table S1.
RADIANCE_TO_MCD = 0.177

# Reference luminance for magnitude zero-point in mcd/m².
# 108,000 cd/m² × 1000 mcd/cd = 108,000,000 mcd/m².
REFERENCE_MCD = 108_000_000

# Bortle scale thresholds: class → (mag_min, mag_max, description).
# Bortle, J.E. (2001). Introducing the Bortle Dark-Sky Scale.
# Updated boundary values from Crumey (2014).
BORTLE_THRESHOLDS = {
    1: (21.75, np.inf, "Excellent dark-sky site"),
    2: (21.50, 21.75, "Typical dark-sky site"),
    3: (21.25, 21.50, "Rural sky"),
    4: (20.50, 21.25, "Rural/suburban transition"),
    5: (19.50, 20.50, "Suburban sky"),
    6: (18.50, 19.50, "Bright suburban sky"),
    7: (18.00, 18.50, "Suburban/urban transition"),
    8: (17.00, 18.00, "City sky"),
    9: (0.00, 17.00, "Inner-city sky"),
}
