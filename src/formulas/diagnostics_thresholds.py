"""
Diagnostic and stability threshold constants.

Citations:
- Durbin, J. & Watson, G.S. (1951). Testing for serial correlation in
  least squares regression. Biometrika, 38(1/2), 159-177.
  For n=13, k=1: dL=1.010, dU=1.340 at 5%. Bounds [1.0, 3.0] are
  conservative (symmetric around 2.0).
- Jarque, C.M. & Bera, A.K. (1987). A test for normality.
  Standard alpha = 0.05.
- Cook, R.D. (1977). Detection of influential observations in linear
  regression. Technometrics, 19(1), 15-18.
  Original guideline: D > 1.0 suggests high influence.
"""

# Standardized residual threshold for outlier detection.
OUTLIER_Z_THRESHOLD = 2.0

# Durbin-Watson statistic bounds for autocorrelation detection.
# DW < low → positive autocorrelation; DW > high → negative autocorrelation.
DW_WARNING_LOW = 1.0
DW_WARNING_HIGH = 3.0

# Jarque-Bera test p-value threshold for residual normality.
JB_P_THRESHOLD = 0.05

# Cook's distance threshold for influential observations.
COOKS_D_THRESHOLD = 1.0

# R-squared threshold below which model fit is considered poor.
R_SQUARED_WARNING = 0.5

# Coefficient of variation thresholds for temporal stability classification.
# CV < stable → "stable"; CV < erratic → "moderate"; CV >= erratic → "erratic".
CV_STABLE_THRESHOLD = 0.2
CV_ERRATIC_THRESHOLD = 0.5
