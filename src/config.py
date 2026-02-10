"""
Centralized configuration for Maharashtra VIIRS ALAN Analysis.

All analysis parameters, thresholds, location definitions, and paths are
defined here with inline scientific citations justifying each choice.
"""

# ─── QUALITY FILTERING PARAMETERS ────────────────────────────────────────
# Following Elvidge et al. (2017) VIIRS preprocessing guidelines:
# "Pixels with fewer cloud-free observations are more susceptible to
# ephemeral light contamination and temporal noise."
# Citation: Elvidge, C.D. et al. (2017). VIIRS night-time lights.
#           Int. J. Remote Sensing, 38(21), 5860-5879.
CF_COVERAGE_THRESHOLD = 5  # Minimum cloud-free observations per year

# Following Elvidge et al. (2021) annual composite methodology:
# "The lit_mask excludes background pixels (water, uninhabited desert)
# that may contain sensor noise or gas flares."
# Citation: Elvidge, C.D. et al. (2021). Annual time series of global
#           VIIRS nighttime lights. Remote Sensing, 13(5), 922.
USE_LIT_MASK = True   # Exclude background/non-lit pixels
USE_CF_FILTER = True  # Apply cloud-free coverage threshold

# ─── TEMPORAL PARAMETERS ─────────────────────────────────────────────────
STUDY_YEARS = range(2012, 2025)  # 2012-2024 inclusive (VIIRS era)

# ─── TREND MODELING PARAMETERS ───────────────────────────────────────────
# Log-linear model: log(radiance + epsilon) ~ year
# Small constant prevents log(0); value chosen to be negligible relative
# to minimum VIIRS detectable radiance (~0.1 nW/cm²/sr).
LOG_EPSILON = 1e-6
BOOTSTRAP_RESAMPLES = 1000  # Number of bootstrap iterations for CI
BOOTSTRAP_CI_LEVEL = (2.5, 97.5)  # 95% confidence interval percentiles
BOOTSTRAP_SEED = 42  # Random seed for reproducibility
MIN_YEARS_FOR_TREND = 2  # Minimum data points for district trend
MIN_YEARS_FOR_SITE_TREND = 3  # Minimum data points for site trend

# ─── SPATIAL ANALYSIS PARAMETERS ─────────────────────────────────────────
# Following Wang et al. (2022) protected area buffer methodology:
# "A 10 km buffer around protected area boundaries captures the
# transition zone where urban light spillover attenuates."
# Citation: Wang, J. et al. (2022). Protected area buffer analysis.
SITE_BUFFER_RADIUS_KM = 10   # Buffer around point sites
PROTECTED_AREA_BUFFER_KM = 10  # Buffer outside protected boundaries

# Following Zheng et al. (2019) anisotropic ALAN investigation:
# "Radial extraction at 1, 5, 10, 20, 50 km from city centers
# characterises the exponential decay of urban light domes."
# Citation: Zheng, Q. et al. (2019). Developing a new cross-sensor
#           calibration model. Remote Sensing, 11(18), 2132.
URBAN_GRADIENT_RADII_KM = [1, 5, 10, 20, 50]

# Approximate Maharashtra bounding box (with buffer for subsetting)
MAHARASHTRA_BBOX = {
    "west": 72.5,
    "south": 15.5,
    "east": 81.0,
    "north": 22.1,
}

# UTM Zone for Maharashtra metric operations
MAHARASHTRA_UTM_EPSG = 32643  # UTM Zone 43N

# ─── ALAN CLASSIFICATION THRESHOLDS (nW/cm²/sr) ─────────────────────────
# Based on dark-sky research consensus and IDA (International Dark-Sky
# Association) guidelines for site certification.
# Low: suitable for astronomical observation and dark-sky designation.
# Medium: moderate artificial illumination, visible light pollution.
# High: significant urban/industrial illumination.
ALAN_LOW_THRESHOLD = 1.0   # Below = pristine/low
ALAN_MEDIUM_THRESHOLD = 5.0  # 1-5 = moderate, >5 = high

# Percentile classification bins for graduated ALAN ranking
ALAN_PERCENTILE_BINS = [0, 20, 40, 60, 80, 100]
ALAN_PERCENTILE_LABELS = ["Pristine", "Low", "Medium", "High", "Very High"]

# ─── LOCATION DEFINITIONS ────────────────────────────────────────────────
# Urban benchmarks: major Maharashtra cities for ALAN reference
URBAN_BENCHMARKS = {
    "Mumbai":           {"lat": 18.9600, "lon": 72.8200, "district": "Mumbai City"},
    "Pune":             {"lat": 18.5167, "lon": 73.8554, "district": "Pune"},
    "Nagpur":           {"lat": 21.1466, "lon": 79.0889, "district": "Nagpur"},
    "Thane":            {"lat": 19.2183, "lon": 72.9781, "district": "Thane"},
    "Pimpri-Chinchwad": {"lat": 18.6278, "lon": 73.8131, "district": "Pune"},
}

# Dark-sky candidate sites: protected areas, reserves, remote villages
DARKSKY_SITES = {
    "Lonar Crater":             {"lat": 19.9761, "lon": 76.5079, "district": "Buldhana",
                                 "type": "crater"},
    "Tadoba Tiger Reserve":     {"lat": 20.2485, "lon": 79.4254, "district": "Chandrapur",
                                 "type": "tiger_reserve"},
    "Pench Tiger Reserve":      {"lat": 21.6900, "lon": 79.2300, "district": "Nagpur",
                                 "type": "tiger_reserve"},
    "Udmal Tribal Village":     {"lat": 20.6587, "lon": 73.4836, "district": "Nashik",
                                 "type": "tribal_village"},
    "Kaas Plateau":             {"lat": 17.7200, "lon": 73.8228, "district": "Satara",
                                 "type": "plateau"},
    "Toranmal":                 {"lat": 21.7333, "lon": 74.4167, "district": "Nandurbar",
                                 "type": "hill_station"},
    "Bhandardara":              {"lat": 19.5375, "lon": 73.7695, "district": "Ahmednagar",
                                 "type": "reservoir"},
    "Harihareshwar":            {"lat": 17.9942, "lon": 73.0258, "district": "Raigad",
                                 "type": "coastal"},
    "Yawal Wildlife Sanctuary": {"lat": 21.3781, "lon": 75.8750, "district": "Jalgaon",
                                 "type": "wildlife_sanctuary"},
    "Melghat Tiger Reserve":    {"lat": 21.4458, "lon": 77.1972, "district": "Amravati",
                                 "type": "tiger_reserve"},
    "Bhimashankar":             {"lat": 19.0739, "lon": 73.5352, "district": "Pune",
                                 "type": "wildlife_sanctuary"},
}

# ─── TEST DATA GENERATION ────────────────────────────────────────────────
# Cities used for synthetic test data (lon, lat, brightness nW)
TEST_DATA_CITIES = {
    "Mumbai":      (72.88, 19.08, 50.0),
    "Pune":        (73.86, 18.52, 25.0),
    "Nagpur":      (79.09, 21.15, 15.0),
    "Nashik":      (73.79, 20.00,  8.0),
    "Aurangabad":  (75.34, 19.88,  7.0),
    "Solapur":     (75.92, 17.68,  6.0),
    "Kolhapur":    (74.24, 16.70,  5.0),
    "Thane":       (72.98, 19.20, 30.0),
    "Navi Mumbai": (73.02, 19.03, 20.0),
}

# Synthetic data parameters
TEST_GROWTH_RATE = 0.08      # 8% annual growth from 2012 baseline
TEST_GAUSSIAN_SIGMA = 0.15   # City light Gaussian falloff (degrees)
TEST_NOISE_STD = 0.1         # Year-to-year noise standard deviation

# ─── VIIRS PRODUCT METADATA ─────────────────────────────────────────────
VIIRS_LAYERS = {
    "median": "median_masked",
    "average": "average_masked",
    "cf_cvg": "cf_cvg",
    "lit_mask": "lit_mask",
}

# VIIRS layer name patterns in NOAA EOG filenames
LAYER_PATTERNS = {
    "average": "avg_rade9h",
    "median": "median_masked",
    "cf_cvg": "cf_cvg",
    "lit_mask": "lit_mask",
    "average_masked": "average_masked",
}

# VIIRS version by year (NOAA changed product versions)
VIIRS_VERSION_MAPPING = {
    2012: "v21", 2013: "v21",
    2014: "v22", 2015: "v22", 2016: "v22", 2017: "v22",
    2018: "v22", 2019: "v22", 2020: "v22", 2021: "v22",
    2022: "v22", 2023: "v22", 2024: "v22",
}

VIIRS_RESOLUTION_DEG = 0.004166667  # ~15 arc-seconds

# ─── VISUALIZATION PARAMETERS ────────────────────────────────────────────
MAP_DPI = 300
TIMESERIES_HIGHLIGHT_DISTRICTS = [
    "Mumbai", "Pune", "Nagpur", "Garhchiroli", "Nandurbar", "Sindhudurg",
]

# ─── OUTPUT PATHS ─────────────────────────────────────────────────────────
OUTPUT_DIRS = {
    "csv": "csv",
    "maps": "maps",
    "subsets": "subsets",
    "district_reports": "district_reports",
    "site_reports": "site_reports",
    "diagnostics": "diagnostics",
}

# ─── SHAPEFILE SOURCE ─────────────────────────────────────────────────────
SHAPEFILE_URL = (
    "https://github.com/HindustanTimesLabs/shapefiles/raw/master/"
    "state_ut/maharashtra/district/maharashtra_district.zip"
)

EXPECTED_DISTRICT_COUNT = 36
