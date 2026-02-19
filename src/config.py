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

# Specific Analysis Thresholds
DARKNESS_THRESHOLD_NW = 0.25 # "Pristine" darkness (nW/cm²/sr)
SPRAWL_THRESHOLD_NW = 1.5   # Threshold for "Lit" urban area in sprawl maps


# Percentile classification bins for graduated ALAN ranking
ALAN_PERCENTILE_BINS = [0, 20, 40, 60, 80, 100]
ALAN_PERCENTILE_LABELS = ["Pristine", "Low", "Medium", "High", "Very High"]

# ─── LOCATION DEFINITIONS ────────────────────────────────────────────────
# 43 most populous cities in Maharashtra (2011 Census) for ALAN reference.
# Coordinates are approximate city-centre points for VIIRS buffer analysis.
URBAN_CITIES = {
    "Mumbai":           {"lat": 18.9750, "lon": 72.8258, "district": "Mumbai"},
    "Pune":             {"lat": 18.5204, "lon": 73.8567, "district": "Pune"},
    "Nagpur":           {"lat": 21.1458, "lon": 79.0882, "district": "Nagpur"},
    "Thane":            {"lat": 19.2183, "lon": 72.9781, "district": "Thane"},
    "Pimpri-Chinchwad": {"lat": 18.6278, "lon": 73.8131, "district": "Pune"},
    "Nashik":           {"lat": 20.0063, "lon": 73.7900, "district": "Nashik"},
    "Kalyan-Dombivli":  {"lat": 19.2437, "lon": 73.1355, "district": "Thane"},
    "Vasai-Virar":      {"lat": 19.3927, "lon": 72.8616, "district": "Palghar"},
    "Aurangabad":       {"lat": 19.8762, "lon": 75.3433, "district": "Aurangabad"},
    "Navi Mumbai":      {"lat": 19.0330, "lon": 73.0297, "district": "Thane"},
    "Solapur":          {"lat": 17.6599, "lon": 75.9064, "district": "Solapur"},
    "Mira-Bhayandar":   {"lat": 19.2952, "lon": 72.8544, "district": "Thane"},
    "Bhiwandi-Nizampur": {"lat": 19.2967, "lon": 73.0631, "district": "Thane"},
    "Amravati":         {"lat": 20.9320, "lon": 77.7523, "district": "Amravati"},
    "Nanded-Waghala":   {"lat": 19.1602, "lon": 77.3150, "district": "Nanded"},
    "Kolhapur":         {"lat": 16.7050, "lon": 74.2433, "district": "Kolhapur"},
    "Ulhasnagar":       {"lat": 19.2167, "lon": 73.1500, "district": "Thane"},
    "Sangli":           {"lat": 16.8524, "lon": 74.5815, "district": "Sangli"},
    "Malegaon":         {"lat": 20.5549, "lon": 74.5346, "district": "Nashik"},
    "Jalgaon":          {"lat": 21.0077, "lon": 75.5626, "district": "Jalgaon"},
    "Akola":            {"lat": 20.7096, "lon": 76.9981, "district": "Akola"},
    "Latur":            {"lat": 18.4088, "lon": 76.5604, "district": "Latur"},
    "Dhule":            {"lat": 20.9042, "lon": 74.7749, "district": "Dhule"},
    "Ahmednagar":       {"lat": 19.0952, "lon": 74.7496, "district": "Ahmadnagar"},
    "Chandrapur":       {"lat": 19.9500, "lon": 79.2961, "district": "Chandrapur"},
    "Parbhani":         {"lat": 19.2578, "lon": 76.7737, "district": "Parbhani"},
    "Ichalkaranji":     {"lat": 16.6912, "lon": 74.4605, "district": "Kolhapur"},
    "Jalna":            {"lat": 19.8410, "lon": 75.8864, "district": "Jalna"},
    "Ambernath":        {"lat": 19.1864, "lon": 73.1919, "district": "Thane"},
    "Bhusawal":         {"lat": 21.0450, "lon": 75.7879, "district": "Jalgaon"},
    "Panvel":           {"lat": 18.9906, "lon": 73.1173, "district": "Raigarh"},
    "Badlapur":         {"lat": 19.1668, "lon": 73.2368, "district": "Thane"},
    "Beed":             {"lat": 18.9892, "lon": 75.7563, "district": "Bid"},
    "Gondia":           {"lat": 21.4653, "lon": 80.1711, "district": "Gondiya"},
    "Satara":           {"lat": 17.6914, "lon": 74.0009, "district": "Satara"},
    "Barshi":           {"lat": 18.2158, "lon": 75.6920, "district": "Solapur"},
    "Yavatmal":         {"lat": 20.3888, "lon": 78.1204, "district": "Yavatmal"},
    "Achalpur":         {"lat": 21.2567, "lon": 77.5101, "district": "Amravati"},
    "Osmanabad":        {"lat": 18.1861, "lon": 76.0419, "district": "Osmanabad"},
    "Nandurbar":        {"lat": 21.3700, "lon": 74.2400, "district": "Nandurbar"},
    "Wardha":           {"lat": 20.7389, "lon": 78.6188, "district": "Wardha"},
    "Udgir":            {"lat": 18.3926, "lon": 77.1161, "district": "Latur"},
    "Hinganghat":       {"lat": 20.5490, "lon": 78.8360, "district": "Wardha"},
}
# Backwards-compatible alias
URBAN_BENCHMARKS = URBAN_CITIES

# Dark-sky candidate sites: protected areas, reserves, remote villages
DARKSKY_SITES = {
    "Lonar Crater":             {"lat": 19.9761, "lon": 76.5079, "district": "Buldana",
                                 "type": "crater"},
    "Tadoba Tiger Reserve":     {"lat": 20.2485, "lon": 79.4254, "district": "Chandrapur",
                                 "type": "tiger_reserve"},
    "Pench Tiger Reserve":      {"lat": 21.6900, "lon": 79.2300, "district": "Nagpur",
                                 "type": "tiger_reserve"},
    "Udmal Tribal Village":     {"lat": 20.6567, "lon": 73.4856, "district": "Nashik",
                                 "type": "tribal_village"},
    "Kaas Plateau":             {"lat": 17.7200, "lon": 73.8228, "district": "Satara",
                                 "type": "plateau"},
    "Toranmal":                 {"lat": 21.7333, "lon": 74.4167, "district": "Nandurbar",
                                 "type": "hill_station"},
    "Bhandardara":              {"lat": 19.5375, "lon": 73.7695, "district": "Ahmadnagar",
                                 "type": "reservoir"},
    "Harihareshwar":            {"lat": 17.9942, "lon": 73.0258, "district": "Raigarh",
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
    "Mumbai", "Pune", "Nagpur", "Gadchiroli", "Nandurbar", "Sindhudurg",
]

# ─── OUTPUT PATHS ─────────────────────────────────────────────────────────
import os

OUTPUT_DIRS = {
    "csv": "csv",
    "maps": "maps",
    "subsets": "subsets",
    "district_reports": "district_reports",
    "site_reports": "site_reports",
    "diagnostics": "diagnostics",
}


def get_entity_dirs(base_dir, entity_type):
    """Get output directories for an entity type.

    Parameters
    ----------
    base_dir : str
        Run-level output directory.
    entity_type : str
        "district", "city", or "site".

    Returns
    -------
    dict
        Keys: csv, maps, reports, diagnostics. Values: full paths.
    """
    entity_dir = os.path.join(base_dir, entity_type)
    return {
        "csv": os.path.join(entity_dir, "csv"),
        "maps": os.path.join(entity_dir, "maps"),
        "reports": os.path.join(entity_dir, "reports"),
        "diagnostics": os.path.join(entity_dir, "diagnostics"),
    }

# ─── BOUNDARY SOURCE ─────────────────────────────────────────────────────
# Source: datta07/INDIAN-SHAPEFILES (modern district spellings, EPSG:4326)
SHAPEFILE_URL = (
    "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/"
    "STATES/MAHARASHTRA/MAHARASHTRA_DISTRICTS.geojson"
)
SHAPEFILE_FORMAT = "geojson"

# Column name mapping: source GeoJSON property → standardised pipeline name.
# The pipeline always expects a "district" column; this mapping ensures any
# upstream source can be adapted without changing downstream code.
SHAPEFILE_COLUMN_MAP = {
    "dtname": "district",
    "stname": "state",
    "dtcode11": "district_code",
    "stcode11": "state_code",
    "Dist_LGD": "district_lgd",
    "State_LGD": "state_lgd",
}

EXPECTED_DISTRICT_COUNT = 36

# ─── DEFAULT PATHS ───────────────────────────────────────────────────────
DEFAULT_SHAPEFILE_PATH = "./data/shapefiles/maharashtra_district.geojson"
DEFAULT_VIIRS_DIR = "./viirs"
DEFAULT_OUTPUT_DIR = "./outputs"

# ─── EXTERNAL DATASETS ────────────────────────────────────────────────
# Toggle datasets for cross-comparison with VNL radiance data.
# Each entry maps to a module in src/datasets/.
# Set "enabled": True to include in pipeline runs.

EXTERNAL_DATASETS = {
    "census_2011": {
        "enabled": False,  # Off by default — opt-in via --datasets
        "data_dir": "data/census",
        "description": "Census of India 2011 PCA (total population)",
    },
    "census_2001": {
        "enabled": False,
        "data_dir": "data/census",
        "description": "Census of India 2001 PCA (total population)",
    },
    "census_1991": {
        "enabled": False,
        "data_dir": "data/census",
        "description": "Census of India 1991 PCA (total population, urban+rural)",
    },
    "census_2011_pca": {
        "enabled": False,
        "data_dir": "data/census",
        "description": "Census of India 2011 PCA (legacy adapter)",
    },
    "census_projected": {
        "enabled": False,
        "data_dir": "data/census",
        "description": "Census-based linear projections for VIIRS years (2012-2024)",
    },
    # Town-level datasets
    "census_2011_towns": {
        "enabled": False,
        "data_dir": "data/census",
        "description": "Census of India 2011 PCA — town level (537 towns)",
    },
    "census_2001_towns": {
        "enabled": False,
        "data_dir": "data/census",
        "description": "Census of India 2001 PCA — town level (379 towns)",
    },
    "census_1991_towns": {
        "enabled": False,
        "data_dir": "data/census",
        "description": "Census of India 1991 PCA — town level (336 towns)",
    },
    "census_towns_projected": {
        "enabled": False,
        "data_dir": "data/census",
        "description": "Census-based town-level linear projections (2012-2024)",
    },
}

# ── Census common schema ────────────────────────────────────────────
# Columns present in all three censuses (1991, 2001, 2011).
# All years use Total (Urban + Rural) population data.
# Pre-extracted into data/census/census_{year}_pca.csv by
# scripts/extract_census_csvs.py.
CENSUS_COMMON_COLUMNS = [
    "No_HH",       # Households
    "TOT_P",       # Total population
    "TOT_M",       # Male population
    "TOT_F",       # Female population
    "P_06",        # Population 0-6
    "P_SC",        # Scheduled Castes
    "P_ST",        # Scheduled Tribes
    "P_LIT",       # Literate population
    "TOT_WORK_P",  # Total workers
    "MAINWORK_P",  # Main workers
    "MARGWORK_P",  # Marginal workers
    "NON_WORK_P",  # Non-workers
]

# Derived ratios computable from the common columns
CENSUS_COMMON_DERIVED_RATIOS = {
    "literacy_rate":    ("P_LIT", "TOT_P"),
    "workforce_rate":   ("TOT_WORK_P", "TOT_P"),
    "dependency_ratio": ("NON_WORK_P", "TOT_WORK_P"),
    "child_ratio":      ("P_06", "TOT_P"),
    "sc_st_share":      ("P_SC + P_ST", "TOT_P"),
    "household_size":   ("TOT_P", "No_HH"),
    "sex_ratio":        ("TOT_F", "TOT_M"),
}

# Legacy alias kept for backward compatibility
CENSUS_2011_DISTRICT_COLUMNS = CENSUS_COMMON_COLUMNS
CENSUS_2011_DERIVED_RATIOS = CENSUS_COMMON_DERIVED_RATIOS

# VNL metrics to correlate against external datasets
VNL_CORRELATION_METRICS = [
    "median_radiance",
    "mean_radiance",
    "annual_pct_change",
    "r_squared",
]
