"""
Constants for bird migration pathway analysis.

IUCN Red List categories, migration season definitions, neighboring state
mappings, and analysis thresholds for the Maharashtra eBird pipeline.
"""

# ─── IUCN RED LIST CATEGORIES ──────────────────────────────────────────────
# Official codes with display colors for map visualizations.
# Colors follow a red→blue gradient from most to least threatened.
IUCN_CATEGORIES = {
    "CR": {"label": "Critically Endangered", "color": "#d73027", "priority": 1},
    "EN": {"label": "Endangered", "color": "#fc8d59", "priority": 2},
    "VU": {"label": "Vulnerable", "color": "#fee08b", "priority": 3},
    "NT": {"label": "Near Threatened", "color": "#d9ef8b", "priority": 4},
    "LC": {"label": "Least Concern", "color": "#91bfdb", "priority": 5},
    "DD": {"label": "Data Deficient", "color": "#999999", "priority": 6},
    "NE": {"label": "Not Evaluated", "color": "#cccccc", "priority": 7},
}

# Categories considered "threatened" for conservation hotspot analysis.
THREATENED_CODES = {"CR", "EN", "VU"}

# ─── MIGRATION SEASONS ─────────────────────────────────────────────────────
# Central Asian Flyway / Indian subcontinent seasonal definitions.
# Post-nuptial (autumn) migration = Sep-Nov: birds moving south after breeding.
# Pre-nuptial (spring) migration = Dec-May: birds returning north.
# Breeding season = Jun-Aug: peak nesting period in temperate source regions.
MIGRATION_SEASONS = {
    "post_nuptial": {"months": [9, 10, 11], "label": "Post-nuptial (Sep-Nov)"},
    "pre_nuptial": {"months": [12, 1, 2, 3, 4, 5], "label": "Pre-nuptial (Dec-May)"},
    "breeding": {"months": [6, 7, 8], "label": "Breeding (Jun-Aug)"},
}

# ─── NEIGHBORING STATES ────────────────────────────────────────────────────
# States bordering Maharashtra with their eBird data files and compass direction
# relative to Maharashtra. Used for detecting migration entry/exit corridors.
NEIGHBORING_STATES = {
    "gujarat": {"file": "gujarat_modern.tsv.gz", "direction": "NW"},
    "madhya_pradesh": {"file": "madhya_pradesh_modern.tsv.gz", "direction": "N"},
    "chhattisgarh": {"file": "chhattisgarh_modern.tsv.gz", "direction": "NE"},
    "telangana": {"file": "telangana_modern.tsv.gz", "direction": "E"},
    "karnataka": {"file": "karnataka_modern.tsv.gz", "direction": "S"},
    "goa": {"file": "goa_modern.tsv.gz", "direction": "SW"},
    "andhra_pradesh": {"file": "andhra_pradesh_modern.tsv.gz", "direction": "SE"},
}

# ─── SHAPEFILE SOURCES ──────────────────────────────────────────────────────
# From datta07/INDIAN-SHAPEFILES (same repo as Maharashtra district boundaries).
SHAPEFILE_BASE_URL = (
    "https://raw.githubusercontent.com/datta07/INDIAN-SHAPEFILES/master/"
)
INDIA_STATES_GEOJSON_URL = SHAPEFILE_BASE_URL + "INDIA/INDIA_STATES.geojson"

# ─── EBIRD DATA ────────────────────────────────────────────────────────────
EBIRD_DATA_DIR = "data/ebird"
MAHARASHTRA_EBIRD_FILE = "maharashtra_modern.tsv.gz"

# Columns to read from eBird TSV (9 of 50 — keeps memory ~10x lower).
EBIRD_USECOLS = [
    "speciesKey", "species", "scientificName",
    "order", "family",
    "decimalLatitude", "decimalLongitude",
    "month", "year", "individualCount",
]

# Chunk size for reading large TSV.gz files.
EBIRD_CHUNK_SIZE = 100_000

# ─── ANALYSIS THRESHOLDS ───────────────────────────────────────────────────
# Minimum observations per species per month to consider statistically meaningful.
MIN_OBS_PER_MONTH = 5

# Minimum distinct months a species must be observed to include in analysis.
MIN_MONTHS_PRESENT = 3

# If a species is observed in >= this many months, classify as resident.
RESIDENT_MONTHS_THRESHOLD = 10

# Longitude threshold for "Konkan coast" (west Maharashtra).
# Species appearing west of this with no prior eastern-neighbor sightings
# are flagged as potential ocean-origin arrivals.
KONKAN_COAST_LON = 73.5

# ─── GBIF API ──────────────────────────────────────────────────────────────
GBIF_SPECIES_API = "https://api.gbif.org/v1/species"
GBIF_RATE_LIMIT_PER_SEC = 5  # Unauthenticated limit; we use conservative value

# ─── iNATURALIST DATA ─────────────────────────────────────────────────────
INATURALIST_DATA_DIR = "data/inaturalist"
INATURALIST_RAW_CSV = "data/inaturalist/maharashtra_birds.csv"
INATURALIST_MATCHED_CSV = "data/inaturalist/matched_species.csv"

# Column mapping from iNaturalist export to our internal format.
INATURALIST_COLUMNS = {
    "latitude": "lat",
    "longitude": "lon",
    "scientific_name": "scientificName",
    "common_name": "species",
    "taxon_id": "inat_taxon_id",
    "observed_on": "observed_on",
}

# ─── OUTPUT ─────────────────────────────────────────────────────────────────
IUCN_LOOKUP_CSV = "data/ebird/iucn_species_lookup.csv"
SPECIES_PROFILES_CSV = "data/ebird/species_monthly_profiles.csv"
SPECIES_CLASSIFICATION_CSV = "data/ebird/species_classification.csv"
NEIGHBOR_PROFILES_CSV = "data/ebird/neighbor_monthly_profiles.csv"
OBSERVATION_POINTS_CSV = "data/ebird/observation_points.csv.gz"
REGION_SHAPEFILE = "data/shapefiles/maharashtra_region.geojson"
MIGRATION_OUTPUT_DIR = "outputs/ebird"

# Maximum observation points to sample per species per month for KDE heatmaps.
# Keeps the observation_points file manageable (~2-3M rows instead of 6.5M).
MAX_OBS_PER_SPECIES_MONTH = 500
