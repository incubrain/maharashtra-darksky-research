# AGENTS.md — Developer & Agent Reference

This file contains technical reference information for agents and developers working on the Maharashtra VIIRS pipeline. For project overview, methods, and user-facing documentation see [README.md](README.md).

## Repository Structure

```
maharashtra-viirs/
├── README.md
├── AGENTS.md                          # This file
├── requirements.txt
├── data_manifest.json                 # VIIRS product metadata & processing history
├── .gitignore
├── src/
│   ├── config.py                      # Centralized configuration (all parameters)
│   │
│   │   # Shared formulas and constants
│   ├── formulas/
│   │   ├── __init__.py                # Re-exports all public names
│   │   ├── classification.py          # classify_alan(), classify_alan_series(), classify_stability()
│   │   ├── trend.py                   # fit_log_linear_trend() — shared implementation
│   │   ├── sky_brightness.py          # NATURAL_SKY_BRIGHTNESS, BORTLE_THRESHOLDS
│   │   ├── spatial.py                 # EARTH_RADIUS_KM, DIRECTION_DEFINITIONS
│   │   ├── ecology.py                 # LAND_COVER_CLASSES, ECOLOGICAL_SENSITIVITY
│   │   ├── benchmarks.py             # PUBLISHED_BENCHMARKS (Elvidge, Li)
│   │   ├── diagnostics_thresholds.py # DW, JB, Cook's D, R², CV thresholds
│   │   ├── fitting.py                # EXP_DECAY_BOUNDS, LIGHT_DOME params
│   │   └── quality.py                # LIT_MASK_THRESHOLD, CF_CVG_VALID_RANGE
│   │
│   │   # Pipeline infrastructure
│   ├── pipeline_types.py              # StepResult, YearProcessingResult, PipelineRunResult
│   ├── pipeline_steps.py              # 15 district pipeline step functions
│   ├── site_pipeline_steps.py         # 12 site/city pipeline step functions
│   ├── pipeline_runner.py             # Orchestrator with validation gates
│   ├── logging_config.py              # Centralized JSON Lines + console logging
│   ├── preprocess.py                  # Independent VIIRS data extraction
│   ├── generate_maps.py               # Independent map generation from saved CSVs
│   ├── generate_reports.py            # Independent PDF report generation
│   │
│   │   # Core pipelines
│   ├── viirs_process.py               # District-level pipeline (thin orchestrator)
│   ├── site_analysis.py               # Site-level pipeline (--type city|site|all)
│   ├── download_viirs.py              # Data download / synthetic test data generator
│   ├── validate_names.py              # District name cross-validation
│   │
│   │   # Spatial analysis
│   ├── gradient_analysis.py           # Urban-rural radial gradient profiles
│   ├── buffer_comparison.py           # Inside vs outside buffer ALAN comparison
│   ├── directional_analysis.py        # N/S/E/W quadrant brightness analysis
│   ├── proximity_analysis.py          # Nearest city distance metrics (Haversine)
│   │
│   │   # Temporal analysis
│   ├── stability_metrics.py           # CV, IQR, max year-to-year change
│   ├── breakpoint_analysis.py         # AIC-based piecewise linear regression
│   ├── trend_diagnostics.py           # Durbin-Watson, Jarque-Bera, Cook's distance
│   │
│   │   # Validation & quality
│   ├── quality_diagnostics.py         # Per-district pixel filter breakdown
│   ├── benchmark_comparison.py        # Maharashtra vs global/regional growth rates
│   ├── sensitivity_analysis.py        # CF threshold parameter sweeps
│   │
│   │   # Reporting
│   ├── district_reports.py            # Consolidated district atlas PDF
│   ├── site_reports.py                # Consolidated site atlas PDF
│   ├── visualization_suite.py         # Publication-quality maps and charts
│   ├── light_dome_modeling.py         # Exponential decay light dome fitting
│   │
│   │   # Optimization
│   ├── cache_manager.py               # SHA-256 keyed result caching
│   ├── parallel_processing.py         # Multiprocessing zonal stats
│   ├── incremental_update.py          # Smart output regeneration
│   │
│   │   # Enhancements
│   ├── sky_brightness_model.py        # Radiance to mag/arcsec² + Bortle scale
│   ├── graduated_classification.py    # Percentile-based ALAN tier system
│   └── ecological_overlay.py          # Land cover cross-tabulation
│
├── scripts/
│   ├── run_all.sh                     # Full pipeline end-to-end
│   ├── run_preprocess.sh              # Preprocessing only
│   ├── run_district.sh                # District analysis + maps + reports
│   ├── run_city.sh                    # City analysis + maps + reports
│   ├── run_site.sh                    # Site analysis + maps + reports
│   └── run_sensitivity.py             # Standalone sensitivity analysis runner
│
├── tests/
│   ├── conftest.py                    # Shared fixtures, synthetic raster helpers
│   ├── test_formulas.py               # Formulas subpackage (67 tests)
│   ├── test_research_validation.py    # Research-backed threshold tests with citations
│   ├── test_edge_cases.py             # Extreme inputs, NaN, empty data
│   ├── test_integration.py            # Stage-to-stage pipeline flow
│   ├── test_regression.py             # Golden file regression tests
│   ├── golden/                        # Golden reference files for regression
│   ├── test_quality_filtering.py      # Quality filter logic tests
│   ├── test_zonal_stats.py            # Per-district aggregation tests
│   ├── test_trend_model.py            # Log-linear trend fitting tests
│   ├── test_alan_classification.py    # ALAN threshold classification tests
│   ├── test_sky_brightness.py         # Radiance → mag → Bortle tests
│   ├── test_site_buffer.py            # UTM buffer geometry tests
│   ├── test_trend_diagnostics.py      # DW, JB, Cook's D tests
│   ├── test_layer_identification.py   # VIIRS filename parsing tests
│   └── test_config_integrity.py       # Config value sanity checks
│
├── data/
│   └── shapefiles/                    # Maharashtra district boundaries
├── viirs/                             # VIIRS data (not in git, user-provided)
└── outputs/
    ├── runs/<timestamp>/              # Timestamped run outputs
    │   ├── config_snapshot.json
    │   ├── pipeline_run.json          # Full StepResult provenance
    │   ├── pipeline.jsonl             # Structured JSON Lines log
    │   ├── subsets/{year}/            # Maharashtra-clipped rasters (shared)
    │   ├── district/
    │   │   ├── csv/                   # District CSVs
    │   │   ├── maps/                  # Choropleth maps, heatmaps
    │   │   ├── reports/               # 36 district PDFs
    │   │   └── diagnostics/           # Trend diagnostic panels
    │   ├── city/
    │   │   ├── csv/                   # Urban radial profiles, light dome
    │   │   ├── maps/                  # Radial profile plots
    │   │   ├── reports/               # City PDFs
    │   │   └── diagnostics/           # City diagnostics
    │   └── site/
    │       ├── csv/                   # Sky brightness, proximity
    │       ├── maps/                  # Site overlay, polar plots
    │       ├── reports/               # Dark-sky site PDFs
    │       └── diagnostics/           # Site diagnostics
    └── latest -> runs/<timestamp>     # Symlink to most recent
```

## Pipeline Architecture

The pipeline is organized into 4 independently runnable stages:

```
1. PREPROCESS  →  subsets/{year}/
   python3 -m src.preprocess --years 2012-2024

2. ANALYZE     →  {entity}/csv/ + {entity}/diagnostics/
   python3 -m src.viirs_process          # district analysis
   python3 -m src.site_analysis --type city   # city analysis
   python3 -m src.site_analysis --type site   # dark-sky site analysis

3. VISUALIZE   →  {entity}/maps/
   python3 -m src.generate_maps --type all

4. REPORT      →  {entity}/reports/
   python3 -m src.generate_reports --type all
```

### Formulas Subpackage (`src/formulas/`)

All scientific constants and pure computational functions live here. Source modules import from `src.formulas` instead of defining values inline. Key design principles:
- **Pure functions**: no I/O, no side effects
- **Single source of truth**: each threshold/constant defined once with citation
- **Backwards compatible**: source modules re-export at module level

### Pipeline Step Functions

Each pipeline step is a discrete function with explicit typed inputs and outputs:
- **`src/pipeline_steps.py`**: 15 district step functions
- **`src/site_pipeline_steps.py`**: 12 site/city step functions
- Each returns `(StepResult, output_data)` tuple
- Each uses `StepTimer` for timing and `log_step_summary()` for structured logging
- Two-tier exception handling: specific exceptions first, catch-all fallback with `exc_info=True`

### Structured Logging

All modules use `get_pipeline_logger(__name__)` from `src/logging_config`:
- **Console**: human-readable INFO level
- **File**: JSON Lines DEBUG level in `{run_dir}/pipeline.jsonl`
- **Rotating**: `logs/pipeline.log` (10 MB, 3 backups)
- **Step summaries**: machine-parseable entries via `log_step_summary()`

### Pipeline Runner (`src/pipeline_runner.py`)

Orchestrator with validation gates between steps:
- `--pipeline district|city|site|all` — select pipeline
- `--step fit_trends` — run single step from saved CSV
- Validates DataFrames between steps (column checks, NaN detection)
- Saves `PipelineRunResult` as JSON for provenance

## CLI Reference

### `python3 -m src.preprocess`

| Flag | Default | Description |
|------|---------|-------------|
| `--viirs-dir` | `./viirs` | Root directory with year folders |
| `--output-dir` | `./outputs` | Output directory |
| `--years` | `2012-2024` | Year range or comma-separated list |

### `python3 -m src.viirs_process`

| Flag | Default | Description |
|------|---------|-------------|
| `--viirs-dir` | `./viirs` | Root directory with year folders |
| `--shapefile-path` | `./data/shapefiles/maharashtra_district.geojson` | District boundaries |
| `--output-dir` | `./outputs` | Output directory |
| `--cf-threshold` | `5` | Minimum cloud-free observations |
| `--years` | `2012-2024` | Year range or comma-separated list |
| `--download-shapefiles` | off | Auto-download shapefiles if missing |

### `python3 -m src.site_analysis`

| Flag | Default | Description |
|------|---------|-------------|
| `--type` | `all` | Entity type: `city`, `site`, or `all` |
| `--output-dir` | `./outputs` | Output directory (must match viirs_process.py) |
| `--shapefile-path` | `./data/shapefiles/maharashtra_district.geojson` | District boundaries |
| `--buffer-km` | `10` | Buffer radius around sites (km) |
| `--cf-threshold` | `5` | Minimum cloud-free observations |
| `--years` | `2012-2024` | Year range |

### `python3 -m src.generate_maps`

| Flag | Default | Description |
|------|---------|-------------|
| `--type` | `all` | Entity type: `district`, `city`, `site`, or `all` |
| `--output-dir` | `./outputs` | Output directory |

### `python3 -m src.generate_reports`

| Flag | Default | Description |
|------|---------|-------------|
| `--type` | `all` | Entity type: `district`, `city`, `site`, or `all` |
| `--output-dir` | `./outputs` | Output directory |

### `python3 -m src.pipeline_runner`

| Flag | Default | Description |
|------|---------|-------------|
| `--pipeline` | `all` | Pipeline: `district`, `city`, `site`, or `all` |
| `--step` | (none) | Run single step by name (e.g., `fit_trends`) |
| `--output-dir` | `./outputs` | Output directory |
| `--years` | `2012-2024` | Year range |

### Shell Scripts (`scripts/`)

| Script | Description |
|--------|-------------|
| `bash scripts/run_all.sh` | Full pipeline: preprocess → analyze → maps → reports |
| `bash scripts/run_preprocess.sh` | Download, unpack, subset VIIRS data |
| `bash scripts/run_district.sh` | District analysis + maps + reports |
| `bash scripts/run_city.sh` | City analysis + maps + reports |
| `bash scripts/run_site.sh` | Dark-sky site analysis + maps + reports |

All scripts accept passthrough arguments (e.g., `--years 2024`).

## Configuration Reference

All parameters are in `src/config.py` with inline scientific citations:

| Category | Key constants |
|----------|---------------|
| Quality filtering | `CF_COVERAGE_THRESHOLD=5`, `USE_LIT_MASK=True`, `USE_CF_FILTER=True` |
| Trend modeling | `LOG_EPSILON=1e-6`, `BOOTSTRAP_RESAMPLES=1000`, `BOOTSTRAP_CI_LEVEL=(2.5, 97.5)`, `MIN_YEARS_FOR_TREND=2` |
| Spatial analysis | `SITE_BUFFER_RADIUS_KM=10`, `URBAN_GRADIENT_RADII_KM=[1,5,10,20,50]`, `MAHARASHTRA_UTM_EPSG=32643` |
| ALAN thresholds | `ALAN_LOW_THRESHOLD=1.0` nW, `ALAN_MEDIUM_THRESHOLD=5.0` nW, percentile bins |
| Locations | `URBAN_CITIES` (43 cities), `DARKSKY_SITES` (11 pilot sites) |
| Visualization | `MAP_DPI=300`, highlight districts, output directory structure |
| VIIRS metadata | `VIIRS_VERSION_MAPPING` (v21 for 2012-2013, v22 for 2014+), `VIIRS_RESOLUTION_DEG=0.00417` |

## Testing

Run the test suite from the project root:

```bash
source .venv/bin/activate
pytest tests/ -v
```

231 tests covering:

| Test File | Coverage |
|-----------|----------|
| `test_formulas.py` | Classification, trend fitting, all constants (67 tests) |
| `test_research_validation.py` | Every threshold with full research citation |
| `test_edge_cases.py` | Extreme values, NaN, empty inputs, type coercion |
| `test_integration.py` | Stage-to-stage data flow, schema validation, StepResult |
| `test_regression.py` | Golden file comparison (trends, classification, sky brightness) |
| `test_quality_filtering.py` | Quality filter logic |
| `test_zonal_stats.py` | Per-district aggregation |
| `test_trend_model.py` | Log-linear trend fitting (DataFrame wrapper) |
| `test_alan_classification.py` | ALAN threshold + boundary consistency |
| `test_sky_brightness.py` | Radiance → mag → Bortle conversion |
| `test_site_buffer.py` | UTM buffer geometry |
| `test_trend_diagnostics.py` | DW, JB, Cook's D diagnostics |
| `test_layer_identification.py` | VIIRS filename parsing |
| `test_config_integrity.py` | Config value sanity checks |

To update golden files after an intentional change:
```bash
pytest tests/test_regression.py --update-golden
```

## Troubleshooting

### "No subsets found" error
Run preprocessing first: `python3 -m src.preprocess --years 2024`

### Missing diagnostics plots
Check `{entity}/diagnostics/` — all diagnostic outputs now route to entity-specific subdirectories.

### "No module named src.formulas"
Ensure you're running from the project root directory.

### Log analysis
```bash
# Parse structured log
jq '.step_name, .status, .timing_seconds' outputs/latest/pipeline.jsonl

# Find failed steps
jq 'select(.level == "ERROR")' outputs/latest/pipeline.jsonl
```
