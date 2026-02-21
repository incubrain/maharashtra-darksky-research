# AGENTS.md — Developer & Agent Reference

This file contains technical reference information for agents and developers working on the Maharashtra VIIRS pipeline. For project overview, methods, and user-facing documentation see [README.md](README.md).

## Repository Structure

```
maharashtra-viirs/
├── README.md
├── AGENTS.md                          # This file
├── requirements.txt
├── pyproject.toml                     # pytest, coverage, marker config
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
│   ├── pipeline_types.py              # StepResult, PipelineRunResult (with from_dict)
│   ├── pipeline_steps.py              # 15 district pipeline step functions
│   ├── site_pipeline_steps.py         # 12 site/city pipeline step functions
│   ├── pipeline_runner.py             # Orchestrator with Pandera validation + NaN tracking
│   ├── logging_config.py              # JSON Lines logging with run_id, reset_logging()
│   ├── schemas.py                     # Pandera DataFrameSchema definitions
│   ├── data_audit.py                  # Consolidated raster data audit (standalone + step)
│   ├── run_diagnostics.py             # Pipeline run comparison / diff tool
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
│   │   # Reporting & diagnostics
│   ├── outputs/
│   │   ├── diagnostic_report.py       # Per-run HTML diagnostic report generator
│   │   ├── district_reports.py        # Consolidated district atlas PDF
│   │   ├── site_reports.py            # Consolidated site atlas PDF
│   │   ├── visualization_suite.py     # Publication-quality maps and charts
│   │   └── light_dome_modeling.py     # Exponential decay light dome fitting
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
│   ├── conftest.py                    # Shared fixtures, logging reset, error-case rasters
│   ├── test_formulas.py               # Formulas subpackage (67 tests)
│   ├── test_research_validation.py    # Research-backed threshold tests with citations
│   ├── test_edge_cases.py             # Extreme values, NaN, empty inputs, type coercion
│   ├── test_integration.py            # Stage-to-stage data flow, logging, StepResult
│   ├── test_regression.py             # Golden file regression tests
│   ├── test_property_based.py         # Hypothesis property-based tests for formulas
│   ├── test_validation_gates.py       # Pandera schema + pipeline error handling tests
│   ├── golden/                        # Golden reference files for regression
│   ├── test_quality_filtering.py      # Quality filter logic tests
│   ├── test_zonal_stats.py            # Per-district aggregation tests
│   ├── test_trend_model.py            # Log-linear trend fitting tests
│   ├── test_alan_classification.py    # ALAN threshold + boundary consistency
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
    │   ├── pipeline_run.json          # Full StepResult provenance (with git_sha)
    │   ├── pipeline.jsonl             # Structured JSON Lines log (with run_id)
    │   ├── diagnostics/
    │   │   ├── run_report.html        # Auto-generated HTML diagnostic report
    │   │   └── data_audit/            # Raster audit CSVs and histograms
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

### Structured Logging (`src/logging_config.py`)

All modules use `get_pipeline_logger(__name__)` from `src/logging_config`:
- **Console**: human-readable, level controlled by `LOG_LEVEL` env var (default: INFO)
- **File**: JSON Lines DEBUG level in `{run_dir}/pipeline.jsonl`
- **Rotating**: `logs/pipeline.log` (10 MB, 3 backups)
- **Step summaries**: machine-parseable entries via `log_step_summary()`
- **Run correlation**: every log entry includes a `run_id` (UUID4 short form)
- **Millisecond timestamps**: `%Y-%m-%dT%H:%M:%S.NNN` format in JSON
- **Test isolation**: `reset_logging()` clears all handlers between tests

### Pandera Schema Validation (`src/schemas.py`)

Declarative DataFrame validation at pipeline boundaries:
- `YearlyRadianceSchema` — district yearly data (radiance 0–500, year 2012–2050, pixel_count > 0)
- `TrendsSchema` — district trends (r_squared 0–1, annual_pct_change -50–100)
- `SiteYearlySchema`, `SiteTrendsSchema` — site pipeline equivalents
- `StabilitySchema` — stability metrics (cv >= 0)
- `validate_schema(df, schema, step_name, strict=False)` — returns warnings or raises

### NaN Propagation Tracking

The pipeline runner tracks NaN counts per column after each step via `track_nan_counts()`. When NaN count increases between consecutive steps, a WARNING is logged identifying the column and delta. NaN summaries are stored in `StepResult.nan_summary`.

### Pipeline Runner (`src/pipeline_runner.py`)

Orchestrator with validation gates between steps:
- `--pipeline district|city|site|all` — select pipeline
- `--step fit_trends` — run single step from saved CSV
- `--strict-validation` — abort on Pandera schema failures (default: warn)
- `--compare-run <path>` — compare against a previous run
- Validates DataFrames with Pandera schemas between steps
- Tracks NaN propagation between steps
- Saves `PipelineRunResult` as JSON with git SHA and timestamps
- Auto-generates HTML diagnostic report at `diagnostics/run_report.html`

### Pipeline Types (`src/pipeline_types.py`)

- `StepResult` — per-step outcome with `started_at`, `completed_at`, `nan_summary`, `from_dict()`
- `PipelineRunResult` — full run result with `git_sha`, `started_at`, `from_dict()`
- `StepStatus` enum — `SUCCESS`, `SKIPPED`, `ERROR`

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

### `python3 -m src.pipeline_runner`

| Flag | Default | Description |
|------|---------|-------------|
| `--pipeline` | `district` | Pipeline: `district`, `city`, `site`, or `all` |
| `--step` | (none) | Run single step by name (e.g., `fit_trends`) |
| `--output-dir` | `./outputs` | Output directory |
| `--years` | `2012-2024` | Year range |
| `--strict-validation` | off | Abort pipeline on Pandera schema failures |
| `--compare-run` | (none) | Path to previous run for comparison |

### `python3 -m src.data_audit`

| Flag | Default | Description |
|------|---------|-------------|
| `--viirs-dir` | `./outputs` | Directory containing `subsets/` |
| `--years` | `2012-2024` | Year range |
| `--output-dir` | (auto) | Output directory for audit reports |
| `--histograms` | off | Generate radiance distribution histograms |

### `python3 -m src.run_diagnostics`

| Flag | Default | Description |
|------|---------|-------------|
| `--run-a` | (required) | Path to baseline run directory |
| `--run-b` | (required) | Path to comparison run directory |
| `--output` | (stdout) | Output path for Markdown report |
| `--json` | off | Output raw JSON diff instead of Markdown |

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

### Test Categories

| Test File | Coverage |
|-----------|----------|
| `test_formulas.py` | Classification, trend fitting, all constants (67 tests) |
| `test_research_validation.py` | Every threshold with full research citation |
| `test_edge_cases.py` | Extreme values, NaN, empty inputs, type coercion |
| `test_integration.py` | Stage-to-stage data flow, logging infrastructure, StepResult |
| `test_regression.py` | Golden file comparison (trends, classification, sky brightness) |
| `test_property_based.py` | Hypothesis property-based tests for all formula modules |
| `test_validation_gates.py` | Pandera schema validation, error paths, NaN tracking |
| `test_quality_filtering.py` | Quality filter logic |
| `test_zonal_stats.py` | Per-district aggregation |
| `test_trend_model.py` | Log-linear trend fitting (DataFrame wrapper) |
| `test_alan_classification.py` | ALAN threshold + boundary consistency |
| `test_sky_brightness.py` | Radiance → mag → Bortle conversion |
| `test_site_buffer.py` | UTM buffer geometry |
| `test_trend_diagnostics.py` | DW, JB, Cook's D diagnostics |
| `test_layer_identification.py` | VIIRS filename parsing |
| `test_config_integrity.py` | Config value sanity checks |

### Running with Coverage

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

Coverage is configured in `pyproject.toml` with `fail_under = 15` (branch coverage enabled).

### Custom Markers

```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Skip tests that need real VIIRS data
pytest tests/ -m "not requires_data"
```

### Property-Based Tests

The `test_property_based.py` file uses Hypothesis to verify invariants across all valid inputs:
- **Classification totality**: every valid radiance gets a class
- **Monotonicity**: higher radiance never produces a lower class
- **Trend fitting**: valid positive input never produces NaN
- **Sky brightness**: higher radiance always means brighter (lower magnitude)
- **Radiance transforms**: values remain non-negative after log transform

### Updating Golden Files

After an intentional change to formula outputs:
```bash
pytest tests/test_regression.py --update-golden
```

## Debugging & Diagnostics

### Data Audit Tool

The consolidated data audit tool (`src/data_audit.py`) replaces the former ad-hoc `debug_*.py` scripts:

```bash
# Full audit: statistics, histograms, threshold sensitivity
python -m src.data_audit --viirs-dir ./outputs --years 2012-2024 --histograms

# Quick stats for specific years
python -m src.data_audit --viirs-dir ./outputs --years 2012,2020,2024
```

Outputs:
- `data_audit_report.csv` — per-year raster statistics (min, P01, P05, median, P95, P99, max, pct_below thresholds)
- `radiance_distribution_histograms.png` — overlay histogram of low-radiance distributions
- `threshold_sensitivity.csv` — pixel counts at various radiance thresholds

### Pipeline Diagnostic Report

Every pipeline run automatically generates `diagnostics/run_report.html` containing:
- **Run summary**: git SHA, timing, step pass/fail counts
- **Step results table**: status, timing, errors for each step
- **Trend overview**: horizontal bar chart of annual % change by district
- **Data quality**: NaN distribution heatmap by district × column
- **Anomaly flags**: districts with R² < 0.3 or |change| > 20%/yr

### Comparing Pipeline Runs

```bash
# Markdown diff
python -m src.run_diagnostics --run-a outputs/run_20240115 --run-b outputs/run_20240116

# JSON diff
python -m src.run_diagnostics --run-a outputs/run_a --run-b outputs/run_b --json

# Save to file
python -m src.run_diagnostics --run-a run_a --run-b run_b --output comparison.md
```

The comparison report shows:
- Step status changes (success → error, etc.)
- Timing regressions between runs
- CSV value changes (mean, median shifts per column)
- Added/removed columns and files

### Adding New Pandera Validations

To add a new schema, edit `src/schemas.py`:

```python
NewSchema = DataFrameSchema(
    columns={
        "col_name": Column(float, Check.in_range(0.0, 100.0), nullable=False),
    },
    strict=False,
    name="NewSchema",
)
```

Then use it in the pipeline runner:
```python
from src.schemas import validate_schema, NewSchema
warnings = validate_schema(df, NewSchema, "step_name", strict=args.strict_validation)
```

### Log Analysis

```bash
# Parse structured log with run_id
jq '.run_id, .step_name, .status, .timing_seconds' outputs/latest/pipeline.jsonl

# Find failed steps
jq 'select(.level == "ERROR")' outputs/latest/pipeline.jsonl

# Track NaN propagation
jq 'select(.nan_summary != null)' outputs/latest/pipeline.jsonl

# Filter by run_id
jq 'select(.run_id == "abc12345")' outputs/latest/pipeline.jsonl

# Control console verbosity
LOG_LEVEL=DEBUG python3 -m src.pipeline_runner --pipeline district
```

## Troubleshooting

### "No subsets found" error
Run preprocessing first: `python3 -m src.preprocess --years 2024`

### Missing diagnostics plots
Check `{entity}/diagnostics/` — all diagnostic outputs route to entity-specific subdirectories.

### "No module named src.formulas"
Ensure you're running from the project root directory.

### Schema validation failures
Use `--strict-validation` to abort on failures, or check the log for WARNING-level schema messages. The diagnostic report lists all validation issues.

### Zonal stats returning all None / pixel_count=0
This was caused by a GDAL MemoryFile corruption bug (fixed in `193dff0`). If you
see this pattern, it means `zonal_stats()` is being called with a file path
(MemoryFile or on-disk) after prior large raster operations have corrupted GDAL's
internal state. The fix is to pass numpy arrays + affine transforms directly:
```python
# WRONG — silently fails after large rasterio.mask.mask() calls:
zonal_stats(gdf, src.name, stats=[...], nodata=np.nan)

# CORRECT — bypasses GDAL RasterIO, always reliable:
zonal_stats(gdf, array, stats=[...], nodata=np.nan, affine=transform)
```
Root cause: rasterio (GDAL 3.12) and fiona (GDAL 3.9) link different GDAL versions.
The `rasterio.mask.mask()` call on ~11 GB global composites leaves GDAL's
`GDALRasterIOExtraArg` struct version in an inconsistent state, causing subsequent
MemoryFile reads to fail silently (returns `None` for all stats, `0` for count).

### Debugging data quality issues
1. Run `python -m src.data_audit --histograms` to see radiance distributions
2. Check `threshold_sensitivity.csv` to understand how thresholds affect pixel counts
3. Compare runs with `python -m src.run_diagnostics` to see what changed
4. Look at `NaN count increased` warnings in the pipeline log
