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
│   ├── download_viirs.py              # Data download / synthetic test data generator
│   ├── viirs_process.py               # Main district-level pipeline
│   ├── site_analysis.py               # Site-level pipeline (cities + dark-sky sites)
│   ├── validate_names.py              # District name cross-validation
│   │
│   │   # Spatial analysis (Phase 2)
│   ├── gradient_analysis.py           # Urban-rural radial gradient profiles
│   ├── buffer_comparison.py           # Inside vs outside buffer ALAN comparison
│   ├── directional_analysis.py        # N/S/E/W quadrant brightness analysis
│   ├── proximity_analysis.py          # Nearest city distance metrics (Haversine)
│   │
│   │   # Temporal analysis (Phase 3)
│   ├── stability_metrics.py           # CV, IQR, max year-to-year change
│   ├── breakpoint_analysis.py         # AIC-based piecewise linear regression
│   ├── trend_diagnostics.py           # Durbin-Watson, Jarque-Bera, Cook's distance
│   │
│   │   # Validation & quality (Phase 4)
│   ├── quality_diagnostics.py         # Per-district pixel filter breakdown
│   ├── benchmark_comparison.py        # Maharashtra vs global/regional growth rates
│   ├── sensitivity_analysis.py        # CF threshold parameter sweeps
│   │
│   │   # Reporting (Phase 5)
│   ├── district_reports.py            # Consolidated district atlas PDF
│   ├── site_reports.py                # Consolidated site atlas PDF
│   ├── visualization_suite.py         # Publication-quality maps and charts
│   ├── light_dome_modeling.py         # Exponential decay light dome fitting
│   │
│   │   # Optimization (Phase 6)
│   ├── cache_manager.py               # SHA-256 keyed result caching
│   ├── parallel_processing.py         # Multiprocessing zonal stats
│   ├── incremental_update.py          # Smart output regeneration
│   │
│   │   # Enhancements (Phase 7)
│   ├── sky_brightness_model.py        # Radiance to mag/arcsec² + Bortle scale
│   ├── graduated_classification.py    # Percentile-based ALAN tier system
│   └── ecological_overlay.py          # Land cover cross-tabulation
│
├── scripts/
│   └── run_sensitivity.py             # Standalone sensitivity analysis runner
│
├── tests/
│   ├── conftest.py                    # Shared fixtures, synthetic raster helpers
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
│   ├── 2012/
│   ├── ...
│   └── 2024/
└── outputs/
    ├── runs/<timestamp>/              # Timestamped run outputs with config snapshot
    ├── latest -> runs/<timestamp>     # Symlink to most recent run
    ├── csv/                           # All CSV outputs
    ├── maps/                          # PNG/PDF visualizations (300 DPI)
    ├── subsets/                       # Maharashtra-clipped rasters per year
    ├── diagnostics/                   # Trend diagnostic plots
    ├── district_reports/              # District atlas PDF
    └── site_reports/                  # Site atlas PDF
```

## CLI Reference

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
| `--output-dir` | `./outputs` | Output directory (must match viirs_process.py) |
| `--shapefile-path` | `./data/shapefiles/maharashtra_district.geojson` | District boundaries |
| `--buffer-km` | `10` | Buffer radius around sites (km) |
| `--cf-threshold` | `5` | Minimum cloud-free observations |
| `--years` | `2012-2024` | Year range |

### `scripts/run_sensitivity.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `./outputs` | Output directory |
| `--shapefile-path` | `./data/shapefiles/maharashtra_district.geojson` | District boundaries |
| `--year` | `2024` | Year to test |
| `--thresholds` | `1,3,5,7,10` | Comma-separated CF threshold values |

### `python3 -m src.download_viirs`

| Flag | Default | Description |
|------|---------|-------------|
| `--viirs-dir` | `./viirs` | Output directory for VIIRS data |
| `--years` | `2012-2024` | Year range |
| `--generate-test-data` | off | Generate synthetic rasters |
| `--shapefile-path` | `./data/shapefiles/maharashtra_district.geojson` | Boundaries for masking test data |

## Output Files

### CSV files (`outputs/csv/`)

| File | Description |
|------|-------------|
| `districts_yearly_radiance.csv` | Per-district, per-year radiance statistics |
| `districts_trends.csv` | Annual % change, CI, R², ALAN classification |
| `district_stability_metrics.csv` | CV, IQR, max change, stability class per district |
| `district_breakpoints.csv` | Detected trend breakpoints per district |
| `trend_diagnostics.csv` | R², Durbin-Watson, Jarque-Bera, Cook's D per district |
| `quality_all_years.csv` | Pixel-level quality filter breakdown |
| `benchmark_comparison.csv` | Maharashtra vs global/regional growth benchmarks |
| `graduated_classification.csv` | Percentile-based ALAN tier assignments |
| `light_dome_metrics.csv` | Light dome radius/decay for major cities |
| `site_yearly_radiance.csv` | Per-site, per-year radiance statistics |
| `site_trends.csv` | Site-level annual % change with CI |
| `site_stability_metrics.csv` | Stability metrics for all 16 sites |
| `sky_brightness_*.csv` | mag/arcsec² + Bortle class per site |
| `urban_radial_profiles_*.csv` | Radial decay profiles for cities |

### Maps and visualizations (`outputs/maps/`)

| File | Description |
|------|-------------|
| `maharashtra_alan_trends.png` | Choropleth of annual % change |
| `maharashtra_radiance_latest.png` | Latest-year median radiance choropleth |
| `radiance_timeseries.png` | Time series for selected districts |
| `radiance_heatmap_log.png` | Districts x years heatmap (log-scale) |
| `multi_year_comparison.png` | 5-panel temporal choropleth grid |
| `growth_classification.png` | Classified growth rate map |
| `quality_heatmap.png` | Data quality heatmap |
| `tier_distribution.png` | ALAN tier stacked bar over time |
| `stability_scatter.png` | Mean radiance vs CV scatter |
| `breakpoint_timeline.png` | Breakpoint detection summary |
| `urban_radial_profiles.png` | Radial decay curves for cities |
| `light_dome_comparison.png` | Light dome extent comparison |
| `site_overlay_map.png` | Sites on Maharashtra map |
| `site_comparison_chart.png` | City vs dark-sky site bar chart |
| `sky_brightness_distribution.png` | Histogram with Bortle scale |
| `city_vs_site_boxplot.png` | Boxplot comparison |

### Reports

| File | Description |
|------|-------------|
| `outputs/district_reports/district_atlas.pdf` | Consolidated multi-page district atlas (all 36 districts) |
| `outputs/site_reports/site_atlas.pdf` | Consolidated multi-page site atlas (all 16 sites) |

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

## Known Issues & Bugs

### ~~ALAN classification boundary inconsistency~~ (FIXED)
Fixed by adding `right=False` to the `pd.cut` call in `site_analysis.py`, so both methods now use `[low, high)` intervals matching the `< threshold` logic in `viirs_process.py`. Verified by `tests/test_alan_classification.py::TestSiteALANClassification::test_boundary_consistency_with_district_method`.

### ~~Sky brightness RADIANCE_TO_MCD calibration~~ (FIXED)
Root cause: `REFERENCE_MCD` was `108,000` (cd/m² units) while `RADIANCE_TO_MCD` produced mcd/m², a 1000× unit mismatch. Fixed by setting `REFERENCE_MCD = 108,000,000` (mcd/m²) and recalibrating `RADIANCE_TO_MCD = 0.177` against World Atlas + SQM ground truth (1.22 nW → 20.92 mag). Model now gives physically correct results (0.5 nW → ~21.1 mag rural, 50 nW → ~17.7 mag city).

## Testing

Run the test suite from the project root:

```bash
source .venv/bin/activate
pytest tests/ -v
```

Tests use synthetic data (no real VIIRS data required). The test suite covers: quality filtering, zonal statistics, trend modeling, ALAN classification, sky brightness conversion, site buffer geometry, trend diagnostics, layer identification, and config integrity.

## Pipeline Architecture

1. **Data layer**: `download_viirs.py` fetches or generates `.tif.gz` VIIRS annual composites
2. **District pipeline**: `viirs_process.py` orchestrates all district-level analysis (quality → zonal stats → trends → diagnostics → maps → reports)
3. **Site pipeline**: `site_analysis.py` runs independently for 16 point locations with buffer analysis
4. **Each run** creates a timestamped directory under `outputs/runs/` with a `config_snapshot.json` and a `outputs/latest` symlink
