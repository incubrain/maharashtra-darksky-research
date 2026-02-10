# Maharashtra VIIRS Nighttime Lights: ALAN Trend Analysis (2012-2024)

Analysis of Artificial Light at Night (ALAN) trends across Maharashtra districts using VIIRS DNB annual composites, following the methodology in *"Preserving India's Rural Night Skies"* (Section 3.1).

## Objectives

1. **Quantify annual % change** in radiance (nW/cm^2/sr) using log-linear models with bootstrap CI
2. **Identify low-ALAN districts** (median < 1 nW/cm^2/sr) for dark-sky potential
3. **Analyze 16 sites** (5 cities + 11 dark-sky candidates) with buffer-based spatial metrics
4. **Generate comprehensive reports**: district PDFs, site PDFs, publication-quality maps
5. **Ensure reproducibility** with centralized configuration, data provenance, and validation

## Data Sources

- **VIIRS DNB Annual Composites**: [NOAA Earth Observation Group](https://eogdata.mines.edu/products/vnl/)
  - Layers: `median_masked`, `average_masked`, `cf_cvg`, `lit_mask`
  - Years: 2012-2024 (v21 for 2012-2013, v22 for 2014-2024)
- **Maharashtra District Boundaries**: [datta07/INDIAN-SHAPEFILES](https://github.com/datta07/INDIAN-SHAPEFILES) GeoJSON (36 districts)

## Methods

1. **Unpack** `.gz` VIIRS annual composites per year (one layer at a time to conserve disk)
2. **Subset** global rasters to Maharashtra extent using district shapefile union
3. **Filter** pixels: `lit_mask > 0` AND `cf_cvg >= 5` cloud-free observations (Elvidge et al. 2017)
4. **Aggregate** per district via zonal statistics (mean, median, count, min, max, std)
5. **Fit trend**: `log(median_radiance + 1e-6) ~ year` (OLS) per district
6. **Compute**: annual % change = `(exp(beta) - 1) * 100`, with 1000-resample bootstrap 95% CI
7. **Classify**: absolute thresholds (low < 1, medium 1-5, high > 5 nW) and percentile-based graduated tiers
8. **Spatial analysis**: radial gradients, buffer comparison, directional brightness, proximity metrics
9. **Temporal analysis**: stability (CV, IQR), breakpoint detection (piecewise AIC), trend diagnostics (DW, JB, Cook's D)
10. **Sky brightness**: convert radiance to mag/arcsec^2 with Bortle scale classification (Falchi et al. 2016)

## Repository Structure

```
maharashtra-viirs/
├── README.md
├── requirements.txt
├── data_manifest.json              # VIIRS product metadata & processing history
├── .gitignore
├── src/
│   ├── config.py                   # Centralized configuration (all parameters)
│   ├── download_viirs.py           # Data download / synthetic test data generator
│   ├── viirs_process.py            # Main district-level pipeline
│   ├── site_analysis.py            # Site-level pipeline (cities + dark-sky sites)
│   ├── validate_names.py           # District name cross-validation
│   │
│   │   # Spatial analysis (Phase 2)
│   ├── gradient_analysis.py        # Urban-rural radial gradient profiles
│   ├── buffer_comparison.py        # Inside vs outside buffer ALAN comparison
│   ├── directional_analysis.py     # N/S/E/W quadrant brightness analysis
│   ├── proximity_analysis.py       # Nearest city distance metrics (Haversine)
│   │
│   │   # Temporal analysis (Phase 3)
│   ├── stability_metrics.py        # CV, IQR, max year-to-year change
│   ├── breakpoint_analysis.py      # AIC-based piecewise linear regression
│   ├── trend_diagnostics.py        # Durbin-Watson, Jarque-Bera, Cook's distance
│   │
│   │   # Validation & quality (Phase 4)
│   ├── quality_diagnostics.py      # Per-district pixel filter breakdown
│   ├── benchmark_comparison.py     # Maharashtra vs global/regional growth rates
│   ├── sensitivity_analysis.py     # CF threshold parameter sweeps
│   │
│   │   # Reporting (Phase 5)
│   ├── district_reports.py         # 36 multi-page district PDF reports
│   ├── site_reports.py             # 16 multi-page site PDF reports
│   ├── visualization_suite.py      # Publication-quality maps and charts
│   ├── light_dome_modeling.py      # Exponential decay light dome fitting
│   │
│   │   # Optimization (Phase 6)
│   ├── cache_manager.py            # SHA-256 keyed result caching
│   ├── parallel_processing.py      # Multiprocessing zonal stats
│   ├── incremental_update.py       # Smart output regeneration
│   │
│   │   # Enhancements (Phase 7)
│   ├── sky_brightness_model.py     # Radiance to mag/arcsec^2 + Bortle scale
│   ├── graduated_classification.py # Percentile-based ALAN tier system
│   └── ecological_overlay.py       # Land cover cross-tabulation
│
├── scripts/
│   └── run_sensitivity.py          # Standalone sensitivity analysis runner
│
├── data/
│   └── shapefiles/                 # Maharashtra district boundaries
├── viirs/                          # VIIRS data (not in git, user-provided)
│   ├── 2012/
│   ├── ...
│   └── 2024/
└── outputs/
    ├── csv/                        # All CSV outputs
    ├── maps/                       # PNG/PDF visualizations (300 DPI)
    ├── subsets/                     # Maharashtra-clipped rasters per year
    ├── diagnostics/                # Trend diagnostic plots
    ├── district_reports/           # Per-district PDF reports
    └── site_reports/               # Per-site PDF reports
```

## Setup & Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd maharashtra-viirs

# 2. Create virtual environment and install dependencies
python3 -m venv viirs_env
source viirs_env/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline

There are **4 commands** to run, in order. Steps 1-2 prepare data, steps 3-4 run analysis.

### Step 1: Get VIIRS data

**Option A - Real data** (requires free NOAA EOG registration):

Download annual composites from [NOAA EOG](https://eogdata.mines.edu/nighttime_light/annual/v22/) into `./viirs/<YEAR>/` directories. Each year folder should contain `.tif.gz` files for `median_masked`, `average_masked`, `cf_cvg`, and `lit_mask`.

**Option B - Synthetic test data** (for pipeline validation):

```bash
python3 -m src.download_viirs --generate-test-data --years 2012-2024
```

This creates realistic synthetic rasters with urban hotspots (Mumbai, Pune, Nagpur, etc.) and a gradual 8%/yr growth trend, suitable for end-to-end pipeline testing.

### Step 2: Run the district-level pipeline

```bash
python3 -m src.viirs_process --download-shapefiles
```

This is the main pipeline. It processes all years (2012-2024) and runs:
- District-level zonal statistics and trend fitting
- Quality diagnostics and benchmark comparisons
- Stability metrics, breakpoint detection, trend diagnostics
- Urban radial gradient analysis and light dome modeling
- Graduated percentile classification
- Publication-quality maps and 36 district PDF reports

### Step 3: Run the site-level pipeline

```bash
python3 -m src.site_analysis
```

Analyzes 5 urban benchmark cities and 11 dark-sky candidate sites using 10 km circular buffers:
- Site-level radiance trends with bootstrap CI
- Sky brightness estimation (mag/arcsec^2) with Bortle scale
- Inside vs outside buffer comparison
- Directional (N/S/E/W) brightness analysis
- Nearest city distance metrics
- Temporal stability and breakpoint analysis
- City vs site comparison visualizations
- 16 per-site PDF reports

### Step 4 (Optional): Run sensitivity analysis

```bash
python3 scripts/run_sensitivity.py --year 2024 --thresholds 1,3,5,7,10
```

Sweeps the cloud-free coverage threshold parameter to test result robustness.

## Quick Start (Full Pipeline with Synthetic Data)

All commands must be run from the project root (`maharashtra-viirs/`) because modules use `from src import config`.

```bash
# Install
python3 -m venv viirs_env && source viirs_env/bin/activate
pip install -r requirements.txt

# Generate test data
python3 -m src.download_viirs --generate-test-data --years 2012-2024

# Run district analysis
python3 -m src.viirs_process --download-shapefiles

# Run site analysis
python3 -m src.site_analysis

# (Optional) Sensitivity analysis
python3 scripts/run_sensitivity.py --year 2024
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

## Outputs

### CSV files (`outputs/csv/`)

| File | Description |
|------|-------------|
| `districts_yearly_radiance.csv` | Per-district, per-year radiance statistics |
| `districts_trends.csv` | Annual % change, CI, R^2, ALAN classification |
| `district_stability_metrics.csv` | CV, IQR, max change, stability class per district |
| `district_breakpoints.csv` | Detected trend breakpoints per district |
| `trend_diagnostics.csv` | R^2, Durbin-Watson, Jarque-Bera, Cook's D per district |
| `quality_all_years.csv` | Pixel-level quality filter breakdown |
| `benchmark_comparison.csv` | Maharashtra vs global/regional growth benchmarks |
| `graduated_classification.csv` | Percentile-based ALAN tier assignments |
| `light_dome_metrics.csv` | Light dome radius/decay for major cities |
| `site_yearly_radiance.csv` | Per-site, per-year radiance statistics |
| `site_trends.csv` | Site-level annual % change with CI |
| `site_stability_metrics.csv` | Stability metrics for all 16 sites |
| `sky_brightness_*.csv` | mag/arcsec^2 + Bortle class per site |
| `urban_radial_profiles_*.csv` | Radial decay profiles for cities |

### Maps and visualizations (`outputs/maps/`)

| File | Description |
|------|-------------|
| `maharashtra_alan_trends.png` | Choropleth of annual % change |
| `maharashtra_radiance_latest.png` | Latest-year median radiance choropleth |
| `radiance_timeseries.png` | Time series for selected districts |
| `radiance_heatmap.png` | Districts x years heatmap |
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

| Directory | Description |
|-----------|-------------|
| `outputs/district_reports/` | 36 multi-page PDF reports (one per district) |
| `outputs/site_reports/` | 16 multi-page PDF reports (one per site) |

## Configuration

All analysis parameters are centralized in `src/config.py` with inline scientific citations:

- **Quality filtering**: `CF_COVERAGE_THRESHOLD`, `USE_LIT_MASK`, `USE_CF_FILTER`
- **Trend modeling**: `LOG_EPSILON`, `BOOTSTRAP_RESAMPLES`, `BOOTSTRAP_CI_LEVEL`, `MIN_YEARS_FOR_TREND`
- **Spatial analysis**: `SITE_BUFFER_RADIUS_KM`, `URBAN_GRADIENT_RADII_KM`, `MAHARASHTRA_UTM_EPSG`
- **ALAN thresholds**: `ALAN_LOW_THRESHOLD` (1.0 nW), `ALAN_MEDIUM_THRESHOLD` (5.0 nW), percentile bins
- **Locations**: `URBAN_BENCHMARKS` (5 cities), `DARKSKY_SITES` (11 candidate sites)
- **Visualization**: `MAP_DPI` (300), highlight districts, output directory structure

## Key References

- Elvidge, C.D. et al. (2017). VIIRS night-time lights. *Int. J. Remote Sensing*, 38(21), 5860-5879.
- Elvidge, C.D. et al. (2021). Annual time series of global VIIRS nighttime lights. *Remote Sensing*, 13(5), 922.
- Falchi, F. et al. (2016). The new world atlas of artificial night sky brightness. *Science Advances*, 2(6), e1600377.
- Wang, J. et al. (2022). Protected area buffer analysis.
- Zheng, Q. et al. (2019). Developing a new cross-sensor calibration model. *Remote Sensing*, 11(18), 2132.

## Citation

If using this analysis, please cite:

> *"Preserving India's Rural Night Skies: A VIIRS-based Assessment of Artificial Light at Night Trends and Dark-Sky Potential"*

## License

MIT
