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

## Setup & Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd maharashtra-darksky-research

# 2. Create virtual environment and install dependencies
python3 -m venv viirs_env
source viirs_env/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
src/
├── config.py                    # Centralized configuration (thresholds, paths, sites)
├── pipeline_runner.py           # CLI orchestrator for district pipeline
├── pipeline_steps.py            # District-level step functions (steps 1-15)
├── pipeline_types.py            # StepResult dataclass
├── logging_config.py            # StepTimer, pipeline logger
├── viirs_process.py             # Core VIIRS raster processing
├── viirs_utils.py               # Raster utilities (DBS, subsetting)
├── preprocess.py                # Unpack/subset preprocessing
├── validate_names.py            # District name validation
├── cache_manager.py             # Intermediate result caching
├── incremental_update.py        # Incremental year updates
├── dataset_aggregator.py        # Cross-dataset merge logic
├── cross_dataset_steps.py       # Steps 16-20 (dataset correlation/classification)
│
├── formulas/                    # Pure functions (no I/O, no state)
│   ├── classification.py        # ALAN thresholds, stability, graduated tiers
│   ├── trend.py                 # Log-linear OLS with bootstrap CI
│   ├── sky_brightness.py        # Radiance → mag/arcsec², Bortle scale
│   ├── spatial.py               # Direction definitions, Haversine
│   ├── ecology.py               # Land-cover classes, ecological sensitivity
│   ├── diagnostics.py           # DW, JB, Cook's D thresholds
│   ├── benchmarks.py            # IDA/satellite reference values
│   └── correlation.py           # Pearson, Spearman, partial, OLS
│
├── datasets/                    # External dataset adapters (registry pattern)
│   ├── __init__.py              # DATASET_REGISTRY mapping
│   ├── _base.py                 # DatasetMeta, DatasetResult dataclasses
│   ├── _name_resolver.py        # Fuzzy district-name matching
│   └── census_2011_pca.py       # Census 2011 PCA adapter
│
├── analysis/                    # Domain-specific analysis modules
│   ├── benchmark_comparison.py  # Compare to IDA/satellite benchmarks
│   ├── breakpoint_analysis.py   # Piecewise trend breakpoint detection
│   ├── buffer_comparison.py     # Inside vs outside buffer radiance
│   ├── directional_analysis.py  # N/S/E/W quadrant brightness
│   ├── ecological_overlay.py    # Land-cover / ecological sensitivity
│   ├── gradient_analysis.py     # Radial decay profiles from urban cores
│   ├── graduated_classification.py  # Percentile-based tier classification
│   ├── light_dome_modeling.py   # Urban light dome falloff models
│   ├── proximity_analysis.py    # Nearest-city distance metrics
│   ├── quality_diagnostics.py   # Data quality heatmaps
│   ├── sensitivity_analysis.py  # CF-threshold sensitivity sweeps
│   ├── sky_brightness_model.py  # Sky brightness computation + Bortle
│   ├── stability_metrics.py     # Temporal CV, IQR stability
│   └── trend_diagnostics.py     # Residual diagnostics (DW, JB, Cook's D)
│
├── outputs/                     # Reports, maps, visualizations, downloads
│   ├── district_reports.py      # Per-district PDF report generation
│   ├── download_viirs.py        # VIIRS data download / synthetic test data
│   ├── generate_maps.py         # Publication-quality choropleth maps
│   ├── generate_reports.py      # Report orchestrator (district + site)
│   ├── site_reports.py          # Per-site PDF report generation
│   ├── visualization_suite.py   # Comparison charts, boxplots, quality maps
│   └── visualizations.py        # Sprawl/differential/darkness animations
│
└── site/                        # Site/city-level analysis pipeline
    ├── site_analysis.py         # Site pipeline entry point + helper functions
    └── site_pipeline_steps.py   # Site-level step functions
```

## Running the Pipeline

There are **4 commands** to run, in order. Steps 1-2 prepare data, steps 3-4 run analysis.

### Step 1: Get VIIRS data

**Option A - Real data** (requires free NOAA EOG registration):

Download annual composites from [NOAA EOG](https://eogdata.mines.edu/nighttime_light/annual/v22/) into `./viirs/<YEAR>/` directories. Each year folder should contain `.tif.gz` files for `median_masked`, `average_masked`, `cf_cvg`, and `lit_mask`.

**Option B - Synthetic test data** (for pipeline validation):

```bash
python3 -m src.outputs.download_viirs --generate-test-data --years 2012-2024
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

To include cross-dataset analysis (e.g., Census 2011 correlations), add the `--datasets` flag:

```bash
python3 -m src.pipeline_runner --datasets census_2011_pca --census-dir ../census
```

This runs additional steps 16-20: loading external datasets, merging with VIIRS data, computing Pearson/Spearman/partial correlations, quadrant classification (HH/HL/LH/LL), and cross-dataset summary reports.

### Step 3: Run the site-level pipeline

```bash
python3 -m src.site.site_analysis
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

All commands must be run from the project root (`maharashtra-darksky-research/`) because modules use `from src import config`.

```bash
# Install
python3 -m venv viirs_env && source viirs_env/bin/activate
pip install -r requirements.txt

# Generate test data
python3 -m src.outputs.download_viirs --generate-test-data --years 2012-2024

# Run district analysis
python3 -m src.viirs_process --download-shapefiles

# Run site analysis
python3 -m src.site.site_analysis

# (Optional) Cross-dataset analysis with Census 2011
python3 -m src.pipeline_runner --datasets census_2011_pca --census-dir ../census

# (Optional) Sensitivity analysis
python3 scripts/run_sensitivity.py --year 2024
```

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
