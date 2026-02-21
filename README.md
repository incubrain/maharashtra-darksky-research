# Maharashtra VIIRS Nighttime Lights: ALAN Trend Analysis (2012-2024)

Analysis of Artificial Light at Night (ALAN) trends across Maharashtra districts using VIIRS DNB annual composites, following the methodology in *"Preserving India's Rural Night Skies"* (Section 3.1).

## Objectives

1. **Quantify annual % change** in radiance (nW/cm^2/sr) using log-linear models with bootstrap CI
2. **Identify low-ALAN districts** (median < 1 nW/cm^2/sr) for dark-sky potential
3. **Analyze 43 cities + 11 dark-sky sites** with buffer-based spatial metrics
4. **Generate comprehensive reports**: district PDFs, site PDFs, publication-quality maps
5. **Ensure reproducibility** with centralized configuration, data provenance, and validation

## Quick Start

All commands must be run from the project root (`maharashtra-darksky-research/`).

```bash
# Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate synthetic test data (or download real VIIRS data — see below)
python3 -m src.outputs.download_viirs --generate-test-data --years 2012-2024

# Run full pipeline (district + city + site)
python3 -m src.pipeline_runner --pipeline all --download-shapefiles

# Audit pipeline plan without running
python3 -m src.pipeline_runner --pipeline all --dryrun
```

## Running the Pipeline

There is a **single entry point**: `python3 -m src.pipeline_runner`

```bash
# Full pipeline (district + city + site analysis)
python3 -m src.pipeline_runner --pipeline all --download-shapefiles

# District analysis only
python3 -m src.pipeline_runner --pipeline district --years 2012-2024

# City analysis only (43 cities with 10 km buffers)
python3 -m src.pipeline_runner --pipeline city

# Site analysis only (11 dark-sky candidates)
python3 -m src.pipeline_runner --pipeline site

# Include Census cross-dataset correlation analysis
python3 -m src.pipeline_runner --pipeline district --datasets census

# Re-run a single step from saved intermediate CSVs
python3 -m src.pipeline_runner --step fit_trends

# Dry run — print planned steps and exit
python3 -m src.pipeline_runner --pipeline all --dryrun
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--pipeline` | `district` | `district`, `city`, `site`, or `all` |
| `--years` | `2012-2024` | Year range or comma-separated years |
| `--download-shapefiles` | off | Download Maharashtra district boundaries if missing |
| `--dryrun` | off | Print pipeline steps and exit without running |
| `--datasets` | none | Dataset group (`census`, `census_district`, `census_towns`) or individual names or `all` |
| `--census-dir` | `data/census` | Override census data directory |
| `--step` | none | Run a single step from saved CSVs |
| `--strict-validation` | off | Abort on Pandera schema violations |
| `--buffer-km` | `10` | Buffer radius for site/city analysis |
| `--city-source` | `config` | `config` (43 hand-picked) or `census` (geocoded towns) |
| `--cf-threshold` | `5` | Minimum cloud-free observations per pixel |
| `--compare-run` | none | Path to a previous run for comparison |

### Getting VIIRS Data

**Option A — Synthetic test data** (for pipeline validation):

```bash
python3 -m src.outputs.download_viirs --generate-test-data --years 2012-2024
```

**Option B — Real data** (requires free [NOAA EOG](https://eogdata.mines.edu/products/vnl/) registration):

Download annual composites into `./viirs/<YEAR>/` directories. Each year folder should contain `.tif.gz` files for `median_masked`, `average_masked`, `cf_cvg`, and `lit_mask`.

### Output Structure

Each pipeline run creates a timestamped directory with a `latest` symlink:

```
outputs/
├── latest → runs/2026-02-21_143022
└── runs/
    └── 2026-02-21_143022/
        ├── config_snapshot.json
        ├── pipeline_run.json
        ├── subsets/{year}/            # Clipped Maharashtra rasters
        ├── district/
        │   ├── csv/                   # Yearly radiance, trends, stability
        │   ├── maps/                  # Choropleths, animations, per-district maps
        │   ├── reports/               # 36 district PDF reports
        │   └── diagnostics/           # Quality heatmaps, trend diagnostics
        ├── city/
        │   ├── csv/  maps/  reports/  diagnostics/
        └── site/
            ├── csv/  maps/  reports/  diagnostics/
```

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

## Project Structure

```
src/
├── config.py                    # Centralized configuration (thresholds, paths, sites)
├── pipeline_runner.py           # Single entry point — CLI orchestrator
├── pipeline_steps.py            # District-level step functions
├── pipeline_types.py            # StepResult / PipelineRunResult dataclasses
├── viirs_process.py             # Core VIIRS raster processing functions
├── viirs_utils.py               # Raster utilities (DBS)
├── dataset_aggregator.py        # Cross-dataset merge logic + group aliases
├── cross_dataset_steps.py       # Census correlation/classification steps
│
├── formulas/                    # Pure functions (no I/O, no state)
├── datasets/                    # External dataset adapters (census 1991/2001/2011)
├── analysis/                    # Domain-specific analysis modules
├── outputs/                     # Reports, maps, visualizations, downloads
└── site/                        # Site/city-level analysis pipeline
```

See [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) for detailed step-by-step documentation.

## Key References

- Elvidge, C.D. et al. (2017). VIIRS night-time lights. *Int. J. Remote Sensing*, 38(21), 5860-5879.
- Elvidge, C.D. et al. (2021). Annual time series of global VIIRS nighttime lights. *Remote Sensing*, 13(5), 922.
- Falchi, F. et al. (2016). The new world atlas of artificial night sky brightness. *Science Advances*, 2(6), e1600377.

## Citation

> *"Preserving India's Rural Night Skies: A VIIRS-based Assessment of Artificial Light at Night Trends and Dark-Sky Potential"*

## License

MIT
