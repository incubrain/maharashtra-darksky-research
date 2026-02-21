# Pipeline Guide

Comprehensive reference for all pipelines, outputs, and visualization code in the Maharashtra Dark Sky Research project.

## Overview

The project has three analysis pipelines:

| Pipeline | Entities | Description |
|----------|----------|-------------|
| **District** | 36 Maharashtra districts | Zonal statistics, trends, classification, reports |
| **City** | 43 urban centers | Buffer-based metrics, spatial analysis, site reports |
| **Site** | 11 dark-sky candidate sites | Buffer-based metrics, sky brightness, suitability assessment |

All three can be run together with `--pipeline all`, or individually.

---

## Quick Start

```bash
# Full district pipeline (most common)
python3 -m src.pipeline_runner --pipeline district --years 2012-2024

# All three pipelines
python3 -m src.pipeline_runner --pipeline all --years 2012-2024

# City pipeline only
python3 -m src.pipeline_runner --pipeline city --years 2012-2024

# Site pipeline only
python3 -m src.pipeline_runner --pipeline site --years 2012-2024

# Single step from saved CSVs (no re-analysis)
python3 -m src.pipeline_runner --pipeline district --step animation_frames

# Regenerate maps only (from saved CSVs, no re-analysis)
python3 -m src.outputs.generate_maps --type all

# Regenerate PDF reports only
python3 -m src.outputs.generate_reports --type all

# City/site pipeline standalone (alternative entry point)
python3 -m src.site.site_analysis --type city --years 2012-2024
python3 -m src.site.site_analysis --type site --years 2012-2024
```

---

## CLI Reference

### `python3 -m src.pipeline_runner`

| Flag | Default | Description |
|------|---------|-------------|
| `--pipeline` | `district` | `district`, `city`, `site`, or `all` |
| `--step` | None | Run a single step from saved CSVs |
| `--years` | `2012-2024` | Year range or comma-separated |
| `--viirs-dir` | `config.DEFAULT_VIIRS_DIR` | VIIRS data root directory |
| `--shapefile-path` | `config.DEFAULT_SHAPEFILE_PATH` | District boundaries GeoJSON |
| `--output-dir` | `./outputs` | Output directory |
| `--cf-threshold` | `5` | Cloud-free coverage threshold |
| `--strict-validation` | `false` | Abort on Pandera schema violations |
| `--datasets` | None | Enable cross-dataset analysis (e.g., `census_2011_pca`) |
| `--buffer-km` | `config.SITE_BUFFER_RADIUS_KM` | Buffer radius for city/site analysis |
| `--city-source` | `config` | City locations: `config` (43) or `census` (geocoded) |

### Available `--step` values for district pipeline

`fit_trends`, `stability`, `breakpoints`, `trend_diagnostics`, `benchmark`, `graduated_classification`, `animation_frames`, `per_district_radiance_maps`

---

## District Pipeline Steps

| # | Step | Source | Outputs | Critical? |
|---|------|--------|---------|-----------|
| 1 | `load_boundaries` | `pipeline_steps.py` | GeoDataFrame (36 districts) | Yes |
| 2 | `process_years` | `pipeline_steps.py` → `viirs_process.py` | DataFrame (36 districts x N years) | Yes |
| 3 | `save_yearly` | `pipeline_steps.py` | `district/csv/districts_yearly_radiance.csv` | No |
| 4 | `fit_trends` | `pipeline_steps.py` → `viirs_process.py` | `district/csv/districts_trends.csv` | Yes |
| 5 | `basic_maps` | `pipeline_steps.py` → `viirs_process.generate_maps()` | 3 PNGs (see below) | No |
| 6 | `stability` | `pipeline_steps.py` → `stability_metrics` | CSV + scatter PNG | No |
| 7 | `breakpoints` | `pipeline_steps.py` → `breakpoint_analysis` | CSV + timeline PNG | No |
| 8 | `trend_diagnostics` | `pipeline_steps.py` → `trend_diagnostics` | CSV + diagnostic PNGs for flagged districts | No |
| 9 | `quality_diagnostics` | `pipeline_steps.py` → `quality_diagnostics` | CSV + heatmap PNG | No |
| 10 | `benchmark` | `pipeline_steps.py` → `benchmark_comparison` | CSV | No |
| 11 | `radial_gradient` | `pipeline_steps.py` → `gradient_analysis` | CSV + radial profiles PNG | No |
| 12 | `light_dome` | `pipeline_steps.py` → `light_dome_modeling` | CSV + dome comparison PNG | No |
| 13 | `statewide_viz` | `pipeline_steps.py` → `visualization_suite` | 4 PNGs (see below) | No |
| 14 | `graduated_classification` | `pipeline_steps.py` → `graduated_classification` | CSV + 2 PNGs | No |
| 15 | `district_reports` | `pipeline_steps.py` → `district_reports` | `district_atlas.pdf` (36x4 pages) | No |
| 16 | `animation_frames` | `pipeline_steps.py` → `visualizations` | Frame PNGs for GIF creation + trend map | No |
| 17 | `per_district_radiance_maps` | `pipeline_steps.py` → `visualizations` | 36 per-district radiance PNGs | No |
| 18+ | Cross-dataset steps | `cross_dataset_steps.py` | Gated by `--datasets` flag | No |

---

## City/Site Pipeline Steps

Runs for each entity type (`city`, `site`). Uses the same subset rasters as the district pipeline.

| # | Step | Source | Outputs |
|---|------|--------|---------|
| 1 | `build_site_buffers` | `site_pipeline_steps.py` → `site_analysis` | GeoDataFrame with circular buffers |
| 2 | `compute_yearly_metrics` | `site_pipeline_steps.py` → `site_analysis` | DataFrame (sites x years) |
| 3 | `save_site_yearly` | `site_pipeline_steps.py` | `{type}/csv/{type}_yearly_radiance.csv` |
| 4 | `fit_site_trends` | `site_pipeline_steps.py` → `site_analysis` | `{type}/csv/{type}_trends.csv` |
| 5 | `site_maps` | `site_pipeline_steps.py` → `site_analysis` | Overlay, radiance chart, raster overlay PNGs + per-district city maps |
| 6 | `spatial_analysis` | `site_pipeline_steps.py` | Buffer comparison, directional, proximity CSVs + PNGs |
| 7 | `sky_brightness` | `site_pipeline_steps.py` → `sky_brightness_model` | CSV + distribution PNG |
| 8 | `site_stability` | `site_pipeline_steps.py` → `stability_metrics` | CSV + scatter PNG |
| 9 | `site_breakpoints` | `site_pipeline_steps.py` → `breakpoint_analysis` | CSV |
| 10 | `site_benchmark` | `site_pipeline_steps.py` → `benchmark_comparison` | CSV |
| 11 | `site_reports` | `site_pipeline_steps.py` → `site_reports` | `site_atlas.pdf` |

---

## Output Directory Structure

```
outputs/
├── latest -> <timestamp>/             # Symlink to most recent run
├── <timestamp>/                       # Run-level directory
│   ├── pipeline_run.json              # Run provenance (steps, timings, git SHA)
│   ├── diagnostics/
│   │   └── run_report.html            # HTML diagnostic report
│   ├── subsets/                        # Intermediate raster subsets
│   │   ├── 2012/
│   │   │   ├── maharashtra_median_2012.tif
│   │   │   ├── maharashtra_lit_mask_2012.tif
│   │   │   └── maharashtra_cf_cvg_2012.tif
│   │   └── .../
│   ├── district/
│   │   ├── csv/
│   │   │   ├── districts_yearly_radiance.csv
│   │   │   ├── districts_trends.csv
│   │   │   ├── district_stability_metrics.csv
│   │   │   ├── district_breakpoints.csv
│   │   │   ├── trend_diagnostics.csv
│   │   │   ├── quality_all_years.csv
│   │   │   ├── benchmark_comparison.csv
│   │   │   ├── urban_radial_profiles_<year>.csv
│   │   │   ├── light_dome_metrics.csv
│   │   │   └── graduated_classification.csv
│   │   ├── maps/
│   │   │   ├── maharashtra_alan_trends.png
│   │   │   ├── maharashtra_radiance_latest.png
│   │   │   ├── radiance_timeseries.png
│   │   │   ├── multi_year_comparison.png
│   │   │   ├── growth_classification.png
│   │   │   ├── radiance_heatmap_log.png
│   │   │   ├── data_quality_map.png
│   │   │   ├── urban_radial_profiles.png
│   │   │   ├── light_dome_comparison.png
│   │   │   ├── tier_distribution.png
│   │   │   ├── tier_transitions_<start>_<end>.png
│   │   │   ├── alan_trend_map.png
│   │   │   ├── frames/
│   │   │   │   ├── sprawl/sprawl_<year>.png
│   │   │   │   ├── differential/diff_<year>.png
│   │   │   │   └── darkness/darkness_<year>.png
│   │   │   └── districts/
│   │   │       ├── <district>_radiance.png   (x36)
│   │   │       └── <district>_cities.png     (from city pipeline)
│   │   ├── diagnostics/
│   │   │   ├── stability_scatter.png
│   │   │   ├── breakpoint_timeline.png
│   │   │   ├── quality_heatmap.png
│   │   │   └── <district>_diagnostics.png    (flagged districts only)
│   │   └── reports/
│   │       └── district_atlas.pdf
│   ├── city/
│   │   ├── csv/
│   │   │   ├── city_yearly_radiance.csv
│   │   │   ├── city_trends.csv
│   │   │   ├── city_stability_metrics.csv
│   │   │   ├── city_breakpoints.csv
│   │   │   └── city_benchmark_comparison.csv
│   │   ├── maps/
│   │   │   ├── overlay_map.png
│   │   │   ├── radiance_chart.png
│   │   │   ├── radiance_overlay.png
│   │   │   ├── buffer_comparison.png
│   │   │   ├── directional_brightness_polar.pdf
│   │   │   └── sky_brightness_distribution.png
│   │   ├── diagnostics/
│   │   │   └── city_stability_scatter.png
│   │   └── reports/
│   │       └── site_atlas.pdf
│   └── site/
│       ├── csv/
│       │   ├── site_yearly_radiance.csv
│       │   ├── site_trends.csv
│       │   └── ...
│       ├── maps/
│       │   ├── overlay_map.png
│       │   ├── radiance_chart.png
│       │   ├── radiance_overlay.png
│       │   └── sky_brightness_distribution.png
│       ├── diagnostics/
│       │   └── site_stability_scatter.png
│       └── reports/
│           └── site_atlas.pdf
```

---

## Visualization Catalog

### District Pipeline — Basic Maps (`viirs_process.generate_maps`)

| Output | Description | Source |
|--------|-------------|--------|
| `maharashtra_alan_trends.png` | Choropleth: annual % change per district | `src/viirs_process.py:generate_maps()` |
| `maharashtra_radiance_latest.png` | Choropleth: latest year median radiance (log-scale) | `src/viirs_process.py:generate_maps()` |
| `radiance_timeseries.png` | Line plot: selected districts over time | `src/viirs_process.py:generate_maps()` |

### District Pipeline — Statewide Visualizations (`visualization_suite`)

| Output | Description | Source |
|--------|-------------|--------|
| `multi_year_comparison.png` | 5-panel choropleth grid (2012, 2015, 2018, 2021, 2024) | `src/outputs/visualization_suite.py` |
| `growth_classification.png` | Quartile-based growth classification choropleth | `src/outputs/visualization_suite.py` |
| `radiance_heatmap_log.png` | District x Year heatmap (log10 radiance, magma) | `src/outputs/visualization_suite.py` |
| `data_quality_map.png` | Mean quality % per district (RdYlGn) | `src/outputs/visualization_suite.py` |

### District Pipeline — Analysis Plots

| Output | Description | Source |
|--------|-------------|--------|
| `stability_scatter.png` | CV vs mean radiance scatter | `src/analysis/stability_metrics.py` |
| `breakpoint_timeline.png` | Detected trend breakpoints | `src/analysis/breakpoint_analysis.py` |
| `quality_heatmap.png` | District x metric NaN counts | `src/analysis/quality_diagnostics.py` |
| `urban_radial_profiles.png` | Radiance decay from city centers | `src/analysis/gradient_analysis.py` |
| `light_dome_comparison.png` | Multi-panel dome profile comparison | `src/analysis/light_dome_modeling.py` |
| `tier_distribution.png` | ALAN tier counts stacked bar chart | `src/analysis/graduated_classification.py` |
| `tier_transitions_*.png` | Tier movement heatmap (first→last year) | `src/analysis/graduated_classification.py` |

### District Pipeline — Animation Frames (`visualizations`)

| Output | Description | Source |
|--------|-------------|--------|
| `frames/sprawl/sprawl_<year>.png` | Binary lit/unlit per year (yellow on black) | `src/outputs/visualizations.py` |
| `frames/differential/diff_<year>.png` | Year vs baseline change (RdBu_r) | `src/outputs/visualizations.py` |
| `frames/darkness/darkness_<year>.png` | Dark area erosion per year (teal on black) | `src/outputs/visualizations.py` |
| `alan_trend_map.png` | Pixel-wise regression slope (coolwarm) | `src/outputs/visualizations.py` |

### District Pipeline — Per-District Maps

| Output | Description | Source |
|--------|-------------|--------|
| `districts/<name>_radiance.png` | Zoomed raster radiance per district (magma, log-scale) | `src/outputs/visualizations.py` |
| `districts/<name>_cities.png` | Cities colored by ALAN class (from city pipeline) | `src/site/site_analysis.py` |

### District Pipeline — Reports

| Output | Description | Source |
|--------|-------------|--------|
| `district_atlas.pdf` | 36x4 page PDF (summary, time series, spatial context, pixel counts) | `src/outputs/district_reports.py` |
| `run_report.html` | HTML diagnostic report (step summary, trends, anomalies) | `src/outputs/diagnostic_report.py` |

### City/Site Pipeline — Maps

| Output | Description | Source |
|--------|-------------|--------|
| `overlay_map.png` | State map with site buffers colored by radiance | `src/site/site_analysis.py` |
| `radiance_chart.png` | Radiance distribution (histogram + boxplot or bar chart) | `src/site/site_analysis.py` |
| `radiance_overlay.png` | Log-scale raster with site overlays | `src/site/site_analysis.py` |
| `buffer_comparison.png` | Inside vs outside buffer radiance | `src/analysis/buffer_comparison.py` |
| `directional_brightness_polar.pdf` | Polar plot of directional light spillover | `src/analysis/directional_analysis.py` |
| `sky_brightness_distribution.png` | Sky brightness histogram with Bortle scale | `src/analysis/sky_brightness_model.py` |
| `site_atlas.pdf` | Per-site PDF reports (3-4 pages each) | `src/outputs/site_reports.py` |

---

## Creating GIFs from Animation Frames

After running the pipeline (or `--step animation_frames`), convert frames to GIFs:

```bash
# Using ImageMagick
convert -delay 80 -loop 0 outputs/latest/district/maps/frames/sprawl/sprawl_*.png sprawl.gif
convert -delay 80 -loop 0 outputs/latest/district/maps/frames/differential/diff_*.png differential.gif
convert -delay 80 -loop 0 outputs/latest/district/maps/frames/darkness/darkness_*.png darkness.gif

# Using ffmpeg (higher quality)
ffmpeg -framerate 2 -pattern_type glob -i 'outputs/latest/district/maps/frames/sprawl/sprawl_*.png' -vf "scale=1400:-1" sprawl.gif
```

---

## Standalone Regeneration

These tools regenerate visualizations from saved CSVs without re-running analysis:

```bash
# Regenerate all maps (district + city + site)
python3 -m src.outputs.generate_maps --type all

# Regenerate only district maps (includes animation frames + per-district maps)
python3 -m src.outputs.generate_maps --type district

# Regenerate all PDF reports
python3 -m src.outputs.generate_reports --type all
```

Both tools auto-resolve `outputs/latest/` when using the default output directory.

---

## Source Code Map

```
src/
├── pipeline_runner.py          # Main orchestrator (CLI, step sequencing, validation)
├── pipeline_steps.py           # District step functions (17 steps)
├── pipeline_types.py           # StepResult, PipelineRunResult dataclasses
├── schemas.py                  # Pandera DataFrame schemas
├── viirs_process.py            # Core VIIRS processing + generate_maps()
├── config.py                   # Thresholds, paths, entity lists
├── site/
│   ├── site_analysis.py        # City/site entry point + generate_site_maps()
│   └── site_pipeline_steps.py  # City/site step functions (11 steps)
├── analysis/
│   ├── stability_metrics.py    # Temporal stability + scatter plot
│   ├── breakpoint_analysis.py  # Trend breakpoint detection + timeline
│   ├── trend_diagnostics.py    # Model diagnostics + panel plots
│   ├── quality_diagnostics.py  # Quality reports + heatmap
│   ├── gradient_analysis.py    # Radial profiles + decay curves
│   ├── light_dome_modeling.py  # Light dome models + comparison
│   ├── graduated_classification.py  # ALAN tier classification + charts
│   ├── benchmark_comparison.py # Compare to benchmark locations
│   ├── buffer_comparison.py    # Inside/outside buffer analysis
│   ├── directional_analysis.py # Directional brightness polar plots
│   ├── sky_brightness_model.py # Sky brightness estimation + Bortle
│   ├── sensitivity_analysis.py # Parameter sensitivity sweeps
│   └── ecological_overlay.py   # Landcover overlay analysis
├── outputs/
│   ├── visualizations.py       # Animation frames + per-district radiance maps
│   ├── visualization_suite.py  # Publication-quality statewide maps
│   ├── district_reports.py     # District PDF atlas
│   ├── site_reports.py         # Site/city PDF reports
│   ├── diagnostic_report.py    # HTML diagnostic report
│   ├── generate_maps.py        # Standalone map regeneration
│   └── generate_reports.py     # Standalone report regeneration
└── formulas/
    └── classification.py       # ALAN classification thresholds
```
