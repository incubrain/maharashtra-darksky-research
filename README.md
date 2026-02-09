# Maharashtra VIIRS Nighttime Lights: ALAN Trend Analysis (2012–2024)

Analysis of Artificial Light at Night (ALAN) trends across Maharashtra districts using VIIRS DNB annual composites, following the methodology in *"Preserving India's Rural Night Skies"* (Section 3.1).

## Objectives

1. **Quantify annual % change** in radiance (nW/cm²/sr) using log-linear models
2. **Identify low-ALAN districts** (median < 1 nW/cm²/sr) for dark-sky potential
3. **Output tables and maps** for paper integration (CSV trends, PNG maps)
4. **Ensure reproducibility** with a clean, documented pipeline

## Data Sources

- **VIIRS DNB Annual Composites**: [NOAA Earth Observation Group](https://eogdata.mines.edu/products/vnl/)
  - Layers: `average_masked`, `median_masked`, `cf_cvg`, `lit_mask`
  - Years: 2012–2024 (v21/v22)
- **Maharashtra District Boundaries**: [HindustanTimesLabs shapefiles](https://github.com/HindustanTimesLabs/shapefiles)

## Methods

1. **Unpack** `.gz` VIIRS annual composites per year
2. **Subset** global rasters to Maharashtra extent using district shapefile
3. **Filter** pixels: `lit_mask > 0` AND `cf_cvg >= 5` (cloud-free observations)
4. **Aggregate** per district via zonal statistics (mean, median, count)
5. **Fit trend**: `log(median_radiance + 1e-6) ~ year` (OLS) per district
6. **Compute**: annual % change = `(exp(β) - 1) × 100`, with bootstrap 95% CI
7. **Classify**: low (< 1 nW), medium (1–5 nW), high (> 5 nW)

## Repository Structure

```
maharashtra-viirs/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── viirs_process.py      # Main processing script
├── data/
│   └── shapefiles/            # Maharashtra district boundaries
├── viirs/                     # VIIRS data (not in git, user-provided)
│   ├── 2012/
│   ├── ...
│   └── 2024/
├── outputs/
│   ├── csv/                   # districts_trends.csv, districts_yearly_radiance.csv
│   └── maps/                  # PNG visualizations (300 DPI)
└── notebooks/                 # Optional exploration notebooks
```

## Setup & Reproduction

```bash
# 1. Clone the repository
git clone <repo-url>
cd maharashtra-viirs

# 2. Create virtual environment and install dependencies
python3 -m venv viirs_env
source viirs_env/bin/activate
pip install -r requirements.txt

# 3. Add VIIRS data
# Download annual composites from NOAA EOG into ./viirs/<year>/ directories
# Each year folder should contain .gz files for avg_rade9h, median_masked, cf_cvg, lit_mask

# 4. Run the pipeline
python src/viirs_process.py --download-shapefiles

# 5. Check outputs
ls outputs/csv/    # districts_trends.csv, districts_yearly_radiance.csv
ls outputs/maps/   # 4 PNG visualizations
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--viirs-dir` | `./viirs` | Root directory with year folders |
| `--shapefile-path` | `./data/shapefiles/maharashtra_district.shp` | District shapefile |
| `--output-dir` | `./outputs` | Output directory |
| `--cf-threshold` | `5` | Minimum cloud-free observations |
| `--years` | `2012-2024` | Year range or comma-separated list |
| `--download-shapefiles` | off | Auto-download shapefiles if missing |

## Outputs

- **`districts_trends.csv`**: Per-district annual % change, CI, R², ALAN classification
- **`districts_yearly_radiance.csv`**: Per-district, per-year radiance statistics
- **`maharashtra_alan_trends.png`**: Choropleth of annual % change
- **`maharashtra_radiance_latest.png`**: Choropleth of latest-year median radiance
- **`radiance_timeseries.png`**: Time series for selected districts
- **`radiance_heatmap.png`**: Districts × years heatmap

## Citation

If using this analysis, please cite:

> *"Preserving India's Rural Night Skies: A VIIRS-based Assessment of Artificial Light at Night Trends and Dark-Sky Potential"*

## License

MIT
