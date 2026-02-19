# Census Data Pipeline

This document describes how Census of India data is sourced, processed,
and integrated into the Maharashtra VIIRS nighttime lights pipeline.

## Overview

We use Primary Census Abstract (PCA) data from three decennial censuses
(1991, 2001, 2011) to provide demographic context for VIIRS nighttime
radiance trends. Data is extracted at two granularity levels:

- **District level**: Total/Urban/Rural population per district (29-35 districts)
- **Town level**: Individual city/town populations (336-537 towns)

Census metrics are projected forward to the VIIRS study period (2012-2024)
using linear interpolation/extrapolation, enabling correlation analysis
between demographic change and light pollution at both district and town
scales.

## Data Sources

| Year | Source Files | Format | Coverage |
|------|-------------|--------|----------|
| 1991 | `1991-PCA-Urban-1400.xlsx`, `1991-PCA-Rural-1400.xlsx` | Single file per TRU, ward/village-level rows | 29 districts, 336 towns |
| 2001 | `PC01_PCA_TOT_27_XX.xls` (35 files) | Per-district, hierarchical rows | 35 districts, 379 towns |
| 2011 | `DDW_PCA27XX_2011_MDDS with UI.xlsx` (35 files) | Per-district, hierarchical rows | 35 districts, 537 towns |

Raw Excel files are stored outside the repo in `../census/{year}/` and
are **not** committed to version control (large, vendor-formatted). The
extraction script produces clean, normalised CSVs that **are** committed
in `data/census/`.

Source: [Census of India](https://censusindia.gov.in/)

## Processing Pipeline

### Step 1: Extract and normalise (`scripts/extract_census_csvs.py`)

The extraction script reads raw Excel files and produces four output types:

```bash
python scripts/extract_census_csvs.py [--raw-dir ../census] [--out-dir data/census]
python scripts/extract_census_csvs.py --include-villages  # also extract village rows
```

#### 1a. Consolidated district files (`census_YYYY_pca.csv`)

One row per district, Total population only. Backward compatible with
existing pipeline adapters. Schema: `district, district_code` + 12
metric columns.

#### 1b. Main PCA files (`{year}/27_YYYY_maharashtra_all_districts.csv`)

All districts with Total/Urban/Rural breakdown. Three rows per district.

#### 1c. Per-district files (`{year}/27_XX_YYYY_maharashtra_districtname.csv`)

Full breakdown: district Total/Urban/Rural rows + individual town rows
(+ village rows with `--include-villages`). Schema includes `level`,
`tru`, `entity_name`, `entity_code` columns.

#### 1d. Consolidated town files (`census_YYYY_towns.csv`)

All towns statewide in one flat CSV. Useful for cross-district town
analysis and VIIRS correlation.

#### What the script does per year

- **1991**: Reads both Urban and Rural files. Computes person-level
  totals from M/F split columns (e.g., `M_SC + F_SC → P_SC`). Groups
  urban wards by `Town_Code` and extracts town names from ward strings
  (e.g., `a)PUNE (M.CORP.) Ward No. 01` → `Pune (M.Corp.)`). Aggregates
  villages to district Rural totals. Sums Urban + Rural for district
  Total. Filters out "Greater Bombay" (unmappable to modern boundaries).

- **2001/2011**: Reads per-district Excel files. Extracts DISTRICT rows
  (Total/Urban/Rural), TOWN rows, and optionally VILLAGE rows.
  Normalises column names.

- **All years**: Retains only the 12 demographic columns common to all
  three censuses. Assigns canonical district codes from 2001/2011 scheme.

### Step 2: Load into pipeline (`src/datasets/`)

Dataset adapter modules load the CSVs at pipeline runtime:

| Module | Dataset name | Type | What it loads |
|--------|-------------|------|---------------|
| `census_1991.py` | `census_1991` | snapshot | District totals (29 districts) |
| `census_2001.py` | `census_2001` | snapshot | District totals (35 districts) |
| `census_2011.py` | `census_2011` | snapshot | District totals (35 districts) |
| `census_projected.py` | `census_projected` | timeseries | District projections (2012-2024) |
| `census_1991_towns.py` | `census_1991_towns` | snapshot | Town populations (336 towns) |
| `census_2001_towns.py` | `census_2001_towns` | snapshot | Town populations (379 towns) |
| `census_2011_towns.py` | `census_2011_towns` | snapshot | Town populations (537 towns) |
| `census_towns_projected.py` | `census_towns_projected` | timeseries | Town projections (2012-2024) |
| `_census_loader.py` | (internal) | — | Shared district-level loader |
| `_census_town_loader.py` | (internal) | — | Shared town-level loader |

**Snapshot datasets** broadcast their values to all VIIRS years during
merge.

**Timeseries datasets** merge year-matched: each VIIRS year gets the
projected value for that specific year.

### Step 3: Project to VIIRS years

Two projection modules use the same linear interpolation approach:

**District level** (`census_projected.py`): Loads `census_YYYY_pca.csv`,
projects 35 districts × 13 years = 455 rows.

**Town level** (`census_towns_projected.py`): Loads `census_YYYY_towns.csv`,
matches towns across census years within the same district using
normalised names, projects matched towns for 2012-2024.

#### Town matching strategy

Towns change names across censuses. The matching process:

1. **Normalise**: Strip municipal suffixes — `(M Corp.)`, `(M)`, `(M Cl)`,
   `(CT)`, `(CB)`, `(Cantt.)` — and lowercase
2. **Exact match**: Within the same district, match on normalised name
3. **Fuzzy match**: For unmatched single-year towns, try SequenceMatcher
   with threshold ≥ 0.85 against multi-year towns in the same district

Result: ~360 towns matched with 2+ anchors (projectable), ~264 towns in
only one census year (new Census Towns, skipped from projection).

#### Improving projections

The interpolation logic is isolated in `_interpolate()`. To improve:

1. Add new census years to `CENSUS_YEARS`
2. Or replace the linear formula with exponential, logistic, or spline
3. The rest of the module stays the same

### Step 4: Merge with VNL data (`src/dataset_aggregator.py`)

Column prefix convention:

| Prefix | Dataset |
|--------|---------|
| `c1991_` | Census 1991 district |
| `c2001_` | Census 2001 district |
| `c2011_` | Census 2011 district |
| `cproj_` | Census projected district |
| `c1991t_` | Census 1991 towns |
| `c2001t_` | Census 2001 towns |
| `c2011t_` | Census 2011 towns |
| `ctproj_` | Census projected towns |

### Step 5: Cross-dataset analysis (`src/cross_dataset_steps.py`)

Correlation and classification steps (16-20) compare VNL radiance metrics
against census-derived metrics using Pearson, Spearman, partial
correlation, and OLS regression.

## Common Schema

All three census years are normalised to these 12 columns:

| Column | Description |
|--------|------------|
| `No_HH` | Number of households |
| `TOT_P` | Total population |
| `TOT_M` | Male population |
| `TOT_F` | Female population |
| `P_06` | Population aged 0-6 |
| `P_SC` | Scheduled Caste population |
| `P_ST` | Scheduled Tribe population |
| `P_LIT` | Literate population |
| `TOT_WORK_P` | Total workers |
| `MAINWORK_P` | Main workers |
| `MARGWORK_P` | Marginal workers |
| `NON_WORK_P` | Non-workers |

### Per-district file schema

Per-district files add structural columns:

| Column | Values | Description |
|--------|--------|-------------|
| `level` | DISTRICT, TOWN, VILLAGE | Granularity level |
| `tru` | Total, Urban, Rural | Total/Rural/Urban split |
| `entity_name` | district/town/village name | Name of the entity |
| `entity_code` | Town_Code or Village_Code | Source identifier |

### Derived Ratios

Computed at load time from the common columns:

| Ratio | Formula |
|-------|---------|
| `literacy_rate` | P_LIT / TOT_P |
| `workforce_rate` | TOT_WORK_P / TOT_P |
| `dependency_ratio` | NON_WORK_P / TOT_WORK_P |
| `child_ratio` | P_06 / TOT_P |
| `sc_st_share` | (P_SC + P_ST) / TOT_P |
| `household_size` | TOT_P / No_HH |
| `sex_ratio` | TOT_F / TOT_M |

## Town Growth: 1991 → 2011

The number of towns grew significantly across censuses:

| Year | Towns | Notable |
|------|-------|---------|
| 1991 | 336 | Ward-level data aggregated to towns |
| 2001 | 379 | +43 new towns (villages reclassified) |
| 2011 | 537 | +158 new Census Towns (rapid urbanisation) |

The 158 new Census Towns in 2011 represent villages that crossed the
urban threshold — exactly the "villages that had no light until recent
years" pattern visible in VIIRS data.

## District Boundary Changes

| Period | Event | Affected Districts |
|--------|-------|--------------------|
| Pre-1991 | "Greater Bombay" existed as one district | Mumbai, Mumbai Suburban |
| 1991→2001 | 6 new districts carved from existing ones | Nandurbar (from Dhule), Washim (from Akola), Gondiya (from Bhandara), Hingoli (from Parbhani), Mumbai + Mumbai Suburban (from Greater Bombay) |
| 1991→2001 | Gadchiroli split from Chandrapur | Gadchiroli exists in 1991 rural data |

As a result:
- **1991** has 29 districts
- **2001 and 2011** each have 35 districts
- "Greater Bombay" is excluded from all processed data

## District Name Mismatches

| Census name | VNL shapefile name |
|------------|-------------------|
| Bid | Beed |
| Gondiya | Gondia |
| Raigarh | Raigad |
| Mumbai (Suburban) (2001) | Mumbai Suburban |

Resolved automatically by fuzzy name matching (threshold 0.8) in
`src/datasets/_name_resolver.py`.

## File Layout

```
data/census/
├── census_1991_pca.csv                         # District totals (29 rows)
├── census_2001_pca.csv                         # District totals (35 rows)
├── census_2011_pca.csv                         # District totals (35 rows)
├── census_1991_towns.csv                       # All towns (336 rows)
├── census_2001_towns.csv                       # All towns (379 rows)
├── census_2011_towns.csv                       # All towns (537 rows)
├── 1991/
│   ├── 27_1991_maharashtra_all_districts.csv   # Main PCA (Total/Urban/Rural)
│   ├── 27_02_1991_maharashtra_dhule.csv        # Per-district + towns
│   └── ...                                     # 29 district files
├── 2001/
│   ├── 27_2001_maharashtra_all_districts.csv
│   ├── 27_01_2001_maharashtra_nandurbar.csv
│   └── ...                                     # 35 district files
└── 2011/
    ├── 27_2011_maharashtra_all_districts.csv
    ├── 27_01_2011_maharashtra_nandurbar.csv
    └── ...                                     # 35 district files
```

## Usage

```bash
# District-level analysis
python3 -m src.pipeline_runner --datasets census_2001,census_2011,census_projected

# Town-level analysis
python3 -m src.pipeline_runner --datasets census_2011_towns,census_towns_projected

# Everything
python3 -m src.pipeline_runner --datasets census_1991,census_2001,census_2011,census_projected,census_2011_towns,census_towns_projected

# Re-extract from raw Excel (only needed if source data changes)
python scripts/extract_census_csvs.py
python scripts/extract_census_csvs.py --include-villages  # with village rows
```

## Code Responsibilities

| File | Purpose |
|------|---------|
| `scripts/extract_census_csvs.py` | Reads raw Excel, normalises columns, produces all CSVs |
| `src/datasets/_census_loader.py` | Shared district-level CSV loading, derived ratios, name resolution |
| `src/datasets/_census_town_loader.py` | Shared town-level CSV loading, town name normalisation |
| `src/datasets/census_{year}.py` | Thin adapter: loads district data for one year |
| `src/datasets/census_{year}_towns.py` | Thin adapter: loads town data for one year |
| `src/datasets/census_projected.py` | District-level linear projection to VIIRS years |
| `src/datasets/census_towns_projected.py` | Town-level projection with cross-census name matching |
| `src/config.py` | Common columns, derived ratios, dataset registry config |
| `src/dataset_aggregator.py` | Merges census data with VNL radiance metrics |
| `src/cross_dataset_steps.py` | Correlation/classification analysis (steps 16-20) |
