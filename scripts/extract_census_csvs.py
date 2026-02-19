"""
Extract and normalize Census PCA data from 1991, 2001, and 2011.

Reads raw Excel files from ``../census/{year}/`` (outside the repo) and
produces clean CSVs inside ``data/census/``:

1. **Consolidated district files** ``census_YYYY_pca.csv`` —
   one row per district, Total population only (backward compatible).
2. **Main PCA file per year** ``{year}/27_YYYY_maharashtra_all_districts.csv`` —
   all districts with Total/Urban/Rural rows.
3. **Per-district files** ``{year}/27_XX_YYYY_maharashtra_districtname.csv`` —
   district Total/Urban/Rural + individual towns (+ villages with --include-villages).
4. **Consolidated town files** ``census_YYYY_towns.csv`` —
   all towns statewide in one flat CSV.

Usage:
    python scripts/extract_census_csvs.py [--raw-dir ../census] [--out-dir data/census]
    python scripts/extract_census_csvs.py --include-villages   # also extract village rows
"""

import argparse
import glob
import os
import re

import pandas as pd


# ── Common output columns ───────────────────────────────────────────
# Present in all three census years after normalisation.
METRIC_COLUMNS = [
    "No_HH",          # Households
    "TOT_P",          # Total population
    "TOT_M",          # Male population
    "TOT_F",          # Female population
    "P_06",           # Population 0-6
    "P_SC",           # Scheduled Castes
    "P_ST",           # Scheduled Tribes
    "P_LIT",          # Literate population
    "TOT_WORK_P",     # Total workers
    "MAINWORK_P",     # Main workers
    "MARGWORK_P",     # Marginal workers
    "NON_WORK_P",     # Non-workers
]

# Full schema for per-district files (district-level + sub-entities)
FULL_COLUMNS = [
    "district", "district_code",
    "level",        # DISTRICT, TOWN, or VILLAGE
    "tru",          # Total, Urban, or Rural
    "entity_name",  # Name of entity (district/town/village name)
    "entity_code",  # Town_Code or Village_Code from source
] + METRIC_COLUMNS

# Backward-compat consolidated schema (district Total only)
LEGACY_COLUMNS = ["district", "district_code"] + METRIC_COLUMNS


# ── Canonical district numbering (2001/2011 scheme) ─────────────────
DISTRICT_NUMBERING = {
    "01": "Nandurbar", "02": "Dhule", "03": "Jalgaon", "04": "Buldana",
    "05": "Akola", "06": "Washim", "07": "Amravati", "08": "Wardha",
    "09": "Nagpur", "10": "Bhandara", "11": "Gondiya", "12": "Gadchiroli",
    "13": "Chandrapur", "14": "Yavatmal", "15": "Nanded", "16": "Hingoli",
    "17": "Parbhani", "18": "Jalna", "19": "Aurangabad", "20": "Nashik",
    "21": "Thane", "22": "Mumbai Suburban", "23": "Mumbai", "24": "Raigarh",
    "25": "Pune", "26": "Ahmadnagar", "27": "Bid", "28": "Latur",
    "29": "Osmanabad", "30": "Solapur", "31": "Satara", "32": "Ratnagiri",
    "33": "Sindhudurg", "34": "Kolhapur", "35": "Sangli",
}

_NAME_TO_CODE = {v: k for k, v in DISTRICT_NUMBERING.items()}

_NAME_ALIASES = {
    "Mumbai (Suburban)": "Mumbai Suburban",
    "Greater Bombay": None,  # cannot map to modern districts
}


def _normalise_district(raw_name: str) -> str:
    """Strip whitespace and apply district name aliases."""
    name = raw_name.strip("' *")
    return _NAME_ALIASES.get(name, name) or name


def _district_code(name: str) -> str | None:
    """Look up the 2-digit code for a canonical district name."""
    return _NAME_TO_CODE.get(name)


def _safe_filename(name: str) -> str:
    """Convert district name to a safe lowercase filename component."""
    return re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_"))


# ── 1991 helpers ────────────────────────────────────────────────────

def _compute_1991_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute person-level totals from M/F split columns (1991 format)."""
    df = df.copy()
    df["P_06"] = pd.to_numeric(df["POPLN_M6"], errors="coerce") + pd.to_numeric(df["POPLN_F6"], errors="coerce")
    df["P_SC"] = pd.to_numeric(df["M_SC"], errors="coerce") + pd.to_numeric(df["F_SC"], errors="coerce")
    df["P_ST"] = pd.to_numeric(df["M_ST"], errors="coerce") + pd.to_numeric(df["F_ST"], errors="coerce")
    df["P_LIT"] = pd.to_numeric(df["M_LITERATE"], errors="coerce") + pd.to_numeric(df["F_LITERATE"], errors="coerce")
    df["TOT_WORK_P"] = pd.to_numeric(df["T_M_WORKER"], errors="coerce") + pd.to_numeric(df["T_F_WORKER"], errors="coerce")
    df["MARGWORK_P"] = pd.to_numeric(df["M_MARGINAL"], errors="coerce") + pd.to_numeric(df["F_MARGINAL"], errors="coerce")
    df["NON_WORK_P"] = pd.to_numeric(df["M_NON_WORK"], errors="coerce") + pd.to_numeric(df["F_NON_WORK"], errors="coerce")
    df["MAINWORK_P"] = df["TOT_WORK_P"] - df["MARGWORK_P"]
    df = df.rename(columns={
        "HOUSEHOLDS": "No_HH", "T_POPLN": "TOT_P",
        "T_M_POPLN": "TOT_M", "T_F_POPLN": "TOT_F",
    })
    for col in METRIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _extract_town_name_1991(ward_name: str) -> str:
    """Extract town name from a 1991 ward name string.

    Examples:
        'a)PUNE (M.CORP.) Ward No. 01'          -> 'Pune (M.Corp.)'
        'ALANDI (M) Ward No. 01'                 -> 'Alandi (M)'
        'BHIGWAN (CT)* Ward No. 01'              -> 'Bhigwan (CT)'
        'SHIVATKAR (NIRA) (CT) Ward No. 01'      -> 'Shivatkar (Nira) (CT)'
    """
    name = re.sub(r"^[a-z]\)", "", ward_name).strip()
    name = re.sub(r"\s*Ward\s*No\.\s*[-]?\s*\d+.*$", "", name, flags=re.IGNORECASE).strip()
    name = name.rstrip("*").strip()
    return name.title()


def _load_1991(raw_dir: str, include_villages: bool = False) -> pd.DataFrame:
    """Load 1991 Census with full sub-district breakdown."""
    dir_1991 = os.path.join(raw_dir, "1991")
    urban_path = os.path.join(dir_1991, "1991-PCA-Urban-1400.xlsx")
    rural_path = os.path.join(dir_1991, "1991-PCA-Rural-1400.xlsx")

    rows = []  # will become the full DataFrame

    # ── Urban: extract towns (aggregate wards) and district Urban totals ──
    if os.path.exists(urban_path):
        udf = pd.read_excel(urban_path)
        udf = _compute_1991_metrics(udf)
        udf["district"] = udf["District_Name"].apply(_normalise_district)

        # Town-level: group wards by (district, Town_Code)
        for (dist, tcode), grp in udf.groupby(["district", "Town_Code"]):
            if dist == "Greater Bombay":
                continue
            code = _district_code(dist)
            if code is None:
                continue
            town_name = _extract_town_name_1991(str(grp.iloc[0]["Name"]))
            town_metrics = grp[METRIC_COLUMNS].sum()
            rows.append({
                "district": dist, "district_code": code,
                "level": "TOWN", "tru": "Urban",
                "entity_name": town_name, "entity_code": str(tcode).strip("' "),
                **town_metrics.to_dict(),
            })

        # District Urban totals
        for dist, grp in udf.groupby("district"):
            if dist == "Greater Bombay":
                continue
            code = _district_code(dist)
            if code is None:
                continue
            dist_metrics = grp[METRIC_COLUMNS].sum()
            rows.append({
                "district": dist, "district_code": code,
                "level": "DISTRICT", "tru": "Urban",
                "entity_name": dist, "entity_code": "",
                **dist_metrics.to_dict(),
            })
        print(f"  1991 Urban: towns and district totals extracted")
    else:
        print(f"  WARNING: 1991 Urban file not found: {urban_path}")

    # ── Rural: extract villages and district Rural totals ──
    if os.path.exists(rural_path):
        rdf = pd.read_excel(rural_path)
        rdf = _compute_1991_metrics(rdf)
        # Rural file has trailing-space column names
        name_col = [c for c in rdf.columns if "District" in c and "Name" in c][0]
        code_col = [c for c in rdf.columns if "District" in c and "Name" not in c][0]
        rdf["district"] = rdf[name_col].apply(_normalise_district)

        if include_villages:
            # Village-level rows
            vcode_col = [c for c in rdf.columns if "Village" in c and "Code" in c]
            vcode_col = vcode_col[0] if vcode_col else None
            vname_col = "NAME" if "NAME" in rdf.columns else "Name"
            for _, row in rdf.iterrows():
                dist = row["district"]
                dcode = _district_code(dist)
                if dcode is None:
                    continue
                vname = str(row[vname_col]).strip() if vname_col in rdf.columns else ""
                vcode = str(row[vcode_col]).strip("' ") if vcode_col else ""
                rows.append({
                    "district": dist, "district_code": dcode,
                    "level": "VILLAGE", "tru": "Rural",
                    "entity_name": vname.title(), "entity_code": vcode,
                    **{col: row[col] for col in METRIC_COLUMNS},
                })

        # District Rural totals
        for dist, grp in rdf.groupby("district"):
            code = _district_code(dist)
            if code is None:
                continue
            dist_metrics = grp[METRIC_COLUMNS].sum()
            rows.append({
                "district": dist, "district_code": code,
                "level": "DISTRICT", "tru": "Rural",
                "entity_name": dist, "entity_code": "",
                **dist_metrics.to_dict(),
            })
        print(f"  1991 Rural: district totals extracted"
              + (f" + villages" if include_villages else ""))
    else:
        print(f"  WARNING: 1991 Rural file not found: {rural_path}")

    if not rows:
        raise FileNotFoundError(f"No 1991 census files found in {dir_1991}")

    full_df = pd.DataFrame(rows)

    # ── Compute district Total (Urban + Rural) ──
    dist_rows = full_df[full_df["level"] == "DISTRICT"]
    total_rows = []
    for dist, grp in dist_rows.groupby("district"):
        code = grp.iloc[0]["district_code"]
        total_metrics = grp[METRIC_COLUMNS].sum()
        total_rows.append({
            "district": dist, "district_code": code,
            "level": "DISTRICT", "tru": "Total",
            "entity_name": dist, "entity_code": "",
            **total_metrics.to_dict(),
        })
    total_df = pd.DataFrame(total_rows)
    full_df = pd.concat([total_df, full_df], ignore_index=True)

    # Sort: district Total/Urban/Rural first, then towns, then villages
    level_order = {"DISTRICT": 0, "TOWN": 1, "VILLAGE": 2}
    tru_order = {"Total": 0, "Urban": 1, "Rural": 2}
    full_df["_level_sort"] = full_df["level"].map(level_order)
    full_df["_tru_sort"] = full_df["tru"].map(tru_order)
    full_df = full_df.sort_values(
        ["district_code", "_level_sort", "_tru_sort", "entity_name"]
    ).drop(columns=["_level_sort", "_tru_sort"]).reset_index(drop=True)

    n_dist = full_df[full_df["level"] == "DISTRICT"]["district"].nunique()
    n_towns = len(full_df[full_df["level"] == "TOWN"])
    n_villages = len(full_df[full_df["level"] == "VILLAGE"])
    print(f"  1991 Total: {n_dist} districts, {n_towns} towns, {n_villages} villages")
    return full_df[FULL_COLUMNS]


# ── 2001/2011 loading ──────────────────────────────────────────────

def _load_2001(raw_dir: str, include_villages: bool = False) -> pd.DataFrame:
    """Load 2001 Census with sub-district breakdown."""
    data_dir = os.path.join(raw_dir, "2001")
    files = sorted(glob.glob(os.path.join(data_dir, "PC01_PCA_TOT_27_*.xls")))
    if not files:
        raise FileNotFoundError(f"No 2001 census files in {data_dir}")

    all_rows = []
    for f in files:
        df = pd.read_excel(f)
        dist_num = os.path.basename(f).split("_")[-1].replace(".xls", "")

        # District-level: Total, Urban, Rural
        for tru in ["Total", "Urban", "Rural"]:
            mask = (df["LEVEL"] == "DISTRICT") & (df["TRU"] == tru)
            for _, row in df[mask].iterrows():
                dist_name = _normalise_district(str(row["NAME"]))
                all_rows.append({
                    "district": dist_name, "district_code": dist_num,
                    "level": "DISTRICT", "tru": tru,
                    "entity_name": dist_name, "entity_code": "",
                    **{col: pd.to_numeric(row.get(col), errors="coerce") for col in METRIC_COLUMNS},
                })

        # Town-level
        town_mask = df["LEVEL"] == "TOWN"
        dist_name_for_file = None
        for _, row in df[df["LEVEL"] == "DISTRICT"].head(1).iterrows():
            dist_name_for_file = _normalise_district(str(row["NAME"]))
        if dist_name_for_file is None:
            continue

        for _, row in df[town_mask].iterrows():
            town_name = str(row["NAME"]).strip("* ").title()
            town_code = str(row.get("TOWN_VILL", "")).strip()
            all_rows.append({
                "district": dist_name_for_file, "district_code": dist_num,
                "level": "TOWN", "tru": "Urban",
                "entity_name": town_name, "entity_code": town_code,
                **{col: pd.to_numeric(row.get(col), errors="coerce") for col in METRIC_COLUMNS},
            })

        # Village-level (optional)
        if include_villages:
            village_mask = df["LEVEL"] == "VILLAGE"
            for _, row in df[village_mask].iterrows():
                vname = str(row["NAME"]).strip("* ").title()
                vcode = str(row.get("TOWN_VILL", "")).strip()
                all_rows.append({
                    "district": dist_name_for_file, "district_code": dist_num,
                    "level": "VILLAGE", "tru": "Rural",
                    "entity_name": vname, "entity_code": vcode,
                    **{col: pd.to_numeric(row.get(col), errors="coerce") for col in METRIC_COLUMNS},
                })

    result = pd.DataFrame(all_rows)
    n_dist = result[result["level"] == "DISTRICT"]["district"].nunique()
    n_towns = len(result[result["level"] == "TOWN"])
    n_villages = len(result[result["level"] == "VILLAGE"])
    print(f"  2001: {n_dist} districts, {n_towns} towns, {n_villages} villages from {len(files)} files")
    return result[FULL_COLUMNS]


def _load_2011(raw_dir: str, include_villages: bool = False) -> pd.DataFrame:
    """Load 2011 Census with sub-district breakdown."""
    data_dir = os.path.join(raw_dir, "2011")
    patterns = [
        os.path.join(data_dir, "DDW_PCA27*_2011*.xlsx"),
        os.path.join(data_dir, "DDW_PCA27*_2011*.xls"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No 2011 census files in {data_dir}")

    all_rows = []
    for f in files:
        df = pd.read_excel(f, engine="openpyxl")
        m = re.search(r"PCA(\d{4})", os.path.basename(f))
        dist_num = m.group(1)[-2:] if m else "00"

        # District-level: Total, Urban, Rural
        for tru in ["Total", "Urban", "Rural"]:
            mask = (df["Level"] == "DISTRICT") & (df["TRU"] == tru)
            for _, row in df[mask].iterrows():
                dist_name = _normalise_district(str(row["Name"]))
                all_rows.append({
                    "district": dist_name, "district_code": dist_num,
                    "level": "DISTRICT", "tru": tru,
                    "entity_name": dist_name, "entity_code": "",
                    **{col: pd.to_numeric(row.get(col), errors="coerce") for col in METRIC_COLUMNS},
                })

        # Town-level
        dist_name_for_file = None
        for _, row in df[df["Level"] == "DISTRICT"].head(1).iterrows():
            dist_name_for_file = _normalise_district(str(row["Name"]))
        if dist_name_for_file is None:
            continue

        town_mask = df["Level"] == "TOWN"
        for _, row in df[town_mask].iterrows():
            town_name = str(row["Name"]).strip("* ").title()
            town_code = str(row.get("Town/Village", row.get("Town_Village", ""))).strip()
            all_rows.append({
                "district": dist_name_for_file, "district_code": dist_num,
                "level": "TOWN", "tru": "Urban",
                "entity_name": town_name, "entity_code": town_code,
                **{col: pd.to_numeric(row.get(col), errors="coerce") for col in METRIC_COLUMNS},
            })

        # Village-level (optional)
        if include_villages:
            village_mask = df["Level"] == "VILLAGE"
            for _, row in df[village_mask].iterrows():
                vname = str(row["Name"]).strip("* ").title()
                vcode = str(row.get("Town/Village", row.get("Town_Village", ""))).strip()
                all_rows.append({
                    "district": dist_name_for_file, "district_code": dist_num,
                    "level": "VILLAGE", "tru": "Rural",
                    "entity_name": vname, "entity_code": vcode,
                    **{col: pd.to_numeric(row.get(col), errors="coerce") for col in METRIC_COLUMNS},
                })

    result = pd.DataFrame(all_rows)
    n_dist = result[result["level"] == "DISTRICT"]["district"].nunique()
    n_towns = len(result[result["level"] == "TOWN"])
    n_villages = len(result[result["level"] == "VILLAGE"])
    print(f"  2011: {n_dist} districts, {n_towns} towns, {n_villages} villages from {len(files)} files")
    return result[FULL_COLUMNS]


# ── Output helpers ──────────────────────────────────────────────────

def _save_legacy_consolidated(df: pd.DataFrame, year: int, out_dir: str) -> str:
    """Save backward-compatible consolidated CSV (district Total only)."""
    dist_total = df[(df["level"] == "DISTRICT") & (df["tru"] == "Total")].copy()
    path = os.path.join(out_dir, f"census_{year}_pca.csv")
    dist_total[LEGACY_COLUMNS].to_csv(path, index=False)
    return path


def _save_main_pca(df: pd.DataFrame, year: int, out_dir: str) -> str:
    """Save main PCA file: all districts with Total/Urban/Rural."""
    dist_rows = df[df["level"] == "DISTRICT"].copy()
    year_dir = os.path.join(out_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)
    path = os.path.join(year_dir, f"27_{year}_maharashtra_all_districts.csv")
    dist_rows[FULL_COLUMNS].to_csv(path, index=False)
    return path


def _save_district_files(df: pd.DataFrame, year: int, out_dir: str) -> int:
    """Save per-district files with full breakdown. Returns count."""
    year_dir = os.path.join(out_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)
    count = 0
    for code, grp in df.groupby("district_code"):
        dist_name = grp.iloc[0]["district"]
        safe_name = _safe_filename(dist_name)
        filename = f"27_{code}_{year}_maharashtra_{safe_name}.csv"
        grp[FULL_COLUMNS].to_csv(os.path.join(year_dir, filename), index=False)
        count += 1
    return count


def _save_towns_consolidated(df: pd.DataFrame, year: int, out_dir: str) -> str:
    """Save consolidated town-level file (all towns statewide)."""
    towns = df[df["level"] == "TOWN"].copy()
    path = os.path.join(out_dir, f"census_{year}_towns.csv")
    towns[FULL_COLUMNS].to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Extract and normalize Census PCA data into data/census/"
    )
    parser.add_argument(
        "--raw-dir", default="../census",
        help="Path to raw census Excel files (default: ../census)",
    )
    parser.add_argument(
        "--out-dir", default="data/census",
        help="Output directory for processed CSVs (default: data/census)",
    )
    parser.add_argument(
        "--include-villages", action="store_true",
        help="Include village-level rows in per-district files (default: towns only)",
    )
    args = parser.parse_args()

    raw_dir = os.path.abspath(args.raw_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Raw census data:  {raw_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Include villages: {args.include_villages}")
    print()

    loaders = [
        (1991, _load_1991),
        (2001, _load_2001),
        (2011, _load_2011),
    ]

    for year, loader in loaders:
        try:
            print(f"── {year} ──")
            df = loader(raw_dir, include_villages=args.include_villages)

            # 1. Backward-compat consolidated (district Total only)
            con_path = _save_legacy_consolidated(df, year, out_dir)
            print(f"  -> Consolidated PCA: {con_path}")

            # 2. Main PCA file (all districts, Total/Urban/Rural)
            main_path = _save_main_pca(df, year, out_dir)
            print(f"  -> Main PCA: {main_path}")

            # 3. Per-district files (full breakdown)
            n_files = _save_district_files(df, year, out_dir)
            print(f"  -> {n_files} district files in {out_dir}/{year}/")

            # 4. Consolidated towns file
            towns_path = _save_towns_consolidated(df, year, out_dir)
            n_towns = len(df[df["level"] == "TOWN"])
            print(f"  -> Towns: {towns_path} ({n_towns} towns)")

            print()
        except Exception as e:
            print(f"  ERROR loading {year}: {e}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()
