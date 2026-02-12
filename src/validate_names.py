"""
District name validation across shapefile, CSVs, and configuration.

Ensures consistent naming to prevent silent merge failures and data loss.
"""

from src.logging_config import get_pipeline_logger
import os

import geopandas as gpd
import pandas as pd

from src import config

log = get_pipeline_logger(__name__)


def _normalize(name):
    """Normalize district name for comparison."""
    return str(name).strip().lower()


def get_shapefile_districts(shapefile_path):
    """Extract district names from shapefile."""
    gdf = gpd.read_file(shapefile_path)
    return sorted(gdf["district"].unique().tolist())


def get_csv_districts(csv_paths):
    """Extract district names from all output CSVs."""
    all_districts = set()
    for path in csv_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "district" in df.columns:
                all_districts.update(df["district"].dropna().unique().tolist())
    return sorted(all_districts)


def get_config_districts():
    """Extract district names referenced in config location dicts."""
    districts = set()
    for info in config.URBAN_BENCHMARKS.values():
        districts.add(info["district"])
    for info in config.DARKSKY_SITES.values():
        districts.add(info["district"])
    return sorted(districts)


def validate_district_names(shapefile_path, csv_paths=None, check_config=True):
    """Cross-validate district names across all data sources.

    Args:
        shapefile_path: Path to Maharashtra district shapefile.
        csv_paths: Optional list of CSV output paths to check.
        check_config: Whether to validate config location districts.

    Returns:
        dict with validation results.

    Raises:
        ValueError: If critical mismatches are found.
    """
    results = {
        "shapefile_districts": [],
        "csv_districts": [],
        "config_districts": [],
        "mismatches": [],
        "missing_from_csv": [],
        "extra_in_csv": [],
        "config_mismatches": [],
        "valid": True,
    }

    # Load shapefile districts
    shp_districts = get_shapefile_districts(shapefile_path)
    results["shapefile_districts"] = shp_districts
    shp_normalized = {_normalize(d): d for d in shp_districts}

    log.info("Shapefile: %d districts loaded", len(shp_districts))

    # Check config location districts against shapefile
    if check_config:
        cfg_districts = get_config_districts()
        results["config_districts"] = cfg_districts

        for cd in cfg_districts:
            cd_norm = _normalize(cd)
            if cd_norm not in shp_normalized:
                # Try fuzzy match
                close = [s for s in shp_districts if _normalize(s).startswith(cd_norm[:4])]
                msg = f"Config district '{cd}' not found in shapefile"
                if close:
                    msg += f" (similar: {close})"
                results["config_mismatches"].append(msg)
                results["valid"] = False
                log.error(msg)

        if not results["config_mismatches"]:
            log.info("Config: %d location districts validated against shapefile",
                     len(cfg_districts))

    # Check CSV districts against shapefile
    if csv_paths:
        csv_districts = get_csv_districts(csv_paths)
        results["csv_districts"] = csv_districts
        csv_normalized = {_normalize(d): d for d in csv_districts}

        # Districts in shapefile but missing from CSV
        for sd in shp_districts:
            if _normalize(sd) not in csv_normalized:
                results["missing_from_csv"].append(sd)

        # Districts in CSV but not in shapefile (typos)
        for cd in csv_districts:
            if _normalize(cd) not in shp_normalized:
                close = [s for s in shp_districts if _normalize(s).startswith(_normalize(cd)[:4])]
                msg = f"CSV district '{cd}' not found in shapefile"
                if close:
                    msg += f" (similar: {close})"
                results["mismatches"].append(msg)
                results["valid"] = False

        # Case mismatches (not errors, just warnings)
        for cd in csv_districts:
            cd_norm = _normalize(cd)
            if cd_norm in shp_normalized and cd != shp_normalized[cd_norm]:
                log.warning("Case mismatch: CSV has '%s', shapefile has '%s'",
                            cd, shp_normalized[cd_norm])

        if results["missing_from_csv"]:
            log.warning("Districts in shapefile but missing from CSV: %s",
                        results["missing_from_csv"])

        if not results["mismatches"]:
            log.info("CSV: %d districts validated against shapefile", len(csv_districts))

    # Summary
    n = len(shp_districts)
    if results["valid"]:
        log.info("All %d districts validated across data sources", n)
    else:
        all_errors = results["mismatches"] + results["config_mismatches"]
        log.error("Validation FAILED with %d mismatches:", len(all_errors))
        for err in all_errors:
            log.error("  - %s", err)

    return results


def validate_or_exit(shapefile_path, csv_paths=None, check_config=True):
    """Run validation and exit with error if mismatches found.

    Intended for use at the start of processing scripts.
    """
    results = validate_district_names(shapefile_path, csv_paths, check_config)
    if not results["valid"]:
        raise ValueError(
            f"District name validation failed: {results['mismatches'] + results['config_mismatches']}"
        )
    return results
