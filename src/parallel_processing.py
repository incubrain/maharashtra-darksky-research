"""
Parallel processing for zonal statistics computation.

Uses multiprocessing to speed up district-level raster extraction
across multiple years/layers.
"""

from src.logging_config import get_pipeline_logger
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats

from src import config

log = get_pipeline_logger(__name__)


def _zonal_stats_for_year(args):
    """Worker function: compute zonal stats for one year.

    Args:
        args: Tuple of (year, raster_path, gdf_path, stats_list).

    Returns:
        List of dicts with district-level statistics.
    """
    import geopandas as gpd

    year, raster_path, gdf_path, stats_list = args

    if not os.path.exists(raster_path):
        return []

    gdf = gpd.read_file(gdf_path)
    results = zonal_stats(gdf, raster_path, stats=stats_list,
                          nodata=np.nan, all_touched=True)

    rows = []
    for i, row_stats in enumerate(results):
        entry = {"district": gdf.iloc[i]["district"], "year": year}
        entry.update(row_stats)
        rows.append(entry)

    return rows


def parallel_zonal_extraction(year_raster_map, gdf_path, stats_list=None,
                              max_workers=None):
    """Run zonal statistics in parallel across multiple years.

    Args:
        year_raster_map: Dict mapping year -> raster file path.
        gdf_path: Path to district shapefile.
        stats_list: List of stats to compute (default: mean, median, std, count).
        max_workers: Number of parallel workers (default: CPU count - 1).

    Returns:
        DataFrame with columns [district, year, <stats>].
    """
    if stats_list is None:
        stats_list = ["mean", "median", "std", "count"]

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) - 1)

    tasks = [
        (year, rpath, gdf_path, stats_list)
        for year, rpath in sorted(year_raster_map.items())
        if os.path.exists(rpath)
    ]

    if not tasks:
        log.warning("No valid raster paths for parallel extraction")
        return pd.DataFrame()

    log.info("Running parallel zonal stats: %d years, %d workers",
             len(tasks), max_workers)

    all_rows = []

    if max_workers == 1 or len(tasks) == 1:
        for task in tasks:
            rows = _zonal_stats_for_year(task)
            all_rows.extend(rows)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_zonal_stats_for_year, t): t[0]
                       for t in tasks}
            for future in as_completed(futures):
                year = futures[future]
                try:
                    rows = future.result()
                    all_rows.extend(rows)
                    log.debug("Completed zonal stats for year %d (%d districts)",
                              year, len(rows))
                except Exception as e:
                    log.error("Zonal stats failed for year %d: %s", year, e)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["district", "year"]).reset_index(drop=True)

    log.info("Parallel extraction complete: %d rows", len(df))
    return df


def parallel_raster_subset(year_raster_map, bbox, output_dir, max_workers=None):
    """Subset multiple rasters to bbox in parallel.

    Args:
        year_raster_map: Dict mapping year -> raster file path.
        bbox: Dict with west, south, east, north.
        output_dir: Directory for subsetted rasters.
        max_workers: Number of parallel workers.

    Returns:
        Dict mapping year -> subsetted raster path.
    """
    os.makedirs(output_dir, exist_ok=True)

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) - 1)

    def _subset_one(args):
        year, src_path = args
        out_path = os.path.join(output_dir, f"subset_{year}.tif")
        if os.path.exists(out_path):
            return year, out_path

        with rasterio.open(src_path) as src:
            window = rasterio.windows.from_bounds(
                bbox["west"], bbox["south"], bbox["east"], bbox["north"],
                transform=src.transform
            )
            data = src.read(1, window=window)
            transform = rasterio.windows.transform(window, src.transform)

            meta = src.meta.copy()
            meta.update({
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": transform,
            })

            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(data, 1)

        return year, out_path

    tasks = [(y, p) for y, p in sorted(year_raster_map.items())
             if os.path.exists(p)]

    results = {}
    if max_workers == 1 or len(tasks) == 1:
        for task in tasks:
            year, out_path = _subset_one(task)
            results[year] = out_path
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_subset_one, t): t[0] for t in tasks}
            for future in as_completed(futures):
                year = futures[future]
                try:
                    y, out_path = future.result()
                    results[y] = out_path
                except Exception as e:
                    log.error("Subset failed for year %d: %s", year, e)

    log.info("Parallel subsetting complete: %d rasters", len(results))
    return results
