"""
Intelligent caching system for expensive raster operations.

Avoids redundant recomputation of zonal statistics and raster subsetting
by hashing input parameters and checking for cached results.
"""

import hashlib
import json
import logging
import os
import time

import pandas as pd

log = logging.getLogger(__name__)

# Default cache directory
_DEFAULT_CACHE_DIR = os.path.join("output", ".cache")


class CacheManager:
    """File-based cache for expensive operations.

    Cache keys are SHA-256 hashes of operation parameters.
    Each cached result is stored as a CSV alongside a JSON metadata file.
    """

    def __init__(self, cache_dir=None, max_age_days=30):
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self.max_age_days = max_age_days
        os.makedirs(self.cache_dir, exist_ok=True)
        self._stats = {"hits": 0, "misses": 0}

    def _make_key(self, operation, **params):
        """Generate deterministic cache key from operation name and params."""
        payload = {"op": operation}
        for k, v in sorted(params.items()):
            payload[k] = str(v)
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cache_paths(self, key):
        csv_path = os.path.join(self.cache_dir, f"{key}.csv")
        meta_path = os.path.join(self.cache_dir, f"{key}.meta.json")
        return csv_path, meta_path

    def get(self, operation, **params):
        """Retrieve cached DataFrame, or None if not cached/expired."""
        key = self._make_key(operation, **params)
        csv_path, meta_path = self._cache_paths(key)

        if not os.path.exists(csv_path) or not os.path.exists(meta_path):
            self._stats["misses"] += 1
            return None

        with open(meta_path) as f:
            meta = json.load(f)

        age_days = (time.time() - meta.get("created", 0)) / 86400
        if age_days > self.max_age_days:
            log.debug("Cache expired for %s (%.1f days old)", operation, age_days)
            self._stats["misses"] += 1
            return None

        log.debug("Cache hit: %s [%s]", operation, key)
        self._stats["hits"] += 1
        return pd.read_csv(csv_path)

    def put(self, operation, df, **params):
        """Store a DataFrame result in cache."""
        key = self._make_key(operation, **params)
        csv_path, meta_path = self._cache_paths(key)

        df.to_csv(csv_path, index=False)
        meta = {
            "operation": operation,
            "params": {k: str(v) for k, v in params.items()},
            "created": time.time(),
            "rows": len(df),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        log.debug("Cached %d rows for %s [%s]", len(df), operation, key)

    def invalidate(self, operation=None):
        """Remove cached results. If operation is None, clear entire cache."""
        removed = 0
        for fname in os.listdir(self.cache_dir):
            fpath = os.path.join(self.cache_dir, fname)
            if operation is not None:
                if fname.endswith(".meta.json"):
                    with open(fpath) as f:
                        meta = json.load(f)
                    if meta.get("operation") != operation:
                        continue
                    key = fname.replace(".meta.json", "")
                    csv_path = os.path.join(self.cache_dir, f"{key}.csv")
                    if os.path.exists(csv_path):
                        os.remove(csv_path)
                    os.remove(fpath)
                    removed += 1
            else:
                os.remove(fpath)
                removed += 1

        log.info("Invalidated %d cache entries", removed)

    def get_stats(self):
        """Return cache hit/miss statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total * 100 if total > 0 else 0
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate_pct": round(hit_rate, 1),
        }


def cached_zonal_stats(cache, operation_name, compute_fn, **params):
    """Wrapper: check cache first, compute and cache if miss.

    Args:
        cache: CacheManager instance (or None to skip caching).
        operation_name: Descriptive operation string.
        compute_fn: Callable that returns a DataFrame.
        **params: Parameters used for cache key and passed to compute_fn.

    Returns:
        DataFrame result.
    """
    if cache is not None:
        cached = cache.get(operation_name, **params)
        if cached is not None:
            return cached

    df = compute_fn()

    if cache is not None:
        cache.put(operation_name, df, **params)

    return df
