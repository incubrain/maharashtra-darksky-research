"""
Reusable HTTP API utilities with rate limiting, retries, and checkpointing.

Extracted from geocoding scripts to provide a shared foundation for any
external API calls (GBIF, Nominatim, etc.) in the pipeline.
"""

import csv
import os
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def make_session(
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: tuple = (429, 500, 502, 503, 504),
    user_agent: str = "maharashtra-darksky-research/1.0",
) -> requests.Session:
    """Create a requests.Session with automatic retry and backoff.

    Parameters
    ----------
    max_retries : int
        Total retry attempts per request.
    backoff_factor : float
        Exponential backoff multiplier (0.5 → 0.5s, 1s, 2s, ...).
    status_forcelist : tuple
        HTTP status codes that trigger a retry.
    user_agent : str
        User-Agent header value.

    Returns
    -------
    requests.Session
        Configured session with retry adapter mounted.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(status_forcelist),
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class RateLimiter:
    """Simple rate limiter that enforces a minimum delay between calls.

    Usage::

        limiter = RateLimiter(calls_per_second=5)
        for item in items:
            limiter.wait()
            do_api_call(item)
    """

    def __init__(self, calls_per_second: float = 5.0):
        self.min_interval = 1.0 / calls_per_second
        self._last_call = 0.0

    def wait(self):
        """Block until enough time has elapsed since the last call."""
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


class CSVCheckpointer:
    """Append-mode CSV writer for incremental checkpoint saves.

    Handles:
    - Creating the file with headers on first use.
    - Appending rows without rewriting the whole file.
    - Loading existing rows for idempotent re-runs.

    Usage::

        cp = CSVCheckpointer("output.csv", fieldnames=["id", "name", "status"])
        existing = cp.load_existing(key_column="id")
        for item in items:
            if item["id"] in existing:
                continue
            result = process(item)
            cp.write_row(result)
    """

    def __init__(self, path: str, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        self._file = None
        self._writer = None

    def load_existing(self, key_column: str) -> set:
        """Load existing rows and return set of key_column values."""
        if not os.path.isfile(self.path):
            return set()
        keys = set()
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get(key_column, "")
                if val:
                    keys.add(val)
        return keys

    def load_existing_df(self):
        """Load existing CSV as a pandas DataFrame (returns None if missing)."""
        import pandas as pd

        if not os.path.isfile(self.path):
            return None
        return pd.read_csv(self.path)

    def open(self):
        """Open the CSV for appending. Writes header if file is new/empty."""
        is_new = not os.path.isfile(self.path) or os.path.getsize(self.path) == 0
        self._file = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if is_new:
            self._writer.writeheader()
            self._file.flush()

    def write_row(self, row: dict):
        """Append a single row and flush immediately."""
        if self._writer is None:
            raise RuntimeError("CSVCheckpointer not opened. Call .open() first.")
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        """Close the file handle."""
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
