"""
Incremental output update system.

Detects which outputs are stale (missing or older than inputs) and
regenerates only those, avoiding full recomputation when only one year
of data changes.
"""

import hashlib
import json
import logging
import os
import time

import pandas as pd

log = logging.getLogger(__name__)


def _file_mtime(path):
    """Return file modification time, or 0 if not found."""
    if os.path.exists(path):
        return os.path.getmtime(path)
    return 0.0


def _file_hash(path, chunk_size=65536):
    """Compute SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


class OutputTracker:
    """Tracks input-output dependencies for incremental updates.

    Maintains a manifest of which outputs depend on which inputs,
    and their modification times, to determine what needs regeneration.
    """

    def __init__(self, manifest_path=None):
        self.manifest_path = manifest_path or os.path.join("output", ".incremental_manifest.json")
        self._manifest = self._load_manifest()

    def _load_manifest(self):
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path) as f:
                return json.load(f)
        return {"outputs": {}}

    def _save_manifest(self):
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def register_output(self, output_path, input_paths):
        """Record an output and its input dependencies."""
        input_info = {}
        for ip in input_paths:
            if os.path.exists(ip):
                input_info[ip] = {
                    "mtime": _file_mtime(ip),
                    "hash": _file_hash(ip),
                }
        self._manifest["outputs"][output_path] = {
            "created": time.time(),
            "inputs": input_info,
        }
        self._save_manifest()

    def is_stale(self, output_path, input_paths=None):
        """Check if an output needs regeneration.

        Returns True if:
        - Output file doesn't exist
        - Output is not in manifest
        - Any input file is newer than the recorded state
        - Any input file hash has changed
        """
        if not os.path.exists(output_path):
            return True

        entry = self._manifest.get("outputs", {}).get(output_path)
        if entry is None:
            return True

        if input_paths is None:
            input_paths = list(entry.get("inputs", {}).keys())

        recorded_inputs = entry.get("inputs", {})
        for ip in input_paths:
            if not os.path.exists(ip):
                continue
            recorded = recorded_inputs.get(ip)
            if recorded is None:
                return True
            if _file_mtime(ip) > recorded.get("mtime", 0):
                current_hash = _file_hash(ip)
                if current_hash != recorded.get("hash"):
                    return True

        return False

    def stale_outputs(self, output_input_map):
        """Given a dict of {output_path: [input_paths]}, return stale ones.

        Args:
            output_input_map: Dict mapping output file to list of input files.

        Returns:
            List of output paths that need regeneration.
        """
        stale = []
        for output_path, input_paths in output_input_map.items():
            if self.is_stale(output_path, input_paths):
                stale.append(output_path)
        return stale

    def summary(self):
        """Return summary of tracked outputs."""
        total = len(self._manifest.get("outputs", {}))
        stale_count = sum(
            1 for op in self._manifest.get("outputs", {})
            if self.is_stale(op)
        )
        return {
            "total_outputs": total,
            "stale_outputs": stale_count,
            "up_to_date": total - stale_count,
        }


def incremental_pipeline_run(tracker, output_input_map, generators):
    """Run only the stale parts of a pipeline.

    Args:
        tracker: OutputTracker instance.
        output_input_map: Dict of {output_path: [input_paths]}.
        generators: Dict of {output_path: callable} that generates the output.

    Returns:
        Dict with counts of regenerated vs skipped outputs.
    """
    stale = tracker.stale_outputs(output_input_map)
    skipped = 0
    regenerated = 0

    for output_path, gen_fn in generators.items():
        if output_path not in stale:
            log.debug("Skipping up-to-date: %s", output_path)
            skipped += 1
            continue

        log.info("Regenerating: %s", output_path)
        try:
            gen_fn()
            input_paths = output_input_map.get(output_path, [])
            tracker.register_output(output_path, input_paths)
            regenerated += 1
        except Exception as e:
            log.error("Failed to regenerate %s: %s", output_path, e)

    log.info("Incremental update: %d regenerated, %d skipped",
             regenerated, skipped)
    return {"regenerated": regenerated, "skipped": skipped}
