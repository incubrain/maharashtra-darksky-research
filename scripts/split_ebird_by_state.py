#!/usr/bin/env python3
"""
split_ebird_by_state.py

Splits a large eBird/GBIF occurrence TSV into per-state gzip files in a single
streaming pass. Writes directly to .tsv.gz files so intermediate uncompressed
data never hits disk — critical when disk space is limited.

Memory usage: ~50MB (only holds open gzip file handles + line buffer).
Disk usage: ~7-8GB total (compressed output), not 32GB.

Output structure:
    <output_dir>/<state>_modern.tsv.gz   - records from 1991 onwards
    <output_dir>/<state>_archive.tsv.gz  - records before 1991
    <output_dir>/_header.tsv             - column headers

Usage:
    python3 scripts/split_ebird_by_state.py <input.zip> <output_dir>
"""

import gzip
import os
import re
import subprocess
import sys
import time


def sanitize_state(name: str) -> str:
    """Convert state name to filesystem-safe lowercase with underscores."""
    s = name.lower().strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-z0-9_-]", "", s)
    return s


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.zip> <output_dir>")
        sys.exit(1)

    input_zip = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isfile(input_zip):
        print(f"ERROR: Input file not found: {input_zip}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    zip_size = os.path.getsize(input_zip) / (1024**3)
    print(f"=== eBird State Splitter (Streaming Gzip) ===")
    print(f"Input:  {input_zip} ({zip_size:.1f} GB)")
    print(f"Output: {output_dir}")
    print()

    # Stream from unzip -p (avoids extracting 32GB to disk)
    proc = subprocess.Popen(
        ["unzip", "-p", input_zip],
        stdout=subprocess.PIPE,
        bufsize=1024 * 1024,  # 1MB buffer
    )

    # Read header
    header_bytes = proc.stdout.readline()
    header_line = header_bytes.decode("utf-8", errors="replace").rstrip("\n\r")

    # Save header
    header_path = os.path.join(output_dir, "_header.tsv")
    with open(header_path, "w") as f:
        f.write(header_line + "\n")

    cols = header_line.split("\t")
    print(f"Header: {len(cols)} columns")

    # Find column indices
    try:
        state_idx = cols.index("stateProvince")
        year_idx = cols.index("year")
    except ValueError as e:
        print(f"ERROR: Required column not found: {e}")
        proc.kill()
        sys.exit(1)

    print(f"stateProvince: column {state_idx + 1}, year: column {year_idx + 1}")
    print()

    # Track open gzip writers: key -> gzip file object
    writers = {}  # file_key -> gzip.GzipFile
    counts = {}   # file_key -> row count
    state_set = set()

    header_encoded = (header_line + "\n").encode("utf-8")

    start = time.time()
    row_count = 0
    skip_count = 0

    print("Streaming split + compress (progress every 5M rows)...")
    print()

    for raw_line in proc.stdout:
        row_count += 1

        if row_count % 5_000_000 == 0:
            elapsed = time.time() - start
            rate = row_count / elapsed
            eta = (60_000_000 - row_count) / rate if rate > 0 else 0
            print(
                f"  {row_count // 1_000_000}M rows | "
                f"{len(state_set)} states | "
                f"{elapsed:.0f}s elapsed | "
                f"~{rate:.0f} rows/s | "
                f"ETA ~{eta:.0f}s",
                flush=True,
            )

        # Parse only the columns we need by splitting the raw bytes
        # This avoids decoding the entire line for performance
        fields = raw_line.split(b"\t")

        if len(fields) <= max(state_idx, year_idx):
            skip_count += 1
            continue

        state_raw = fields[state_idx].decode("utf-8", errors="replace").strip()
        if not state_raw:
            skip_count += 1
            continue

        year_raw = fields[year_idx].strip()
        try:
            year = int(year_raw)
        except (ValueError, TypeError):
            year = 0

        safe_state = sanitize_state(state_raw)
        suffix = "modern" if year >= 1991 else "archive"
        file_key = f"{safe_state}_{suffix}"

        if file_key not in writers:
            filepath = os.path.join(output_dir, f"{file_key}.tsv.gz")
            # compresslevel=1 for speed (still ~5x compression on TSV)
            gz = gzip.open(filepath, "wb", compresslevel=1)
            gz.write(header_encoded)
            writers[file_key] = gz
            counts[file_key] = 0
            state_set.add(safe_state)

        writers[file_key].write(raw_line)
        counts[file_key] += 1

    # Close all writers
    for gz in writers.values():
        gz.close()

    proc.wait()
    elapsed = time.time() - start

    print()
    print(f"=== Split Complete ===")
    print(f"Total data rows: {row_count:,}")
    print(f"Skipped (no state/short): {skip_count:,}")
    print(f"Unique states: {len(state_set)}")
    print(f"Output files: {len(writers)}")
    print(f"Time: {elapsed:.0f}s ({row_count / elapsed:.0f} rows/s)")
    print()

    # Summary sorted by count
    print("Per-file row counts (top 20):")
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    for key, cnt in sorted_counts[:20]:
        filepath = os.path.join(output_dir, f"{key}.tsv.gz")
        size_mb = os.path.getsize(filepath) / (1024**2)
        print(f"  {key}: {cnt:>10,} rows  ({size_mb:.1f} MB)")

    if len(sorted_counts) > 20:
        print(f"  ... and {len(sorted_counts) - 20} more files")

    print()
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if f.endswith(".gz")
    )
    print(f"Total compressed output: {total_size / (1024**3):.2f} GB")
    print("Done!")


if __name__ == "__main__":
    main()
