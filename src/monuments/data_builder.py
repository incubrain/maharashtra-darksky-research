"""Build per-district monument CSV files from the constants data."""

import csv
import os

from src.monuments.constants import MONUMENTS

OUTPUT_DIR = "data/monuments"


def normalize_district(district: str) -> str:
    """Normalize district name for filename."""
    return district.lower().replace(" ", "_")


def build_monument_csvs(output_dir: str = OUTPUT_DIR):
    """Write per-district CSVs from the MONUMENTS list."""
    os.makedirs(output_dir, exist_ok=True)

    by_district: dict[str, list] = {}
    for row in MONUMENTS:
        name, mtype, place, taluka, district, notif_status = row
        by_district.setdefault(district, []).append(row)

    total = 0
    for district in sorted(by_district.keys()):
        rows = by_district[district]
        filename = f"{normalize_district(district)}.csv"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "name", "monument_type", "place", "taluka", "district",
                "notification_status", "lat", "lon", "geocode_status",
            ])
            for name, mtype, place, taluka, dist, notif in rows:
                writer.writerow([
                    name, mtype, place, taluka, dist, notif,
                    "", "", "pending",
                ])

        total += len(rows)
        print(f"  {filename}: {len(rows)} monuments")

    print(f"\nTotal: {total} monuments across {len(by_district)} districts")
