"""Render consistent per-year animation frames for the public-site hero.

Why not reuse ``src.outputs.visualizations`` generators
-------------------------------------------------------
The matplotlib generators in ``visualizations.py`` are tuned for print:
they include per-figure colorbars (vertical), titles, and annotation boxes.
On the web, toggling between animations produces jarring dimension jumps
because some frames reserve horizontal space for a colorbar and others
don't — Maharashtra visibly shrinks when you switch to the differential
view.

This module renders purpose-built frames that are:

* all exactly the same pixel dimensions,
* all pure data + a faint year badge, no colorbars or titles baked in
  (the Vue hero paints the legend/title in clean page typography),
* optimised for a large hero slot with the state filling the frame.

Output pipeline:

1. Per animation kind, iterate over years and render an RGB PNG per year.
2. Stitch the frames into an animated GIF with Pillow's adaptive palette.
3. Emit ``animations.json`` so the frontend can pick each kind up by id.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import Normalize, PowerNorm
from PIL import Image

from src import config, viirs_utils
from src.logging_config import get_pipeline_logger

log = get_pipeline_logger(__name__)


# Consistent frame canvas. 4:3 so the hero tile stays stable across toggles.
FRAME_W = 1280
FRAME_H = 960
DPI = 100


# ── Animation specifications ─────────────────────────────────────────────


@dataclass
class AnimationSpec:
    id: str
    label: str
    description: str
    renderer: str                # one of "sprawl" / "differential" / "darkness"
    legend: dict                 # rendered separately by the frontend


ANIMATIONS: list[AnimationSpec] = [
    AnimationSpec(
        id="sprawl",
        label="Urban sprawl",
        description=(
            "Yellow pixels are locations bright enough to qualify as 'lit' "
            "(> 1.5 nW/cm²/sr). Watch the web of urban light expand across "
            "Maharashtra as villages and highways electrify."
        ),
        renderer="sprawl",
        legend={
            "type": "threshold",
            "colors": [
                {"color": "#000000", "label": "Unlit (< 1.5 nW/cm²/sr)"},
                {"color": "#f2d300", "label": "Lit (≥ 1.5 nW/cm²/sr)"},
            ],
        },
    ),
    AnimationSpec(
        id="differential",
        label="New light vs 2012",
        description=(
            "Each year compared to the 2012 baseline. Red = brighter than "
            "2012, blue = darker. Red has been winning decisively across the "
            "entire state for over a decade."
        ),
        renderer="differential",
        legend={
            "type": "gradient",
            "colormap": "RdBu_r",
            "min": -5,
            "max": 5,
            "unit": "Δ nW/cm²/sr",
        },
    ),
    AnimationSpec(
        id="darkness",
        label="Erosion of darkness",
        description=(
            "Teal pixels are pristine dark-sky reservoirs (< 0.25 nW/cm²/sr). "
            "Watch them shrink as artificial light reaches deeper into rural "
            "Maharashtra year after year."
        ),
        renderer="darkness",
        legend={
            "type": "threshold",
            "colors": [
                {"color": "#111111", "label": "Lit (≥ 0.25 nW/cm²/sr)"},
                {"color": "#2bb2a1", "label": "Pristine dark (< 0.25 nW/cm²/sr)"},
            ],
        },
    ),
]


# ── Raster loading ───────────────────────────────────────────────────────


def _load_year(output_dir: str, year: int) -> tuple[np.ndarray, list[float]] | tuple[None, None]:
    """Load + DBS-correct a year's median raster. Returns (array, extent)."""
    path = os.path.join(output_dir, "subsets", str(year), f"maharashtra_median_{year}.tif")
    if not os.path.exists(path):
        return None, None
    with rasterio.open(path) as src:
        data = src.read(1).astype("float32")
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    data = viirs_utils.apply_dynamic_background_subtraction(data, year=year)
    return data, extent


# ── Frame rendering ──────────────────────────────────────────────────────


def _new_figure():
    """Consistent frame canvas with a fully transparent background.

    The frames are designed to be composited *on top of* the live map in the
    browser — the map supplies the background, state / district outlines and
    any UI chrome, so the frame itself is pure data over an alpha channel.
    """
    fig = plt.figure(figsize=(FRAME_W / DPI, FRAME_H / DPI), dpi=DPI)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor((0, 0, 0, 0))
    ax.set_axis_off()
    return fig, ax


def _save_frame(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # transparent=True preserves the RGBA data; pad_inches=0 keeps the frame
    # exactly FRAME_W × FRAME_H so each year lands on the same pixel grid.
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def _render_sprawl(data: np.ndarray, extent: list[float], year: int, out_path: str,
                   threshold: float = config.SPRAWL_THRESHOLD_NW) -> None:
    fig, ax = _new_figure()
    rows, cols = data.shape
    img = np.zeros((rows, cols, 4), dtype="float32")
    lit = np.isfinite(data) & (data >= threshold)
    img[lit] = [0.95, 0.83, 0.0, 1.0]
    ax.imshow(img, extent=extent, origin="upper", zorder=1)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    _save_frame(fig, out_path)


def _render_darkness(data: np.ndarray, extent: list[float], year: int, out_path: str,
                     threshold: float = config.DARKNESS_THRESHOLD_NW) -> None:
    fig, ax = _new_figure()
    rows, cols = data.shape
    img = np.zeros((rows, cols, 4), dtype="float32")
    finite = np.isfinite(data)
    lit = finite & (data >= threshold)
    dark = finite & ~lit
    # Lit pixels are almost invisible so the map's basemap shows through;
    # dark-sky reservoirs are vivid teal.
    img[lit] = [0.09, 0.09, 0.09, 0.35]
    img[dark] = [0.17, 0.70, 0.63, 0.92]
    ax.imshow(img, extent=extent, origin="upper", zorder=1)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    _save_frame(fig, out_path)


def _render_differential(data: np.ndarray, baseline: np.ndarray, extent: list[float],
                         year: int, out_path: str) -> None:
    fig, ax = _new_figure()
    diff = data - baseline
    norm = Normalize(vmin=-5.0, vmax=5.0)
    import matplotlib
    cmap = matplotlib.colormaps["RdBu_r"] if hasattr(matplotlib, "colormaps") else plt.get_cmap("RdBu_r")
    finite = np.isfinite(diff)
    rgba = cmap(norm(np.where(finite, diff, 0.0)))
    # Magnitude-driven alpha so no-change pixels fade out and only real
    # growth / loss pops on top of the map.
    mag = np.clip(np.abs(np.where(finite, diff, 0.0) / 5.0), 0.0, 1.0) ** 0.55
    alpha = np.where(finite, 0.05 + 0.9 * mag, 0.0)
    rgba[..., 3] = alpha
    ax.imshow(rgba, extent=extent, origin="upper", zorder=1)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    _save_frame(fig, out_path)


# ── Entry point ──────────────────────────────────────────────────────────


def export_web_animations(
    output_dir: str,
    out_dir: str,
    shapefile_path: str | None = None,   # signature kept for pipeline_runner compat
    years: list[int] | None = None,
    duration_ms: int = 650,
) -> dict:
    """Emit per-year transparent frame PNGs + manifest.

    Each ``animations/<kind>/<year>.png`` is a georeferenced, transparent
    raster the frontend map composites on top of its basemap layer —
    advancing through years by swapping the active frame in a canvas source.
    The Maharashtra bounding box is shared across every frame and every
    animation kind so the MapLibre ``image``/``canvas`` source can use one
    set of corner coordinates for everything.
    """
    animations_dir = os.path.join(out_dir, "animations")
    os.makedirs(animations_dir, exist_ok=True)

    if years is None:
        subsets = os.path.join(output_dir, "subsets")
        years = sorted(
            int(d) for d in os.listdir(subsets)
            if d.isdigit() and os.path.exists(
                os.path.join(subsets, d, f"maharashtra_median_{d}.tif")
            )
        )
    if not years:
        raise FileNotFoundError(f"No yearly rasters under {output_dir}/subsets/")

    # Preload baseline for differential animation.
    baseline_year = min(years)
    baseline_data, baseline_extent = _load_year(output_dir, baseline_year)
    if baseline_data is None:
        raise FileNotFoundError(f"Baseline year {baseline_year} raster missing")

    manifest_layers: list[dict] = []
    for spec in ANIMATIONS:
        log.info("Animation %r: rendering %d transparent frames", spec.id, len(years))
        kind_dir = os.path.join(animations_dir, spec.id)
        os.makedirs(kind_dir, exist_ok=True)
        frame_entries: list[dict] = []
        extent_ref: list[float] | None = None
        for year in years:
            data, extent = _load_year(output_dir, year)
            if data is None:
                log.warning("Year %d data missing — skipping", year)
                continue
            if extent_ref is None:
                extent_ref = extent
            fp = os.path.join(kind_dir, f"{year}.png")
            if spec.renderer == "sprawl":
                _render_sprawl(data, extent, year, fp)
            elif spec.renderer == "darkness":
                _render_darkness(data, extent, year, fp)
            elif spec.renderer == "differential":
                _render_differential(data, baseline_data, extent, year, fp)
            else:
                raise ValueError(f"unknown renderer: {spec.renderer}")
            frame_entries.append({
                "year": year,
                "url": f"animations/{spec.id}/{year}.png",
            })

        if not frame_entries:
            log.warning("Animation %r: no frames produced", spec.id)
            continue

        total_bytes = sum(
            os.path.getsize(os.path.join(out_dir, f["url"])) for f in frame_entries
        )
        log.info("Animation %r: %d frames, %.1f MB total",
                 spec.id, len(frame_entries), total_bytes / 1e6)

        # Bounds: share the raster extent so frontend can reuse coordinates.
        bounds = list(extent_ref or (0, 0, 0, 0))  # [west, east, south, north] from rasterio
        # rasterio returns (left, right, bottom, top) in extent-style;
        # normalise to [west, south, east, north] for consistency with
        # rasters.json.
        bounds_wsen = [bounds[0], bounds[2], bounds[1], bounds[3]]

        manifest_layers.append({
            "id": spec.id,
            "label": spec.label,
            "description": spec.description,
            "kind": "animated",
            "bounds": bounds_wsen,
            "frames": frame_entries,
            "year_range": [years[0], years[-1]],
            "frame_duration_ms": duration_ms,
            "size_bytes": total_bytes,
            "width": FRAME_W,
            "height": FRAME_H,
            "legend": spec.legend,
        })

    manifest_path = os.path.join(out_dir, "animations.json")
    with open(manifest_path, "w") as f:
        json.dump({"animations": manifest_layers}, f, indent=2)

    summary = {
        "animations_written": len(manifest_layers),
        "manifest": manifest_path,
        "out_dir": os.path.abspath(animations_dir),
    }
    log.info("Web animations exported: %s", summary)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate animated GIFs for the public site hero.")
    parser.add_argument("--output-dir", default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--out-dir", default="web-export")
    parser.add_argument("--shapefile-path", default=config.DEFAULT_SHAPEFILE_PATH)
    parser.add_argument("--years", default=None, help="'2012-2024' or '2012,2015,…'")
    parser.add_argument("--duration-ms", type=int, default=650)
    args = parser.parse_args()

    if args.years:
        if "-" in args.years:
            a, b = args.years.split("-"); ylist = list(range(int(a), int(b) + 1))
        else:
            ylist = [int(y) for y in args.years.split(",")]
    else:
        ylist = None

    export_web_animations(
        args.output_dir, args.out_dir, args.shapefile_path,
        years=ylist, duration_ms=args.duration_ms,
    )
