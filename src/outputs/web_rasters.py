"""Generate transparent RGBA PNG overlays for the web map.

Unlike the publication-quality matplotlib figures in ``visualizations.py`` —
which bake titles, axes, colorbars and black backgrounds directly into the PNG —
these outputs are *data-only* images meant to be placed on a live map as
`image` sources. Each pixel carries the colormapped radiance value; the rest
of the image is alpha=0 so the basemap shows through.

The output extent matches the source raster bbox exactly (all 13 yearly
VIIRS composites share the same grid), so the web client can place the image
with fixed corner coordinates without worrying about re-projection.

Three layers are produced:

``latest_radiance.png``  — 2024 median radiance, inferno colormap, PowerNorm.
``change_12yr.png``      — 2024 − 2012 median, RdBu_r diverging, [-5, 5].
``annual_trend.png``     — Pixel-wise OLS slope, coolwarm, [-2, 2] nW/yr.

A ``rasters.json`` manifest describes each layer (id, label, url, bounds,
legend metadata) so the frontend can render the layer picker + colour-bar
legend without hard-coding.
"""

from __future__ import annotations

import json
import os

import numpy as np
import rasterio
from matplotlib import cm
from matplotlib.colors import PowerNorm, Normalize
from PIL import Image

from src.logging_config import get_pipeline_logger

log = get_pipeline_logger(__name__)


# ── Raster loading helpers ────────────────────────────────────────────────


def _subset_path(output_dir: str, year: int) -> str:
    return os.path.join(output_dir, "subsets", str(year), f"maharashtra_median_{year}.tif")


def _load_year(output_dir: str, year: int) -> tuple[np.ndarray, rasterio.Affine, tuple[float, float, float, float]]:
    """Load a year's median raster. Returns (array, transform, bounds)."""
    with rasterio.open(_subset_path(output_dir, year)) as src:
        arr = src.read(1).astype("float32")
        # Nodata → NaN
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        return arr, src.transform, tuple(src.bounds)


# ── PNG rendering ─────────────────────────────────────────────────────────


def _render_rgba(
    array: np.ndarray,
    cmap_name: str,
    norm,
    alpha_mode: str = "magnitude",
    alpha_floor: float = 0.05,
    alpha_ceiling: float = 0.92,
    alpha_gamma: float = 0.7,
) -> Image.Image:
    """Apply colormap and a magnitude-driven alpha mask, return an RGBA PIL Image.

    The alpha channel lets the basemap show through in low-information pixels
    (e.g. rural areas on the radiance layer, no-change pixels on diverging
    layers) while keeping the high-signal pixels (cities, growth hotspots)
    prominent.

    alpha_mode
        ``'magnitude'``  — alpha scales with normalised value (0 → transparent,
        1 → opaque). Best for one-sided scales like radiance.
        ``'diverging'``  — alpha scales with |value − 0.5| * 2, so the middle
        of the colormap (no-change) is transparent and the extremes are opaque.
        Use for diverging RdBu/coolwarm palettes.
        ``'solid'``      — full opacity for all finite pixels.
    alpha_floor, alpha_ceiling
        Clamp the alpha range; floor > 0 keeps a hint of colour even in
        low-signal pixels so the dataset remains discoverable.
    alpha_gamma
        Power-curve shaping for perceptual contrast.
    """
    # Matplotlib 3.7+ deprecated ``cm.get_cmap``; use the new accessor.
    cmap = cm.colormaps[cmap_name] if hasattr(cm, "colormaps") else cm.get_cmap(cmap_name)
    finite = np.isfinite(array)
    safe = np.where(finite, array, 0.0)
    normalised = np.clip(norm(safe), 0.0, 1.0)
    rgba = (cmap(normalised) * 255).astype(np.uint8)

    if alpha_mode == "solid":
        alpha01 = np.ones_like(normalised, dtype="float32")
    elif alpha_mode == "diverging":
        alpha01 = np.clip(np.abs(normalised - 0.5) * 2.0, 0.0, 1.0)
    else:  # magnitude
        alpha01 = normalised.astype("float32")

    alpha01 = alpha01 ** alpha_gamma
    alpha01 = alpha_floor + (alpha_ceiling - alpha_floor) * alpha01
    alpha01 = np.where(finite, alpha01, 0.0)
    rgba[..., 3] = (alpha01 * 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _save_png(img: Image.Image, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, "PNG", optimize=True)


# ── Layer generators ──────────────────────────────────────────────────────


def render_latest_radiance(years: list[int], output_dir: str, out_dir: str) -> dict:
    """2024 median radiance, inferno colormap with PowerNorm gamma=0.35."""
    latest_year = max(years)
    arr, _transform, bounds = _load_year(output_dir, latest_year)
    # Clip upper end — a handful of extreme pixels (gas flares, saturation) dominate otherwise.
    finite_vals = arr[np.isfinite(arr)]
    vmax = float(np.nanpercentile(finite_vals, 99.5)) if finite_vals.size else 50.0
    norm = PowerNorm(gamma=0.35, vmin=0.0, vmax=vmax)
    img = _render_rgba(arr, "inferno", norm, alpha_mode="magnitude",
                       alpha_floor=0.02, alpha_ceiling=0.95, alpha_gamma=0.6)
    out_path = os.path.join(out_dir, "latest_radiance.png")
    _save_png(img, out_path)
    return {
        "id": "latest",
        "label": f"Latest radiance ({latest_year})",
        "description": "Median annual VIIRS radiance. Brighter = more artificial light at night.",
        "url": "rasters/latest_radiance.png",
        "bounds": list(bounds),
        "year": latest_year,
        "legend": {
            "colormap": "inferno",
            "min": 0.0,
            "max": round(vmax, 2),
            "scale": "power",
            "gamma": 0.35,
            "unit": "nW/cm²/sr",
        },
    }


def render_change_12yr(years: list[int], output_dir: str, out_dir: str) -> dict:
    """Baseline-to-latest difference, diverging colormap."""
    baseline = min(years)
    latest = max(years)
    arr_b, _tb, bounds = _load_year(output_dir, baseline)
    arr_l, _tl, _bl = _load_year(output_dir, latest)
    # Both grids share the exact same shape/affine (verified).
    diff = arr_l - arr_b
    # Symmetric range; ±5 nW/cm²/sr covers the bulk of the distribution while
    # saturating obvious hotspots.
    norm = Normalize(vmin=-5.0, vmax=5.0)
    img = _render_rgba(diff, "RdBu_r", norm, alpha_mode="diverging",
                       alpha_floor=0.02, alpha_ceiling=0.9, alpha_gamma=0.55)
    out_path = os.path.join(out_dir, "change_12yr.png")
    _save_png(img, out_path)
    return {
        "id": "change",
        "label": f"Change {baseline} → {latest}",
        "description": f"Pixel-by-pixel radiance difference. Red = brighter than {baseline}, blue = darker.",
        "url": "rasters/change_12yr.png",
        "bounds": list(bounds),
        "baseline_year": baseline,
        "latest_year": latest,
        "legend": {
            "colormap": "RdBu_r",
            "min": -5.0,
            "max": 5.0,
            "unit": "nW/cm²/sr",
        },
    }


def render_annual_trend(years: list[int], output_dir: str, out_dir: str) -> dict:
    """Pixel-wise OLS slope (nW/cm²/sr per year)."""
    stack = []
    ref_bounds = None
    for year in years:
        arr, _, bounds = _load_year(output_dir, year)
        if ref_bounds is None:
            ref_bounds = bounds
        stack.append(arr)
    cube = np.stack(stack, axis=0)  # (T, H, W)
    x = np.array(years, dtype="float32")
    x_mean = x.mean()
    # Mask: only compute slope where all years are finite (strict — robust trend)
    valid = np.all(np.isfinite(cube), axis=0)
    y_mean = np.where(valid, cube.mean(axis=0), np.nan)
    numerator = np.where(
        valid,
        np.nansum((x[:, None, None] - x_mean) * (cube - y_mean), axis=0),
        np.nan,
    )
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator
    slope = np.where(valid, slope, np.nan)

    norm = Normalize(vmin=-2.0, vmax=2.0)
    img = _render_rgba(slope, "coolwarm", norm, alpha_mode="diverging",
                       alpha_floor=0.02, alpha_ceiling=0.9, alpha_gamma=0.55)
    out_path = os.path.join(out_dir, "annual_trend.png")
    _save_png(img, out_path)
    return {
        "id": "trend",
        "label": "Annual growth rate",
        "description": "Per-pixel linear trend, 2012–2024. Red = brightening each year, blue = dimming.",
        "url": "rasters/annual_trend.png",
        "bounds": list(ref_bounds),
        "year_range": [min(years), max(years)],
        "legend": {
            "colormap": "coolwarm",
            "min": -2.0,
            "max": 2.0,
            "unit": "nW/cm²/sr per year",
        },
    }


# ── Entry point ───────────────────────────────────────────────────────────


def export_web_rasters(output_dir: str, out_dir: str, years: list[int] | None = None) -> dict:
    """Produce all three web-overlay PNGs plus the manifest.

    Parameters
    ----------
    output_dir
        Path to the research-repo output root (where ``subsets/<year>/`` lives).
    out_dir
        Where to write the rasters and manifest (typically ``web-export/``).
    years
        Years to include. Defaults to every directory under ``subsets/`` that
        has a median raster.
    """
    raster_dir = os.path.join(out_dir, "rasters")
    os.makedirs(raster_dir, exist_ok=True)

    if years is None:
        subsets = os.path.join(output_dir, "subsets")
        candidates = sorted(int(d) for d in os.listdir(subsets) if d.isdigit())
        years = [y for y in candidates if os.path.exists(_subset_path(output_dir, y))]
    if not years:
        raise FileNotFoundError(f"No median rasters under {output_dir}/subsets/")

    log.info("Web rasters: %d years available (%d–%d)", len(years), min(years), max(years))

    layers = [
        render_latest_radiance(years, output_dir, raster_dir),
        render_change_12yr(years, output_dir, raster_dir),
        render_annual_trend(years, output_dir, raster_dir),
    ]

    # Shared bounds from first layer (all agree); expose in the top-level
    # manifest so the frontend can set maxBounds + inverse-mask without
    # having to open each layer.
    manifest_path = os.path.join(out_dir, "rasters.json")
    manifest = {
        "bounds": layers[0]["bounds"],
        "layers": layers,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    summary = {
        "rasters_written": len(layers),
        "manifest": manifest_path,
        "out_dir": os.path.abspath(raster_dir),
    }
    log.info("Web rasters exported: %s", summary)
    return summary


if __name__ == "__main__":
    import argparse
    from src import config
    parser = argparse.ArgumentParser(description="Export web-ready raster overlays (RGBA PNG + manifest).")
    parser.add_argument("--output-dir", default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--out-dir", default="web-export")
    parser.add_argument("--years", default=None, help="Comma list or range (e.g. '2012-2024').")
    args = parser.parse_args()

    if args.years:
        if "-" in args.years:
            a, b = args.years.split("-")
            ylist = list(range(int(a), int(b) + 1))
        else:
            ylist = [int(y) for y in args.years.split(",")]
    else:
        ylist = None

    export_web_rasters(args.output_dir, args.out_dir, ylist)
