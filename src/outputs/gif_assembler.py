"""
GIF assembler for animation frames.

Converts directories of sequentially-named PNG frames into animated GIFs
with adjustable size, quality, frame duration, and looping behaviour.
Uses Pillow (already in requirements.txt).
"""

import glob
import logging
import os
import re

from PIL import Image

log = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────
DEFAULT_FRAME_DURATION_MS = 500  # half-second per frame
DEFAULT_MAX_WIDTH = 800          # pixels; height scales proportionally
DEFAULT_LOOP = 0                 # 0 = infinite loop
DEFAULT_QUALITY = 80             # 1-100 (Pillow quantize quality)
DEFAULT_LAST_FRAME_PAUSE_MS = 2000  # linger on the final frame


def assemble_gif(
    frame_dir: str,
    output_path: str,
    *,
    pattern: str = "*.png",
    duration_ms: int = DEFAULT_FRAME_DURATION_MS,
    max_width: int = DEFAULT_MAX_WIDTH,
    loop: int = DEFAULT_LOOP,
    quality: int = DEFAULT_QUALITY,
    last_frame_pause_ms: int = DEFAULT_LAST_FRAME_PAUSE_MS,
) -> str | None:
    """Assemble PNG frames in *frame_dir* into an animated GIF.

    Parameters
    ----------
    frame_dir : str
        Directory containing the frame images.
    output_path : str
        Where to write the GIF.
    pattern : str
        Glob pattern for frame files (default ``*.png``).
    duration_ms : int
        Milliseconds each frame is shown.
    max_width : int
        Maximum width in pixels.  Frames are down-scaled proportionally.
        Set to 0 to keep original resolution.
    loop : int
        Number of loops (0 = infinite).
    quality : int
        Colour quantisation quality (1-100).  Higher = better colours but
        larger file.
    last_frame_pause_ms : int
        Extra pause on the final frame so the end state is visible.

    Returns
    -------
    str or None
        Path to the generated GIF, or None on failure.
    """
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, pattern)))
    if not frame_paths:
        log.warning("No frames matching '%s' in %s", pattern, frame_dir)
        return None

    # Sort numerically when filenames contain a year or sequence number
    def _sort_key(p):
        nums = re.findall(r"\d+", os.path.basename(p))
        return int(nums[-1]) if nums else p
    frame_paths.sort(key=_sort_key)

    log.info("Assembling %d frames from %s → %s", len(frame_paths),
             frame_dir, output_path)

    frames = []
    for fp in frame_paths:
        img = Image.open(fp).convert("RGBA")
        # Composite onto black background (for transparency)
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        bg.paste(img, mask=img)
        img = bg.convert("RGB")

        if max_width and img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        # Quantize to palette for GIF
        img = img.quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                           dither=Image.Dither.FLOYDSTEINBERG)
        frames.append(img)

    if not frames:
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Build per-frame durations (linger on last frame)
    durations = [duration_ms] * len(frames)
    durations[-1] = last_frame_pause_ms

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=loop,
        optimize=(quality < 90),
    )

    size_kb = os.path.getsize(output_path) / 1024
    log.info("GIF saved: %s (%.0f KB, %d frames)", output_path, size_kb,
             len(frames))
    return output_path


def assemble_all_gifs(
    maps_dir: str,
    *,
    duration_ms: int = DEFAULT_FRAME_DURATION_MS,
    max_width: int = DEFAULT_MAX_WIDTH,
    quality: int = DEFAULT_QUALITY,
    last_frame_pause_ms: int = DEFAULT_LAST_FRAME_PAUSE_MS,
) -> list[str]:
    """Discover all frame directories under *maps_dir*/frames/ and build GIFs.

    Looks for the standard frame subdirectories created by the pipeline:
    - frames/sprawl/           → gifs/sprawl.gif
    - frames/differential/     → gifs/differential.gif
    - frames/darkness/         → gifs/darkness.gif
    - frames/light_increase/   → gifs/light_increase.gif
    - frames/districts/<name>/ → gifs/districts/<name>.gif

    Returns a list of generated GIF paths.
    """
    frames_root = os.path.join(maps_dir, "frames")
    gifs_dir = os.path.join(maps_dir, "gifs")
    os.makedirs(gifs_dir, exist_ok=True)

    gif_settings = dict(
        duration_ms=duration_ms,
        max_width=max_width,
        quality=quality,
        last_frame_pause_ms=last_frame_pause_ms,
    )

    generated = []

    # State-level frame sets
    for name in ("sprawl", "differential", "darkness", "light_increase"):
        frame_dir = os.path.join(frames_root, name)
        if os.path.isdir(frame_dir):
            out = os.path.join(gifs_dir, f"{name}.gif")
            result = assemble_gif(frame_dir, out, **gif_settings)
            if result:
                generated.append(result)

    # Per-district frame sets
    districts_frames = os.path.join(frames_root, "districts")
    if os.path.isdir(districts_frames):
        districts_gif_dir = os.path.join(gifs_dir, "districts")
        os.makedirs(districts_gif_dir, exist_ok=True)
        for district_dir in sorted(os.listdir(districts_frames)):
            full = os.path.join(districts_frames, district_dir)
            if os.path.isdir(full):
                out = os.path.join(districts_gif_dir, f"{district_dir}.gif")
                result = assemble_gif(full, out, **gif_settings)
                if result:
                    generated.append(result)

    log.info("Assembled %d GIFs total in %s", len(generated), gifs_dir)
    return generated
