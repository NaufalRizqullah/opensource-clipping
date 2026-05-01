"""
clipping.studio.transitions — Transition Asset Downloader & Manager

Downloads and caches transition overlay videos (film burn, light leak,
film grain, film leader) from YouTube sources (Think Make Push channel).

These assets have a ~5 second branding intro that must be skipped.
The usable transition content starts at the configured skip offset.
"""

import importlib.util
import os
import random
import subprocess

from yt_dlp import YoutubeDL


def _load_studio_internal_module(file_name: str, module_alias: str):
    module_path = os.path.join(os.path.dirname(__file__), file_name)
    spec = importlib.util.spec_from_file_location(module_alias, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ffmpeg_utils = _load_studio_internal_module(
    "ffmpeg_utils.py", "clipping_studio_ffmpeg_utils"
)
get_ts_encode_args = _ffmpeg_utils.get_ts_encode_args
utils = _load_studio_internal_module("utils.py", "clipping_studio_utils")
_get_render_dims = utils._get_render_dims
_is_vertical_ratio = utils._is_vertical_ratio
RATIO_MAP = utils.RATIO_MAP

# ═══════════════════════════════════════════════════════════════════════
# TRANSITION POOL  (source: Think Make Push — YouTube)
#
# Each entry:
#   url         : YouTube watch URL
#   skip        : seconds to skip (branding/ad intro)
#   duration    : usable duration to extract (seconds, None = rest of video)
#   type        : category tag for future filtering
#   orientation : "landscape" or "vertical"
#   label       : human-readable label for logging
# ═══════════════════════════════════════════════════════════════════════

TMP_TRANSITION_POOL = [
    {
        "url": "https://www.youtube.com/watch?v=yfKv03nLaBE",
        "skip": 5,
        "duration": None,
        "type": "film_burn",
        "orientation": "landscape",
        "label": "GRUNGY Film Burn Transitions",
    },
    {
        "url": "https://www.youtube.com/watch?v=uYBcUpLxtEM",
        "skip": 5,
        "duration": None,
        "type": "film_burn",
        "orientation": "landscape",
        "label": "GRUNGE Film Overlay with Sound",
    },
    {
        "url": "https://www.youtube.com/watch?v=YFzGx0JuUUQ",
        "skip": 5,
        "duration": None,
        "type": "film_overlay",
        "orientation": "landscape",
        "label": "35mm Film Overlay",
    },
    {
        "url": "https://www.youtube.com/watch?v=iGvnBXS3pyM",
        "skip": 5,
        "duration": None,
        "type": "film_leader",
        "orientation": "landscape",
        "label": "Dirty Grainy Film Leader (Burns & Leaks)",
    },
    {
        "url": "https://www.youtube.com/watch?v=BsKj9iiimTE",
        "skip": 5,
        "duration": None,
        "type": "film_leader",
        "orientation": "landscape",
        "label": "Classic Film Leader Overlays",
    },
    {
        "url": "https://www.youtube.com/watch?v=OaK3jjBfOi0",
        "skip": 5,
        "duration": None,
        "type": "film_grain",
        "orientation": "landscape",
        "label": "Film Grain Overlay with Sound Effect",
    },
    # ── Vertical Variants ──
    {
        "url": "https://www.youtube.com/watch?v=k0BvSreLx5E",
        "skip": 5,
        "duration": None,
        "type": "film_burn",
        "orientation": "vertical",
        "label": "Vertical Vibrant Film Burn Overlay",
    },
    {
        "url": "https://www.youtube.com/watch?v=eiditSLUA3I",
        "skip": 5,
        "duration": None,
        "type": "film_burn",
        "orientation": "vertical",
        "label": "Vertical Rich and Vibrant Colors",
    },
]


def _get_cache_dir(cfg) -> str:
    """
    Return (and create) the directory where downloaded transition raw
    files are stored.  Located inside the project base directory under
    ``transitions_cache/``.
    """
    cache_dir = os.path.abspath(
        os.path.join(getattr(cfg, "base_dir", os.getcwd()), "transitions_cache")
    )
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _raw_filename(entry: dict) -> str:
    """Derive a deterministic raw filename from a pool entry's URL."""
    video_id = entry["url"].split("v=")[-1].split("&")[0]
    return f"tmp_raw_{video_id}.mp4"


def download_transition_raw(entry: dict, cfg) -> str | None:
    """
    Download a single transition video from YouTube if not already cached.

    Args:
        entry: A dict from ``TMP_TRANSITION_POOL``.
        cfg:   Runtime config (used for ``base_dir``).

    Returns:
        Absolute path to the downloaded raw MP4, or None on failure.
    """
    cache_dir = _get_cache_dir(cfg)
    raw_path = os.path.join(cache_dir, _raw_filename(entry))

    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 10_000:
        return raw_path

    print(f"   📥 [Transition] Downloading: {entry['label']}...")

    try:
        YoutubeDL(
            {
                "format": "best[ext=mp4]",
                "outtmpl": raw_path,
                "quiet": True,
                "no_warnings": True,
            }
        ).download([entry["url"]])
    except Exception as e:
        print(f"   ⚠️ [Transition] Download gagal ({entry['label']}): {e}")
        return None

    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 10_000:
        print(f"   ✅ [Transition] Tersimpan: {raw_path}")
        return raw_path

    print(f"   ⚠️ [Transition] File terlalu kecil / gagal: {raw_path}")
    return None


def download_all_transitions(cfg, types: list[str] | None = None) -> list[dict]:
    """
    Download all (or filtered) transition assets from the pool.

    Args:
        cfg:   Runtime config.
        types: Optional list of type tags to filter (e.g. ``["film_burn"]``).
               If None, downloads everything.

    Returns:
        List of dicts — each original pool entry augmented with
        ``"raw_path"`` pointing to the local file.  Entries that failed
        to download are excluded.
    """
    pool = TMP_TRANSITION_POOL
    if types:
        pool = [e for e in pool if e["type"] in types]

    results = []
    for entry in pool:
        path = download_transition_raw(entry, cfg)
        if path:
            enriched = dict(entry)
            enriched["raw_path"] = path
            results.append(enriched)

    print(
        f"   📦 [Transition] {len(results)}/{len(pool)} asset berhasil diunduh."
    )
    return results


def get_random_transition(
    cfg,
    transition_type: str | None = None,
    orientation: str | None = None,
) -> dict | None:
    """
    Pick a random transition from the pool, downloading if necessary.

    Args:
        cfg:              Runtime config.
        transition_type:  Filter by type (``film_burn``, ``film_leader``,
                          ``film_grain``, ``film_overlay``).  None = any.
        orientation:      ``"landscape"`` or ``"vertical"``.  None = any.

    Returns:
        A pool entry dict with ``"raw_path"`` set, or None if nothing
        is available.
    """
    candidates = list(TMP_TRANSITION_POOL)

    if transition_type:
        candidates = [c for c in candidates if c["type"] == transition_type]
    if orientation:
        candidates = [c for c in candidates if c["orientation"] == orientation]

    if not candidates:
        return None

    random.shuffle(candidates)

    for entry in candidates:
        path = download_transition_raw(entry, cfg)
        if path:
            result = dict(entry)
            result["raw_path"] = path
            return result

    return None


def prepare_transition_clip(
    entry: dict,
    rasio: str,
    cfg,
    video_encoder: dict,
    source_h: int = 1080,
    clip_duration: float | None = None,
    custom_dims: tuple | None = None,
) -> str | None:
    """
    Extract the usable portion of a downloaded transition asset,
    crop/scale it to match the target output dimensions, and encode
    it as a ``.ts`` segment ready for concatenation or overlay.

    This skips the branding intro (``entry["skip"]`` seconds) and
    extracts ``clip_duration`` seconds of usable content.

    Args:
        entry:          Pool entry dict (must have ``"raw_path"``).
        rasio:          Target aspect ratio string.
        cfg:            Runtime config.
        video_encoder:  Encoder descriptor dict.
        source_h:       Source video height for dimension calculation.
        clip_duration:  How many seconds to extract.  None means the
                        entry's own ``duration`` field (or 3s fallback).
        custom_dims:    Optional ``(w, h)`` to override calculated dims.

    Returns:
        Path to the prepared ``.ts`` file, or None on failure.
    """
    raw_path = entry.get("raw_path")
    if not raw_path or not os.path.exists(raw_path):
        return None

    if custom_dims:
        out_w, out_h = custom_dims
    else:
        out_w, out_h = _get_render_dims(cfg, rasio, source_h=source_h)

    skip = entry.get("skip", 5)
    dur = clip_duration or entry.get("duration") or 3.0

    video_id = entry["url"].split("v=")[-1].split("&")[0]
    ts_path = f"transition_{video_id}_{out_w}x{out_h}.ts"

    if os.path.exists(ts_path):
        return ts_path

    algo = getattr(cfg, "video_scale_algo", "lanczos")

    if custom_dims:
        vf = f"scale={out_w}:{out_h}:flags={algo},setsar=1"
    elif _is_vertical_ratio(rasio):
        w_part, h_part = RATIO_MAP.get(rasio, (9, 16))
        vf = (
            f"crop=ih*{w_part}/{h_part}:ih:(iw-ih*{w_part}/{h_part})/2:0,"
            f"scale={out_w}:{out_h}:flags={algo},setsar=1"
        )
    else:
        vf = f"scale={out_w}:{out_h}:flags={algo},setsar=1"

    cmd = (
        [
            "ffmpeg",
            "-y",
            "-ss", str(skip),
            "-t", str(dur),
            "-i", raw_path,
            "-vf", vf,
            "-an",
        ]
        + get_ts_encode_args(video_encoder, fps=30)
        + [ts_path]
    )

    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return ts_path
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️ [Transition] FFmpeg gagal untuk {entry['label']}: {e}")
        return None
