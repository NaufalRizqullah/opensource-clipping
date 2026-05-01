"""
clipping.studio — Video Rendering Engine

Thin compatibility entry point for the Studio pipeline.
Public API is re-exported from internal modules under clipping/studio/.
"""

import importlib.util
import os

FIREFOX_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0"
)


def _load_studio_internal_module(file_name: str, module_alias: str):
    """
    Load an internal `clipping/studio/*.py` module by file path.

    Args:
        file_name: Python filename inside `clipping/studio`.
        module_alias: Unique import alias used by importlib.

    Returns:
        Loaded Python module object.
    """
    module_path = os.path.join(os.path.dirname(__file__), "studio", file_name)
    spec = importlib.util.spec_from_file_location(module_alias, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Gagal memuat modul internal: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_helpers = _load_studio_internal_module("helpers.py", "clipping_studio_helpers")
_ffmpeg_utils = _load_studio_internal_module("ffmpeg_utils.py", "clipping_studio_ffmpeg_utils")
_core = _load_studio_internal_module("core.py", "clipping_studio_core")

format_seconds = _helpers.format_seconds
escape_ffmpeg_filter_value = _helpers.escape_ffmpeg_filter_value
detect_video_encoder = _ffmpeg_utils.detect_video_encoder
get_ts_encode_args = _ffmpeg_utils.get_ts_encode_args
get_mp4_encode_args = _ffmpeg_utils.get_mp4_encode_args
open_ffmpeg_video_writer = _ffmpeg_utils.open_ffmpeg_video_writer
build_ffmpeg_progress_cmd = _ffmpeg_utils.build_ffmpeg_progress_cmd
run_ffmpeg_with_progress = _ffmpeg_utils.run_ffmpeg_with_progress

get_face_detector = _core.get_face_detector
estimate_speaker_count_from_video = _core.estimate_speaker_count_from_video
download_google_font = _core.download_google_font
register_fonts_for_libass = _core.register_fonts_for_libass
siapkan_font_tipografi = _core.siapkan_font_tipografi
resolve_pixabay_audio_url = _core.resolve_pixabay_audio_url
download_bgm_from_pixabay_page = _core.download_bgm_from_pixabay_page
download_pexels_broll = _core.download_pexels_broll
crop_center_broll = _core.crop_center_broll
buat_video_hybrid = _core.buat_video_hybrid
buat_file_ass = _core.buat_file_ass
siapkan_glitch_video = _core.siapkan_glitch_video
buat_thumbnail = _core.buat_thumbnail
buat_video_split_screen = _core.buat_video_split_screen
buat_video_camera_switch = _core.buat_video_camera_switch
_get_render_dims = _core._get_render_dims
_is_vertical_ratio = _core._is_vertical_ratio
proses_klip = _core.proses_klip
