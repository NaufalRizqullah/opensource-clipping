import html
import importlib.util
import json
import math
import os
import random
import re
import shutil
import string
import subprocess
import textwrap
import time
import urllib.parse
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import requests
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw, ImageFont
from yt_dlp import YoutubeDL

FIREFOX_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0"

def _load_studio_internal_module(file_name: str, module_alias: str):
    """
    Load an internal studio module dynamically by file path.

    Args:
        file_name (str): The filename of the module to load.
        module_alias (str): The alias to assign to the loaded module.

    Returns:
        module: The dynamically loaded Python module.

    Side Effects:
        Loads and executes the module code.

    Raises:
        ImportError: If the module specification cannot be found or loaded.
    """
    module_path = os.path.join(os.path.dirname(__file__), file_name)
    spec = importlib.util.spec_from_file_location(module_alias, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Gagal memuat modul internal: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_helpers = _load_studio_internal_module("helpers.py", "clipping_studio_helpers")
_ffmpeg_utils = _load_studio_internal_module("ffmpeg_utils.py", "clipping_studio_ffmpeg_utils")
format_seconds = _helpers.format_seconds
escape_ffmpeg_filter_value = _helpers.escape_ffmpeg_filter_value
detect_video_encoder = _ffmpeg_utils.detect_video_encoder
get_ts_encode_args = _ffmpeg_utils.get_ts_encode_args
get_mp4_encode_args = _ffmpeg_utils.get_mp4_encode_args
open_ffmpeg_video_writer = _ffmpeg_utils.open_ffmpeg_video_writer
build_ffmpeg_progress_cmd = _ffmpeg_utils.build_ffmpeg_progress_cmd
run_ffmpeg_with_progress = _ffmpeg_utils.run_ffmpeg_with_progress


def _get_cv2_interpolation(cfg=None):
    """
    Resolve OpenCV interpolation mode from runtime config.

    Args:
        cfg: Runtime config that may contain `video_scale_algo`.

    Returns:
        OpenCV interpolation constant.
    """
    algo = str(
        getattr(cfg, "video_scale_algo", os.environ.get("OSC_VIDEO_SCALE_ALGO", "lanczos"))
    ).lower()
    mapping = {
        "lanczos": cv2.INTER_LANCZOS4,
        "bicubic": cv2.INTER_CUBIC,
        "bilinear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
    }
    return mapping.get(algo, cv2.INTER_LANCZOS4)


def _resize_frame(frame, size, cfg=None):
    """
    Resize frame using configured interpolation algorithm.

    Args:
        frame: Input image/frame array.
        size: Target `(width, height)`.
        cfg: Runtime config that may define `video_scale_algo`.

    Returns:
        Resized frame.
    """
    return cv2.resize(frame, size, interpolation=_get_cv2_interpolation(cfg))


def _get_render_dims(cfg, rasio, source_h=1080):
    """
    Calculate target output resolution based on config and aspect ratio.
    If mode is 'source', it uses the provided source_h as the base dimension.

    Args:
        cfg: Runtime config object that specifies `render_output_height`.
        rasio (str): The target aspect ratio ('9:16' or '16:9').
        source_h (int): The original source video height in pixels.

    Returns:
        tuple: A tuple containing (target_width, target_height).

    Side Effects:
        None

    Raises:
        ValueError: If config contains an invalid non-integer height when not 'source'.
    """
    mode = str(getattr(cfg, "render_output_height", "1080")).lower()
    if mode == "source":
        target_h_base = source_h
    else:
        try:
            target_h_base = int(mode)
        except (ValueError, TypeError):
            target_h_base = 1080

    if rasio == "9:16":
        # Horizontal 1080p source -> Vertical 1080x1920 output
        # Here target_h_base is treated as the 'short' side for 9:16
        out_h = int(target_h_base * 16 / 9)
        if out_h % 2 != 0:
            out_h += 1
        out_w = target_h_base
    else:
        # 16:9
        out_w = int(target_h_base * 16 / 9)
        if out_w % 2 != 0:
            out_w += 1
        out_h = target_h_base

    return out_w, out_h


