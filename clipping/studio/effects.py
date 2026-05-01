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
    module_path = os.path.join(os.path.dirname(__file__), file_name)
    spec = importlib.util.spec_from_file_location(module_alias, module_path)
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

utils = _load_studio_internal_module("utils.py", "clipping_studio_utils")
_get_render_dims = utils._get_render_dims

def siapkan_glitch_video(rasio, cfg, video_encoder, source_h=1080, custom_dims=None):
    """
    Generate a 1-second VHS glitch transition video from the source video frame.

    Args:
        cfg: Runtime config defining temporary directory and output rendering height.
        rasio (str): Target output ratio string ('9:16' or '16:9').
        input_video (str): Path to the source video.

    Returns:
        str: Absolute path to the generated glitch transition video file, or None if creation fails.

    Side Effects:
        Creates a new MP4 video file in the `cfg.temp_dir`.
        Uses `subprocess` to run ffmpeg.

    Raises:
        Exceptions caught internally and returns None.
    """
    if custom_dims:
        out_w, out_h = custom_dims
    else:
        out_w, out_h = _get_render_dims(cfg, rasio, source_h=source_h)
    
    # Use dimensions in filename to allow multiple cached versions
    glitch_ts = f"glitch_ready_{out_w}x{out_h}.ts"
    if os.path.exists(glitch_ts):
        return glitch_ts

    if not os.path.exists("glitch_raw.mp4"):
        YoutubeDL(
            {
                "format": "best[ext=mp4]",
                "outtmpl": "glitch_raw.mp4",
                "quiet": True,
            }
        ).download([cfg.url_glitch_video])

    algo = getattr(cfg, "video_scale_algo", "lanczos")
    
    # If custom dims provided, we just scale. If ratio is 9:16, we crop.
    if custom_dims:
        filter_g = f"scale={out_w}:{out_h}:flags={algo},setsar=1"
    else:
        filter_g = (
            f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale={out_w}:{out_h}:flags={algo},setsar=1"
            if rasio == "9:16"
            else f"scale={out_w}:{out_h}:flags={algo},setsar=1"
        )

    cmd = (
        [
            "ffmpeg",
            "-y",
            "-ss",
            "0.2",
            "-t",
            "1",
            "-i",
            "glitch_raw.mp4",
            "-vf",
            filter_g,
        ]
        + get_ts_encode_args(video_encoder, fps=30)
        + [glitch_ts]
    )

    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return glitch_ts


