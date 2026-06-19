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


def get_local_bgm_file(mood, bgm_dir):
    """
    Get a random BGM MP3 file from the local assets directory based on mood.

    Args:
        mood (str): The requested mood (e.g., 'chill', 'epic', 'sad').
        bgm_dir (str): Base directory for BGM assets.

    Returns:
        str: Absolute path to the selected MP3 file, or None if not found/empty.
    """
    mood_dir = os.path.join(bgm_dir, mood)
    
    if not os.path.exists(mood_dir) or not os.path.isdir(mood_dir):
        return None
        
    mp3_files = [f for f in os.listdir(mood_dir) if f.lower().endswith(".mp3")]
    
    if not mp3_files:
        return None
        
    selected_file = random.choice(mp3_files)
    return os.path.abspath(os.path.join(mood_dir, selected_file))


