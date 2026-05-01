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


def download_google_font(
    url, output_filename, font_dir, max_retry=10, min_valid_size=1000
):
    """
    Download a Google font file with retry and basic integrity checks.

    Args:
        url (str): The direct download URL for the font file.
        output_filename (str): Local filename to write the downloaded font.
        font_dir (str): Destination directory where the font will be saved.
        max_retry (int): Maximum network retry attempts before failing. Defaults to 10.
        min_valid_size (int): Minimum file size in bytes to consider the download valid. Defaults to 1000.

    Returns:
        bool: True if the font file is successfully downloaded and validated, False if it fails after all retries.

    Side Effects:
        Writes a temporary file (`.part`) and replaces it upon successful download.
        Prints download progress and status to stdout.

    Raises:
        Exceptions are caught internally and trigger a retry. No exceptions are raised to the caller.
    """
    file_path = os.path.join(font_dir, output_filename)
    temp_path = file_path + ".part"

    def is_valid(path):
        return os.path.exists(path) and os.path.getsize(path) > min_valid_size

    if is_valid(file_path):
        print(f"   ✅ Font '{output_filename}' sudah ada dan valid.")
        return True

    headers = {
        "User-Agent": FIREFOX_UA,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://fontsource.org/",
        "Connection": "keep-alive",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
    }

    for percobaan in range(1, max_retry + 1):
        try:
            print(
                f"   📥 Mendownload font '{output_filename}'... ({percobaan}/{max_retry})"
            )

            for p in [temp_path, file_path]:
                if os.path.exists(p) and not is_valid(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

            with requests.get(
                url, headers=headers, stream=True, timeout=45, allow_redirects=True
            ) as r:
                r.raise_for_status()
                with open(temp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            if not is_valid(temp_path):
                ukuran = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
                raise ValueError(f"file hasil download tidak valid ({ukuran} byte)")

            os.replace(temp_path, file_path)

            if is_valid(file_path):
                print(
                    f"   ✅ Font '{output_filename}' berhasil diunduh dan terverifikasi."
                )
                return True

            raise FileNotFoundError(
                f"File final '{output_filename}' tidak valid di {font_dir}"
            )

        except Exception as e:
            print(
                f"   ⚠️ Gagal download font '{output_filename}' percobaan {percobaan}: {e}"
            )

            for p in [temp_path, file_path]:
                if os.path.exists(p):
                    try:
                        if os.path.getsize(p) <= min_valid_size:
                            os.remove(p)
                    except Exception:
                        pass

            if percobaan < max_retry:
                time.sleep(1.5)

    print(f"   ❌ Gagal total: font '{output_filename}' setelah {max_retry} percobaan.")
    return False


def register_fonts_for_libass(font_dir):
    """
    Copy downloaded fonts to the system font directory and refresh the font cache (Linux only).
    This ensures that FFmpeg/libass can correctly find and use the downloaded fonts for subtitles.

    Args:
        font_dir (str): Directory where the downloaded font files (.ttf, .otf) are located.

    Returns:
        None

    Side Effects:
        Creates `~/.local/share/fonts` if it doesn't exist.
        Copies font files into the system user font directory.
        Runs the `fc-cache` shell command on Linux systems.

    Raises:
        None explicitly. OS-level permission errors during file copying may occur.
    """
    if os.name == "nt":
        # On Windows, libass can use fontsdir directly — skip fc-cache
        return

    user_font_dir = os.path.expanduser("~/.local/share/fonts")
    os.makedirs(user_font_dir, exist_ok=True)

    copied = []
    for fn in os.listdir(font_dir):
        if fn.lower().endswith((".ttf", ".otf")):
            src = os.path.join(font_dir, fn)
            dst = os.path.join(user_font_dir, fn)
            shutil.copy2(src, dst)
            copied.append(dst)

    if copied:
        subprocess.run(
            ["fc-cache", "-f", "-v"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def siapkan_font_tipografi(cfg):
    """
    Ensure all required typography fonts for the selected style are downloaded and registered.

    Args:
        cfg: Runtime configuration object that contains `daftar_font` (font dictionary), 
             `gaya_font_aktif` (active style key), and `font_dir` (destination directory).

    Returns:
        None

    Side Effects:
        Downloads font files to `font_dir`.
        Registers fonts in the system for libass usage.
        Prints status messages to stdout.

    Raises:
        RuntimeError: If either the primary or secondary required fonts fail to download or validate.
    """
    daftar_font = cfg.daftar_font
    gaya = cfg.gaya_font_aktif
    font_dir = cfg.font_dir

    f_utama = daftar_font[gaya]["utama"]
    f_khusus = daftar_font[gaya]["khusus"]

    ok_utama = download_google_font(f_utama["url"], f_utama["file"], font_dir)
    ok_khusus = download_google_font(f_khusus["url"], f_khusus["file"], font_dir)

    path_utama = os.path.join(font_dir, f_utama["file"])
    path_khusus = os.path.join(font_dir, f_khusus["file"])

    if not (
        ok_utama and os.path.exists(path_utama) and os.path.getsize(path_utama) > 1000
    ):
        raise RuntimeError(f"Font utama gagal disiapkan: {path_utama}")

    if not (
        ok_khusus
        and os.path.exists(path_khusus)
        and os.path.getsize(path_khusus) > 1000
    ):
        raise RuntimeError(f"Font khusus gagal disiapkan: {path_khusus}")

    register_fonts_for_libass(font_dir)
    print(f"✅ Semua font berhasil disiapkan di: {font_dir}")


