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


def resolve_pixabay_audio_url(page_url, timeout=45):
    """
    Resolve the direct Pixabay CDN audio URL from a track page URL.

    Args:
        page_url (str): The URL of the Pixabay audio track page.
        timeout (int, optional): HTTP request timeout in seconds. Defaults to 45.

    Returns:
        str: The direct downloadable MP3 URL from Pixabay CDN.

    Side Effects:
        Makes an HTTP GET request to the provided `page_url`.

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
        RuntimeError: If no playable audio URL pattern is found in the HTML content.
    """
    headers = {
        "User-Agent": FIREFOX_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://pixabay.com/",
        "Connection": "keep-alive",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
    }

    r = requests.get(page_url, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    html_text = r.text

    patterns = [
        r'https://cdn\.pixabay\.com/download/audio/[^"\']+',
        r'"contentUrl":"(https:\\/\\/cdn\.pixabay\.com\\/download\\/audio\\/[^"]+)"',
        r'"url":"(https:\\/\\/cdn\.pixabay\.com\\/download\\/audio\\/[^"]+)"',
        r'downloadUrl":"(https:\\/\\/cdn\.pixabay\.com\\/download\\/audio\\/[^"]+)"',
    ]

    for pattern in patterns:
        m = re.search(pattern, html_text)
        if m:
            url = m.group(1) if m.groups() else m.group(0)
            url = url.replace("\\/", "/")
            url = html.unescape(url)
            return url

    raise RuntimeError("MP3 URL tidak ketemu di halaman Pixabay")


def download_bgm_from_pixabay_page(
    page_url, output_path, max_retry=4, min_valid_size=10_000
):
    """
    Download BGM audio from a Pixabay page into a local output path.

    Args:
        page_url (str): The URL of the Pixabay audio track page.
        output_path (str): The local file path where the MP3 will be saved.
        max_retry (int, optional): Maximum download retry attempts. Defaults to 4.
        min_valid_size (int, optional): Minimum valid file size in bytes. Defaults to 10,000.

    Returns:
        bool: True if the file is successfully downloaded and validated, False otherwise.

    Side Effects:
        Calls `resolve_pixabay_audio_url` to get the direct link.
        Writes a temporary file (`.part`) and replaces the target `output_path` on success.
        Prints progress and error messages to stdout.

    Raises:
        Exceptions are caught and retried. Returns False if all retries fail.
    """
    headers = {
        "User-Agent": FIREFOX_UA,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://pixabay.com/",
        "Origin": "https://pixabay.com",
        "Connection": "keep-alive",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
    }

    temp_path = output_path + ".part"

    for attempt in range(1, max_retry + 1):
        try:
            audio_url = resolve_pixabay_audio_url(page_url)
            print(f"   🔗 Resolved BGM URL: {audio_url[:100]}...")

            if os.path.exists(temp_path):
                os.remove(temp_path)

            with requests.get(
                audio_url,
                headers=headers,
                stream=True,
                timeout=(20, 120),
                allow_redirects=True,
            ) as r:
                r.raise_for_status()

                content_type = (r.headers.get("Content-Type") or "").lower()
                if (
                    "audio" not in content_type
                    and "mpeg" not in content_type
                    and "octet-stream" not in content_type
                ):
                    raise ValueError(f"Respon bukan audio: {content_type}")

                with open(temp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)

            if (
                not os.path.exists(temp_path)
                or os.path.getsize(temp_path) < min_valid_size
            ):
                size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
                raise ValueError(f"File BGM tidak valid ({size} byte)")

            os.replace(temp_path, output_path)
            return True

        except Exception as e:
            print(f"   ⚠️ Gagal download BGM attempt {attempt}/{max_retry}: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            time.sleep(1.5 * attempt)

    print(f"   ❌ Gagal total download BGM")
    return False


