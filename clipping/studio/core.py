"""
Core Studio pipeline implementation extracted from clipping/studio.py.

This module keeps original behavior while allowing clipping/studio.py to remain
a thin orchestration and compatibility entry point.
"""

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

FIREFOX_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0"
)


def _load_studio_internal_module(file_name: str, module_alias: str):
    """Load an internal clipping/studio module by file path."""
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


# ==============================================================================
# MEDIAPIPE FACE DETECTOR (SINGLETON)
# ==============================================================================

_FACE_DETECTOR = None


def get_face_detector(cfg):
    """
    Create or reuse a singleton MediaPipe face detector instance.

    Args:
        cfg: Runtime config that includes model path and model URL.

    Returns:
        Initialized MediaPipe face detector singleton.
    """
    global _FACE_DETECTOR

    if _FACE_DETECTOR is None:
        if not os.path.exists(cfg.file_mediapipe_model):
            urllib.request.urlretrieve(
                cfg.url_mediapipe_model, cfg.file_mediapipe_model
            )

        base_options = mp_python.BaseOptions(model_asset_path=cfg.file_mediapipe_model)
        _FACE_DETECTOR = mp_vision.FaceDetector.create_from_options(
            mp_vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=0.5,
            )
        )

    return _FACE_DETECTOR


def estimate_speaker_count_from_video(video_path: str, cfg) -> int:
    """
    Sample frames from the video to estimate the max number of visible faces.
    Used for automatically setting min_speakers for pyannote.
    """
    import cv2

    print("🔍 Auto-detecting speaker count via visual scanning...", flush=True)

    yolo_model = None
    detector = None

    if cfg.face_detector == "yolo":
        from ultralytics import YOLO
        import logging

        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        try:
            model_name = f"yolov{cfg.yolo_size}-face.pt"
            yolo_model = YOLO(model_name)
        except Exception as e:
            print(f"⚠️ YOLO face detect gagal: {e}. Fallback ke Mediapipe.")
            cfg.face_detector = "mediapipe"

    if cfg.face_detector != "yolo":
        detector = get_face_detector(cfg)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps if fps > 0 else 0

    if duration == 0:
        cap.release()
        return 2

    sample_count = 20
    step = duration / sample_count
    max_faces = 0

    for i in range(sample_count):
        t = i * step
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        faces_in_frame = 0

        if cfg.face_detector == "yolo" and yolo_model:
            results = yolo_model(frame, verbose=False)
            if results and len(results[0].boxes) > 0:
                faces_in_frame = len(results[0].boxes)
        else:
            mp_image = mp_python.Image(
                image_format=mp_python.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            results = detector.detect(mp_image)
            if results.detections:
                faces_in_frame = len(results.detections)

        if faces_in_frame > max_faces:
            max_faces = faces_in_frame

    cap.release()
    print(f"   ✅ Ditemukan maksimum {max_faces} wajah dalam satu frame.", flush=True)
    return max(1, max_faces)


# ==============================================================================
# FONT DOWNLOADER & REGISTRATION
# ==============================================================================


def download_google_font(
    url, output_filename, font_dir, max_retry=10, min_valid_size=1000
):
    """
    Download a Google font file with retry and basic integrity checks.

    Args:
        url: Font file URL.
        output_filename: Local filename to write under `font_dir`.
        font_dir: Destination directory for downloaded fonts.
        max_retry: Maximum retry attempts.
        min_valid_size: Minimum valid font size in bytes.

    Returns:
        True when a valid font file is available, otherwise False.
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
    """Copy fonts to system font dir and refresh cache (Linux only)."""
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
    Ensure required typography fonts are downloaded and registered.

    Args:
        cfg: Runtime config that contains active font style definitions.

    Raises:
        RuntimeError: If required fonts fail to download or validate.
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


# ==============================================================================
# BGM (PIXABAY)
# ==============================================================================


def resolve_pixabay_audio_url(page_url, timeout=45):
    """
    Resolve the direct Pixabay CDN audio URL from a track page URL.

    Args:
        page_url: Pixabay audio page URL.
        timeout: HTTP timeout in seconds.

    Returns:
        Direct downloadable MP3 URL from Pixabay CDN.

    Raises:
        RuntimeError: If no playable audio URL is found.
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
        page_url: Pixabay audio page URL.
        output_path: Destination MP3 path.
        max_retry: Maximum retry attempts.
        min_valid_size: Minimum valid file size in bytes.

    Returns:
        True when the output file is successfully downloaded and valid.
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


# ==============================================================================
# PEXELS B-ROLL
# ==============================================================================

USED_PEXELS_IDS = set()


def download_pexels_broll(query, rasio, output_filename, pexels_api_key):
    """
    Search and download one Pexels B-roll clip matching query and ratio.

    Args:
        query: Search query term.
        rasio: Target ratio string (`9:16` or `16:9`).
        output_filename: Destination video path.
        pexels_api_key: Pexels API key.

    Returns:
        True when download succeeds, otherwise False.
    """
    global USED_PEXELS_IDS

    if not pexels_api_key:
        print("   ⚠️ PEXELS_API_KEY tidak ditemukan. B-roll dilewati.")
        return False

    orientation = "portrait" if rasio == "9:16" else "landscape"

    params = urllib.parse.urlencode(
        {
            "query": query,
            "orientation": orientation,
            "per_page": 30,
            "size": "large",
            "resolution_name": "1080p",
        }
    )
    search_url = f"https://api.pexels.com/videos/search?{params}"

    req = urllib.request.Request(
        search_url,
        headers={
            "Authorization": pexels_api_key,
            "User-Agent": "Mozilla/5.0",
        },
    )

    try:
        with urllib.request.urlopen(req) as response:
            data = json.load(response)
    except Exception as e:
        print(f"   ⚠️ Error API Pexels saat mencari '{query}': {e}")
        return False

    if not data.get("videos"):
        print(f"   ⚠️ Pexels tidak menemukan video untuk '{query}'.")
        return False

    available_videos = [v for v in data["videos"] if v["id"] not in USED_PEXELS_IDS]
    if not available_videos:
        print(f"   🔄 B-roll pool untuk '{query}' habis, me-reset.")
        available_videos = data["videos"]

    video_data = random.choice(available_videos)
    USED_PEXELS_IDS.add(video_data["id"])

    video_files = [
        vf
        for vf in video_data.get("video_files", [])
        if vf.get("file_type") == "video/mp4"
    ]
    if not video_files:
        print(f"   ⚠️ Tidak ada file MP4 di dalam data video '{query}'.")
        return False

    video_files.sort(
        key=lambda vf: (
            vf.get("quality") != "hd",
            -(vf.get("width") or 0),
            -(vf.get("height") or 0),
        )
    )

    download_url = video_files[0]["link"]
    download_req = urllib.request.Request(
        download_url, headers={"User-Agent": "Mozilla/5.0"}
    )

    try:
        temp_path = output_filename + ".part"
        with (
            urllib.request.urlopen(download_req) as response,
            open(temp_path, "wb") as f,
        ):
            shutil.copyfileobj(response, f)
        os.replace(temp_path, output_filename)
        return True
    except Exception as e:
        print(f"   ⚠️ Error saat mengunduh B-roll '{query}': {e}")
        return False


def crop_center_broll(img, target_w, target_h):
    """
    Center-crop a frame to target aspect ratio and resize to target size.

    Args:
        img: Input image frame (NumPy array).
        target_w: Output width.
        target_h: Output height.

    Returns:
        Cropped and resized frame with target dimensions.
    """
    h, w = img.shape[:2]
    target_ratio = target_w / target_h
    img_ratio = w / h

    if img_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x = (w - new_w) // 2
        img = img[:, x : x + new_w]
    elif img_ratio < target_ratio:
        new_h = int(w / target_ratio)
        y = (h - new_h) // 2
        img = img[y : y + new_h, :]

    return _resize_frame(img, (target_w, target_h))


# ==============================================================================
# HYBRID VIDEO RENDERER
# ==============================================================================


def _get_render_dims(cfg, rasio, source_h=1080):
    """
    Calculate target output resolution based on config and aspect ratio.
    If mode is 'source', it uses the provided source_h as the base dimension.
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


def buat_video_hybrid(
    input_video,
    output_video,
    start_clip,
    end_clip,
    rasio,
    cfg,
    broll_data=None,
    label="Hybrid",
):
    """
    Render hybrid clip output with speaker tracking and optional B-roll.

    Args:
        input_video: Source video path.
        output_video: Destination video path.
        start_clip: Clip start time in seconds.
        end_clip: Clip end time in seconds.
        rasio: Output ratio string.
        cfg: Runtime config object.
        broll_data: Optional B-roll timeline metadata.
        label: Render label for progress logs.

    Returns:
        Callable that maps timestamps to subtitle X-position hints.
    """
    if broll_data is None:
        broll_data = []

    # =======================================================
    # 🎛️ PARAMETER TUNING KAMERA
    # =======================================================
    STEP_DETEKSI     = cfg.track_step if cfg.track_step is not None else 0.25   # AI mengecek wajah tiap 0.25 detik
    # STEP_DETEKSI     = 0.5   # AI mengecek wajah tiap 0.5 detik
    # STEP_DETEKSI     = max(0.5, (end_clip - start_clip) / 60.0)   # [OLD] AI mengecek wajah tiap max 0.5 atau sepanjang durasi (end_clip - start_clip) detik per menit

    DEADZONE_RATIO   = cfg.track_deadzone if cfg.track_deadzone is not None else 0.15  # 15% area tengah adalah zona aman (kamera tidak ikut gerak)
    # DEADZONE_RATIO   = 0.25  # 25% area tengah adalah zona aman (kamera tidak ikut gerak)
    # DEADZONE_RATIO   = 0.20  # [OLD] 20% area tengah adalah zona aman (kamera tidak ikut gerak)

    SMOOTH_FACTOR    = cfg.track_smooth if cfg.track_smooth is not None else 0.30  # Kecepatan kamera menyusul (30% jarak). Bikin pergerakan sangat mulus.
    # SMOOTH_FACTOR    = 0.15  # Kecepatan kamera menyusul (15% jarak). Bikin pergerakan sangat mulus.
    # SMOOTH_FACTOR    = 0.10  # [NEW; NOT USED]Kecepatan kamera menyusul (10% jarak). Bikin pergerakan sangat mulus.

    JITTER_THRESHOLD = cfg.track_jitter if cfg.track_jitter is not None else 5     # Abaikan pergeseran di bawah 5 pixel (Anti-getar/Micro-jitter)
    # JITTER_THRESHOLD = 4     # [OLD] Abaikan pergeseran di bawah 4 pixel (Anti-getar/Micro-jitter)

    SNAP_THRESHOLD   = cfg.track_snap if cfg.track_snap is not None else 0.25  # Jika wajah lompat > 25% lebar layar, anggap ganti orang (Hard Cut)
    # SNAP_THRESHOLD   = 0.30  # [NEW; NOT USED] Jika wajah lompat > 30% lebar layar, anggap ganti orang (Hard Cut)
    # =======================================================

    video_encoder = detect_video_encoder(cfg)

    yolo_model = None
    detector = None
    if cfg.face_detector == "yolo":
        if not os.path.exists(cfg.file_yolo_model):
            print(f"   📥 Mendownload YOLOv8 Face Model ({cfg.yolo_size})...")
            import urllib.request

            urllib.request.urlretrieve(cfg.url_yolo_model, cfg.file_yolo_model)
        from ultralytics import YOLO

        yolo_model = YOLO(cfg.file_yolo_model)
    else:
        detector = get_face_detector(cfg)

    cap = cv2.VideoCapture(input_video)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if math.isnan(orig_fps) or orig_fps == 0:
        orig_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop_w = int(height * 9 / 16)
    crop_h = height
    default_cx = width // 2
    default_cy = height // 2
    duration = end_clip - start_clip

    broll_caps = []
    for br in broll_data:
        if "filepath" in br and os.path.exists(br["filepath"]):
            broll_caps.append(
                {
                    "start": br["start_time"],
                    "end": br["end_time"],
                    "cap": cv2.VideoCapture(br["filepath"]),
                }
            )

    # FASE 1: DETEKSI WAJAH
    raw_data = []
    current_time = 0.0
    last_detect_percent = -1
    print(f"🧠 {label} - Analisa wajah dimulai...", flush=True)

    while current_time <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, (start_clip + current_time) * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        best_x = default_x
        face_box = None

        if cfg.face_detector == "yolo":
            yolo_results = yolo_model(frame, verbose=False)
            if yolo_results and len(yolo_results[0].boxes) > 0:
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                largest_idx = areas.argmax()
                x1, y1, x2, y2 = boxes[largest_idx]
                center_x = x1 + (x2 - x1) / 2
                center_y = y1 + (y2 - y1) / 2
                face_box = (x1, y1, x2, y2)
        else:
            results = detector.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
            )

            if results.detections:
                largest_face = max(
                    results.detections,
                    key=lambda d: d.bounding_box.width * d.bounding_box.height,
                ).bounding_box
                center_x = largest_face.origin_x + (largest_face.width / 2)
                center_y = largest_face.origin_y + (largest_face.height / 2)
                face_box = (
                    largest_face.origin_x,
                    largest_face.origin_y,
                    largest_face.origin_x + largest_face.width,
                    largest_face.origin_y + largest_face.height,
                )

        raw_data.append(
            {
                "time": current_time,
                "cx": center_x if face_box else default_cx,
                "cy": center_y if face_box else default_cy,
                "box": face_box,
            }
        )

        detect_percent = (
            min(100, int((current_time / duration) * 100)) if duration > 0 else 100
        )
        if detect_percent != last_detect_percent:
            print(f"⏳ {label} - Analisa wajah: {detect_percent:3d}%", flush=True)
            last_detect_percent = detect_percent

        current_time += STEP_DETEKSI

    # FASE 2: SMOOTH CAMERA
    smooth_data = []
    if raw_data:
        cam_cx = raw_data[0]["cx"]
        cam_cy = raw_data[0]["cy"]
        
        deadzone_px = crop_w * DEADZONE_RATIO
        snap_px = width * SNAP_THRESHOLD

        for d in raw_data:
            face_cx = d["cx"]
            face_cy = d["cy"]

            if abs(face_cx - cam_cx) > snap_px:
                cam_cx = face_cx
            else:
                if face_cx > cam_cx + deadzone_px:
                    cam_cx += (face_cx - (cam_cx + deadzone_px)) * SMOOTH_FACTOR
                elif face_cx < cam_cx - deadzone_px:
                    cam_cx += (face_cx - (cam_cx - deadzone_px)) * SMOOTH_FACTOR

            # Vertical smoothing
            cam_cy += (face_cy - cam_cy) * SMOOTH_FACTOR

            smooth_data.append({"time": d["time"], "cx": cam_cx, "cy": cam_cy})

    def get_x(t):
        if not smooth_data:
            return default_cx
        if t <= smooth_data[0]["time"]:
            return smooth_data[0]["cx"]
        if t >= smooth_data[-1]["time"]:
            return smooth_data[-1]["cx"]
        for i in range(len(smooth_data) - 1):
            if smooth_data[i]["time"] <= t <= smooth_data[i + 1]["time"]:
                t1, t2 = smooth_data[i]["time"], smooth_data[i + 1]["time"]
                cx1, cx2 = smooth_data[i]["cx"], smooth_data[i + 1]["cx"]
                if t1 == t2: return cx1
                return cx1 + (cx2 - cx1) * (t - t1) / (t2 - t1)
        return default_cx

    def _get_pos(t):
        if not smooth_data:
            return default_cx, default_cy
        if t <= smooth_data[0]["time"]:
            return smooth_data[0]["cx"], smooth_data[0]["cy"]
        if t >= smooth_data[-1]["time"]:
            return smooth_data[-1]["cx"], smooth_data[-1]["cy"]

        for i in range(len(smooth_data) - 1):
            if smooth_data[i]["time"] <= t <= smooth_data[i + 1]["time"]:
                t1, t2 = smooth_data[i]["time"], smooth_data[i + 1]["time"]
                cx1, cx2 = smooth_data[i]["cx"], smooth_data[i + 1]["cx"]
                cy1, cy2 = smooth_data[i]["cy"], smooth_data[i + 1]["cy"]
                if t1 == t2:
                    return cx1, cy1
                frac = (t - t1) / (t2 - t1)
                return (
                    cx1 + (cx2 - cx1) * frac,
                    cy1 + (cy2 - cy1) * frac
                )
        return default_cx, default_cy

    # FASE 3: RENDER FRAME
    out_w, out_h = _get_render_dims(cfg, rasio, source_h=height)
    
    # DEV MODE: Force 16:9 to show context
    dev_visualize = cfg.dev_mode and rasio == "9:16"
    if dev_visualize:
        out_w, out_h = (1920, 1080)

    writer = open_ffmpeg_video_writer(
        output_video, out_w, out_h, orig_fps, video_encoder
    )

    TRANSITION_DUR = 0.3
    MAX_ZOOM = 1.10

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_clip * 1000)
        frame_count = 0
        last_render_percent = -1

        print(f"🎬 {label} - Render frame dimulai...", flush=True)

        while True:
            ret, frame_utama = cap.read()
            if not ret:
                break

            t = frame_count / orig_fps
            if t > duration:
                break

            waktu_absolut = start_clip + t

            if dev_visualize:
                # Render full context for dev mode
                cx = get_x(t)
                # Resize original to 16:9 canvas
                frame_base = _resize_frame(frame_utama, (out_w, out_h))
                
                # Dim background
                frame_dev = (frame_base * 0.35).astype(np.uint8)
                
                # Highlight 9:16 window
                # Current source width/height vs canvas out_w/out_h
                scale_x = out_w / width
                scale_y = out_h / height
                
                cx, cy = _get_pos(t)
                cx_scaled = int(cx * scale_x)
                cy_scaled = int(cy * scale_y)
                cw_scaled = int(crop_w * scale_x)
                ch_scaled = int(crop_h * scale_y)
                
                x1 = int(max(0, min(cx_scaled - cw_scaled // 2, out_w - cw_scaled)))
                y1 = int(max(0, min(cy_scaled - ch_scaled // 2, out_h - ch_scaled)))
                
                # Paste bright crop
                frame_dev[y1 : y1 + ch_scaled, x1 : x1 + cw_scaled] = frame_base[y1 : y1 + ch_scaled, x1 : x1 + cw_scaled]
                
                # Draw border lines
                cv2.rectangle(frame_dev, (x1, y1), (x1+cw_scaled, y1+ch_scaled), (255, 255, 255), 2)
                
                # Draw face box if detected
                if cfg.box_face_detection or cfg.track_lines or True: # Force in dev mode
                    box = get_box(t)
                    if box:
                        scale_y = out_h / height
                        bx1, by1 = int(box[0] * scale_x), int(box[1] * scale_y)
                        bx2, by2 = int(box[2] * scale_x), int(box[3] * scale_y)
                        cv2.rectangle(frame_dev, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                        
                        if cfg.track_lines or cfg.dev_mode:
                            # Center points of sides
                            mid_x = (bx1 + bx2) // 2
                            mid_y = (by1 + by2) // 2
                            
                            # Horizontal lines to 9:16 boundaries
                            cv2.line(frame_dev, (cx_scaled, mid_y), (bx1, mid_y), (0, 255, 255), 2)
                            cv2.line(frame_dev, (bx2, mid_y), (cx_scaled + cw_scaled, mid_y), (0, 255, 255), 2)
                            
                            # Vertical lines to frame boundaries (top/bottom)
                            cv2.line(frame_dev, (mid_x, 0), (mid_x, by1), (0, 255, 255), 2)
                            cv2.line(frame_dev, (mid_x, by2), (mid_x, out_h), (0, 255, 255), 2)
                
                frame_terpilih = frame_dev
                frame_utama_siap = frame_dev # Ensure this is defined for B-roll transitions

            elif rasio == "9:16":
                cx, cy = _get_pos(t)
                x1 = int(max(0, min(cx - crop_w // 2, width - crop_w)))
                y1 = int(max(0, min(cy - crop_h // 2, height - crop_h)))
                cropped = frame_utama[y1 : y1 + crop_h, x1 : x1 + crop_w]
                frame_utama_siap = _resize_frame(cropped, (out_w, out_h))
                frame_terpilih = frame_utama_siap
            else:
                frame_utama_siap = _resize_frame(frame_utama, (out_w, out_h))
                frame_terpilih = frame_utama_siap

            for bc in broll_caps:
                if bc["start"] <= waktu_absolut <= bc["end"]:
                    elapsed_broll = waktu_absolut - bc["start"]
                    bc["cap"].set(cv2.CAP_PROP_POS_MSEC, elapsed_broll * 1000)
                    ret_b, frame_b = bc["cap"].read()

                    if ret_b:
                        durasi_total_broll = bc["end"] - bc["start"]
                        progress_broll = (
                            elapsed_broll / durasi_total_broll
                            if durasi_total_broll > 0
                            else 0
                        )
                        zoom_factor = 1.0 + ((MAX_ZOOM - 1.0) * progress_broll)

                        frame_b_crop = crop_center_broll(frame_b, out_w, out_h)
                        center_x, center_y = out_w / 2, out_h / 2
                        M = cv2.getRotationMatrix2D(
                            (center_x, center_y), 0, zoom_factor
                        )
                        frame_b_zoomed = cv2.warpAffine(frame_b_crop, M, (out_w, out_h))

                        alpha = 1.0
                        if elapsed_broll < TRANSITION_DUR:
                            alpha = elapsed_broll / TRANSITION_DUR
                        elif (bc["end"] - waktu_absolut) < TRANSITION_DUR:
                            alpha = (bc["end"] - waktu_absolut) / TRANSITION_DUR

                        if alpha >= 1.0:
                            frame_terpilih = frame_b_zoomed
                        else:
                            frame_terpilih = cv2.addWeighted(
                                frame_b_zoomed, alpha, frame_utama_siap, 1.0 - alpha, 0
                            )

                    break

            writer.stdin.write(frame_terpilih.tobytes())
            frame_count += 1

            render_percent = (
                min(100, int((t / duration) * 100)) if duration > 0 else 100
            )
            if render_percent != last_render_percent:
                print(
                    f"⏳ {label} - Render frame: {render_percent:3d}% | "
                    f"{format_seconds(t)} / {format_seconds(duration)}",
                    flush=True,
                )
                last_render_percent = render_percent

        writer.stdin.close()
        stderr_data = writer.stderr.read().decode("utf-8", errors="ignore")
        return_code = writer.wait()

        if return_code != 0:
            raise RuntimeError(f"FFmpeg writer gagal: {stderr_data[-1000:]}")

        print(f"✅ {label} selesai.", flush=True)

    finally:
        cap.release()
        for bc in broll_caps:
            bc["cap"].release()
            
    return get_x


# ==============================================================================
# PEMBUAT SUBTITLE ASS
# ==============================================================================


def buat_file_ass(
    data_segmen,
    start_clip,
    end_clip,
    nama_file_ass,
    rasio,
    cfg,
    typography_plan=None,
    gunakan_advanced=True,
    get_x_func=None,
    source_dim=None,
):
    """
    Build and write ASS subtitle file for a clip segment range.

    Args:
        data_segmen: Full transcript segments.
        start_clip: Clip start time in seconds.
        end_clip: Clip end time in seconds.
        nama_file_ass: Output ASS file path.
        rasio: Output ratio string.
        cfg: Runtime config object.
        typography_plan: Optional typography emphasis plan.
        gunakan_advanced: Toggle advanced subtitle styling.
        get_x_func: Optional callback for dynamic subtitle X-position.
        source_dim: Optional source video dimension tuple.

    Side Effects:
        Writes subtitle content to `nama_file_ass`.
    """
    if typography_plan is None:
        typography_plan = []

    typo_dict = {}
    for plan in typography_plan:
        clean_word = plan.get("kata_utama", "").lower().strip(string.punctuation)
        typo_dict[clean_word] = plan

    pakai_advanced = cfg.use_advanced_text and gunakan_advanced
    pakai_karaoke = cfg.use_karaoke_effect

    outline_val = 3 if pakai_karaoke else 0.2
    shadow_val = 2.5 if pakai_karaoke else 0.2

    daftar_font = cfg.daftar_font
    gaya = cfg.gaya_font_aktif
    font_dir = cfg.font_dir

    font_utama_dict = daftar_font[gaya]["utama"]
    font_khusus_dict = daftar_font[gaya]["khusus"]

    font_utama = font_utama_dict["nama"]
    font_khusus = font_khusus_dict["nama"]

    scale_base_khusus = (
        cfg.scale_kata_khusus_916 if rasio == "9:16" else cfg.scale_kata_khusus_169
    )
    warna_khusus = cfg.warna_kata_khusus

    def get_scale_value(level):
        if level == 3:
            return scale_base_khusus
        elif level == 2:
            return int((scale_base_khusus + 100) / 2)
        else:
            return 110

    def fmt_time(d):
        return f"{int(d // 3600)}:{int((d % 3600) // 60):02d}:{int(d % 60):02d}.{int((d - int(d)) * 100):02d}"

    play_res_x, play_res_y = _get_render_dims(cfg, rasio, source_h=source_dim[1] if source_dim else 1080)
    
    # Calculate scale relative to standard 1080p vertical (1920 height)
    # This ensures typography looks consistent across different render resolutions
    scale_factor = play_res_y / (1920 if rasio == "9:16" else 1080)
    align = cfg.ass_align_916 if rasio == "9:16" else cfg.ass_align_169
    margin_v = int((cfg.ass_margin_916 if rasio == "9:16" else cfg.ass_margin_169) * scale_factor)
    font_sz = int((cfg.ass_font_916 if rasio == "9:16" else cfg.ass_font_169) * scale_factor)
    margin_lr = int((60 if rasio == "9:16" else 40) * scale_factor)

    header = (
        f"[Script Info]\n"
        f"PlayResX: {play_res_x}\n"
        f"PlayResY: {play_res_y}\n"
        f"WrapStyle: 1\n"
        f"ScriptType: v4.00+\n"
        f"ScaledBorderAndShadow: yes\n\n"
        f"[V4+ Styles]\n"
        f"Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_utama},{font_sz},&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,{outline_val},{shadow_val},{align},{margin_lr},{margin_lr},{margin_v},1\n\n"
        f"[Events]\n"
        f"Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    def cek_font_di_folder(nama_file, min_valid_size=1000):
        path = os.path.join(font_dir, nama_file)
        return os.path.exists(path) and os.path.getsize(path) > min_valid_size

    if not pakai_advanced:
        with open(nama_file_ass, "w", encoding="utf-8") as f:
            f.write(header)
            for seg in data_segmen:
                seg_s = max(0, seg["start"] - start_clip)
                seg_e = min(end_clip - start_clip, seg["end"] - start_clip)
                if seg_s >= seg_e:
                    continue

                for i, w in enumerate(seg["words"]):
                    w_s = max(0, w["start"] - start_clip)
                    if i < len(seg["words"]) - 1:
                        w_e = min(
                            end_clip - start_clip,
                            seg["words"][i + 1]["start"] - start_clip,
                        )
                    else:
                        w_e = min(end_clip - start_clip, w["end"] - start_clip)

                    if w_s < w_e:
                        text_parts = []
                        for j, x in enumerate(seg["words"]):
                            if pakai_karaoke:
                                if j == i:
                                    text_parts.append(
                                        f"{{\\c&H00FFFF&}}{x['word']}{{\\c&HFFFFFF&}}"
                                    )
                                else:
                                    text_parts.append(x["word"])
                            else:
                                if j <= i:
                                    text_parts.append(x["word"])
                                else:
                                    text_parts.append(
                                        f"{{\\alpha&HFF&}}{x['word']}{{\\alpha&H00&}}"
                                    )

                        f.write(
                            f"Dialogue: 0,{fmt_time(w_s)},{fmt_time(w_e)},Default,,0,0,0,,{' '.join(text_parts)}\n"
                        )
        return

    # Advanced typography mode
    font_cache = {}

    def get_cached_font(is_khusus, scale_val):
        key = f"{is_khusus}_{scale_val}"
        if key not in font_cache:
            f_info = font_khusus_dict if is_khusus else font_utama_dict
            f_file = f_info["file"]
            f_path = os.path.join(font_dir, f_file)

            if not cek_font_di_folder(f_file):
                raise FileNotFoundError(f"Font tidak ditemukan: {f_path}")

            font_cache[key] = ImageFont.truetype(
                f_path,
                int(font_sz * (scale_val / 100.0)),
            )
        return font_cache[key]

    def build_font_tag(font_info):
        nama = str(font_info["nama"]).replace("{", "").replace("}", "").strip()
        bold = 1 if int(font_info.get("bold", 0)) else 0
        return f"\\fn{nama}\\b{bold}"

    max_line_width = play_res_x - (margin_lr * 2)
    space_width = font_sz * 0.25
    TIGHTNESS = 0.95

    with open(nama_file_ass, "w", encoding="utf-8") as f:
        f.write(header)

        for seg in data_segmen:
            seg_s = max(0, seg["start"] - start_clip)
            seg_e = min(end_clip - start_clip, seg["end"] - start_clip)
            if seg_s >= seg_e:
                continue

            lines = []
            current_line = []
            current_w = 0
            max_line_h = 0

            for w_dict in seg["words"]:
                word_clean = w_dict["word"].lower().strip(string.punctuation)
                plan = typo_dict.get(word_clean)

                if plan:
                    w_style = plan.get("style", "khusus")
                    w_scale = get_scale_value(plan.get("scale_level", 2))
                    is_khusus = w_style == "khusus"

                    pil_font = get_cached_font(is_khusus, w_scale)
                    raw_w = (
                        pil_font.getlength(w_dict["word"])
                        if hasattr(pil_font, "getlength")
                        else len(w_dict["word"]) * 20
                    )
                    w_len = raw_w * TIGHTNESS
                    h_len = font_sz * (w_scale / 100.0)
                else:
                    w_scale = 100
                    pil_font = get_cached_font(False, w_scale)
                    raw_w = (
                        pil_font.getlength(w_dict["word"])
                        if hasattr(pil_font, "getlength")
                        else len(w_dict["word"]) * 15
                    )
                    w_len = raw_w * TIGHTNESS
                    h_len = font_sz

                if current_line and (current_w + space_width + w_len > max_line_width):
                    lines.append(
                        {
                            "words": current_line,
                            "width": current_w,
                            "height": max_line_h,
                        }
                    )
                    current_line = []
                    current_w = 0
                    max_line_h = 0

                x_offset = current_w if not current_line else current_w + space_width
                current_line.append(
                    {
                        "text": w_dict["word"],
                        "plan": plan,
                        "w": w_len,
                        "h": h_len,
                        "x_offset": x_offset,
                        "start": max(0, w_dict["start"] - start_clip),
                        "end": min(end_clip - start_clip, w_dict["end"] - start_clip),
                    }
                )

                current_w = x_offset + w_len
                max_line_h = max(max_line_h, h_len)

            if current_line:
                lines.append(
                    {"words": current_line, "width": current_w, "height": max_line_h}
                )

            line_spacing = 15
            total_stack_h = (
                sum(l["height"] for l in lines) + (len(lines) - 1) * line_spacing
            )
            current_y = play_res_y - margin_v - total_stack_h

            for line in lines:
                start_x = (play_res_x - line["width"]) / 2
                
                if get_x_func and cfg.dev_mode and source_dim:
                    sw, sh = source_dim
                    # Reference time: midpoint of the current segment/line
                    t_ref = line["words"][0]["start"] + start_clip
                    center_x_src = get_x_func(t_ref)
                    # Target center in PlayResX (1920)
                    target_center_x = center_x_src * (play_res_x / sw)
                    start_x = target_center_x - (line["width"] / 2)

                line_y = current_y + line["height"]

                for w_data in line["words"]:
                    word_x = start_x + w_data["x_offset"] + (w_data["w"] / 2)
                    w_appear_ms = int((w_data["start"] - seg_s) * 1000)
                    w_end_ms = int((w_data["end"] - seg_s) * 1000)

                    if w_data["plan"]:
                        w_style = w_data["plan"].get("style", "khusus")
                        w_anim = w_data["plan"].get("animasi", "bounce_pop")
                        target_scale = get_scale_value(
                            w_data["plan"].get("scale_level", 2)
                        )
                        font_info = (
                            font_khusus_dict if w_style == "khusus" else font_utama_dict
                        )
                        f_tag = build_font_tag(font_info)
                        c_tag = f"\\c{warna_khusus}"
                    else:
                        w_anim = "none"
                        target_scale = 100
                        f_tag = build_font_tag(font_utama_dict)
                        c_tag = "\\c&HFFFFFF&"

                    t_start = w_appear_ms
                    t_pop = w_appear_ms + 80
                    t_settle = w_appear_ms + 150

                    if pakai_karaoke:
                        pos_tag = f"\\pos({int(word_x)},{int(line_y)})"
                        c_tag = "\\c&HFFFFFF&"
                        anim_tag = f"\\fscx{target_scale}\\fscy{target_scale}\\t({t_start},{t_start},\\c&H00FFFF&)\\t({w_end_ms},{w_end_ms},\\c&HFFFFFF&)"
                    else:
                        if w_anim == "stagger_up":
                            y_start = int(line_y + 30)
                            pos_tag = f"\\move({int(word_x)},{y_start},{int(word_x)},{int(line_y)},{t_start},{t_settle})"
                            anim_tag = f"\\alpha&HFF&\\fscx{target_scale}\\fscy{target_scale}\\t({t_start},{t_start},\\alpha&H00&)"
                        elif w_anim == "bounce_pop":
                            init_scale = int(target_scale * 0.7)
                            overshoot = int(target_scale * 1.15)
                            pos_tag = f"\\pos({int(word_x)},{int(line_y)})"
                            anim_tag = (
                                f"\\alpha&HFF&\\fscx{init_scale}\\fscy{init_scale}"
                                f"\\t({t_start},{t_start},\\alpha&H00&)"
                                f"\\t({t_start},{t_pop},\\fscx{overshoot}\\fscy{overshoot})"
                                f"\\t({t_pop},{t_settle},\\fscx{target_scale}\\fscy{target_scale})"
                            )
                        else:
                            pos_tag = f"\\pos({int(word_x)},{int(line_y)})"
                            anim_tag = f"\\alpha&HFF&\\fscx{target_scale}\\fscy{target_scale}\\t({t_start},{t_start},\\alpha&H00&)"

                    event_text = (
                        f"{{\\an2{pos_tag}{f_tag}{c_tag}{anim_tag}}}{w_data['text']}"
                    )
                    f.write(
                        f"Dialogue: 0,{fmt_time(seg_s)},{fmt_time(seg_e)},Default,,0,0,0,,{event_text}\n"
                    )

                current_y += line["height"] + line_spacing


# ==============================================================================
# GLITCH & THUMBNAIL
# ==============================================================================


def siapkan_glitch_video(rasio, cfg, video_encoder, source_h=1080, custom_dims=None):
    """
    Prepare or reuse short glitch transition video in TS format.

    Args:
        rasio: Target ratio string.
        cfg: Runtime config with glitch source URL.
        video_encoder: Encoder descriptor dict.
        source_h: Height of the source video for dynamic scaling.
        custom_dims: Optional (w, h) to override ratio-based dims.

    Returns:
        Path to prepared glitch TS file.
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


def buat_thumbnail(video_path, output_image_path, teks, cfg):
    """
    Generate a thumbnail image from a video frame with overlay text.

    Args:
        video_path: Source video path.
        output_image_path: Destination image path.
        teks: Thumbnail title text.
        cfg: Runtime config with thumbnail font settings.

    Side Effects:
        Downloads thumbnail font when missing and writes output image.
    """
    if not os.path.exists(cfg.file_font_thumbnail):
        urllib.request.urlretrieve(cfg.url_font_thumbnail, cfg.file_font_thumbnail)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return

    img = Image.alpha_composite(
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA"),
        Image.new("RGBA", (frame.shape[1], frame.shape[0]), (0, 0, 0, 128)),
    ).convert("RGB")

    draw = ImageDraw.Draw(img)
    font_sz = int(img.size[0] * 0.12)
    font = ImageFont.truetype(cfg.file_font_thumbnail, font_sz)
    lines = textwrap.wrap(teks, width=12)

    y_text = (img.size[1] - (len(lines) * (font_sz + 10))) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        x_text = (img.size[0] - line_w) // 2
        draw.text(
            (x_text, y_text),
            line,
            font=font,
            fill="white",
            stroke_width=5,
            stroke_fill="black",
        )
        y_text += font_sz + 10

    img.save(output_image_path)


# ==============================================================================
# SPLIT-SCREEN RENDERER (PODCAST 2 SPEAKER)
# ==============================================================================


def buat_video_split_screen(
    input_video,
    output_video,
    start_clip,
    end_clip,
    diarization_data,
    cfg,
    label="SplitScreen",
):
    """
    Render a split-screen (top-bottom) 9:16 video for 2-speaker podcasts.

    Each panel shows one speaker's face-tracked crop. The active speaker's
    panel is brighter; the inactive speaker's panel has a subtle dark overlay.

    Parameters
    ----------
    input_video : str
        Path to the source video.
    output_video : str
        Output video path (silent, no audio).
    start_clip, end_clip : float
        Clip time range in seconds.
    diarization_data : list[dict]
        Speaker segments from diarization.run_diarization().
    cfg : SimpleNamespace
        Configuration object.
    label : str
        Label for progress logging.
    """
    from clipping.diarization import get_active_speaker, get_active_speakers

    STEP_DETEKSI     = cfg.track_step if cfg.track_step is not None else 0.25
    DEADZONE_RATIO   = cfg.track_deadzone if cfg.track_deadzone is not None else 0.15
    SMOOTH_FACTOR    = cfg.track_smooth if cfg.track_smooth is not None else 0.30
    JITTER_THRESHOLD = cfg.track_jitter if cfg.track_jitter is not None else 5
    SNAP_THRESHOLD   = cfg.track_snap if cfg.track_snap is not None else 0.25
    DIVIDER_HEIGHT = 4  # px, divider between panels
    INACTIVE_ALPHA = 0.15  # darkening for inactive speaker panel
    ACTIVE_BORDER = 3  # px, highlight border for active speaker

    video_encoder = detect_video_encoder(cfg)

    # Setup face detector
    yolo_model = None
    detector = None
    if cfg.face_detector == "yolo":
        if not os.path.exists(cfg.file_yolo_model):
            print(f"   📥 Mendownload YOLOv8 Face Model ({cfg.yolo_size})...")
            import urllib.request

            urllib.request.urlretrieve(cfg.url_yolo_model, cfg.file_yolo_model)
        from ultralytics import YOLO

        yolo_model = YOLO(cfg.file_yolo_model)
    else:
        detector = get_face_detector(cfg)

    cap = cv2.VideoCapture(input_video)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if math.isnan(orig_fps) or orig_fps == 0:
        orig_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = end_clip - start_clip

    # Output dimensions calculated dynamically
    out_w, out_h = _get_render_dims(cfg, "9:16", source_h=height)
    out_w_final, out_h_final = out_w, out_h
    
    dev_visualize = cfg.dev_mode or cfg.dev_mode_with_output or cfg.dev_mode_with_output_merge
    dual_output = cfg.dev_mode_with_output
    merge_output = cfg.dev_mode_with_output_merge
    
    if merge_output:
        # Merged Full Padded Canvas: 2648 x 1220
        out_w_final, out_h_final = 2648, 1220
    elif dev_visualize and not dual_output:
        # Pure Dev Mode (override output)
        out_w_final, out_h_final = 1920, 1080
    
    # Calculate panel dimensions based on the 1080x1920 orientation
    # regardless of whether dev mode is on, because the internal layout arithmetic
    # must still think in terms of the target 9:16 portrait canvas.
    panel_h = (out_h - DIVIDER_HEIGHT) // 2
    panel_w = out_w

    # --- Panel Crop dimensions (wider aspect for half-height panels) ---
    panel_ratio = panel_w / panel_h
    if (width / height) > panel_ratio:
        crop_h = height
        crop_w = int(height * panel_ratio)
    else:
        crop_w = width
        crop_h = int(width / panel_ratio)

    # --- Full 9:16 Crop dimensions (for solo mode) ---
    full_ratio = out_w / out_h
    if (width / height) > full_ratio:
        crop_h_full = height
        crop_w_full = int(height * full_ratio)
    else:
        crop_w_full = width
        crop_h_full = int(width / full_ratio)

    zoom = getattr(cfg, "split_zoom", 1.0)

    default_x = (width - crop_w) // 2
    default_y = (height - crop_h) // 2
    default_x_full = (width - crop_w_full) // 2

    # ----------------------------------------------------------------
    # Determine top/bottom panel speakers
    # ----------------------------------------------------------------
    if diarization_data:
        all_speakers_in_clip = sorted(set(s["speaker"] for s in diarization_data))
        speaking_time: dict[str, float] = {spk: 0.0 for spk in all_speakers_in_clip}
        for seg in diarization_data:
            eff_start = max(seg["start"], start_clip)
            eff_end = min(seg["end"], end_clip)
            if eff_end > eff_start:
                speaking_time[seg["speaker"]] += eff_end - eff_start

        ranked = sorted(all_speakers_in_clip, key=lambda s: speaking_time[s], reverse=True)
        speaker_top = ranked[0]
        speaker_bottom = ranked[1] if len(ranked) > 1 else ranked[0]
        extra_speakers = ranked[2:]
    else:
        # Visual-only mode: use generic labels
        all_speakers_in_clip = ["FACE_L", "FACE_R"]
        speaker_top = "FACE_L"
        speaker_bottom = "FACE_R"
        ranked = all_speakers_in_clip
        extra_speakers = []

    # ---- FASE 1: DETECT ALL FACES & ASSIGN TO SPEAKERS (diarization-guided) ----
    # Strategy:
    #   1 active + 1 face  → trivial: face belongs to active speaker
    #   N active + N faces → sort faces by X; sort active speakers by label; assign in order
    #   1 active + N faces → use speaker's last-known position to pick nearest face
    #   otherwise          → skip (ambiguous or no data)

    print(f"🧠 {label} - Analisa wajah (split-screen) dimulai...", flush=True)

    all_frame_data: list[dict] = []  # [{time, face_centers, face_boxes, active_now}]
    speaker_solo_cxs: dict[str, list] = {}  # speaker → [cx, ...] from 1:1 frames
    solo_counts: dict[str, int] = {spk: 0 for spk in all_speakers_in_clip}
    multi_counts: dict[str, int] = {spk: 0 for spk in all_speakers_in_clip}
    current_time = 0.0
    last_detect_percent = -1

    def _clamp_x(cx_center: float) -> int:
        return max(0, min(int(cx_center - crop_w / 2), width - crop_w))

    while current_time <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, (start_clip + current_time) * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        face_centers = []
        face_boxes = []

        if cfg.face_detector == "yolo":
            # Higher confidence to filter background noise (microphones, reflections)
            det_conf = getattr(cfg, "track_conf", 0.55)
            yolo_results = yolo_model(frame, verbose=False, conf=det_conf)
            if yolo_results and len(yolo_results[0].boxes) > 0:
                raw_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                
                # IoU filter to merge overlapping boxes for one person
                def compute_iou(b1, b2):
                    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
                    area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
                    return inter / (area1 + area2 - inter + 1e-6)

                final_boxes = []
                for rb in raw_boxes:
                    merged = False
                    for i, fb in enumerate(final_boxes):
                        iou_thresh = getattr(cfg, "track_iou_threshold", 0.2)
                        if compute_iou(rb, fb) > iou_thresh: # More aggressive merging
                            # Keep the larger box
                            if (rb[2]-rb[0])*(rb[3]-rb[1]) > (fb[2]-fb[0])*(fb[3]-fb[1]):
                                final_boxes[i] = rb
                            merged = True
                            break
                    if not merged:
                        final_boxes.append(rb)

                for box in final_boxes:
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    face_centers.append((cx, cy))
                    face_boxes.append((x1, y1, x2, y2))
        else:
            results = detector.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
            )
            if results.detections:
                for d in results.detections:
                    bb = d.bounding_box
                    cx = bb.origin_x + bb.width / 2
                    cy = bb.origin_y + bb.height / 2
                    face_centers.append((cx, cy))
                    face_boxes.append(
                        (
                            bb.origin_x,
                            bb.origin_y,
                            bb.origin_x + bb.width,
                            bb.origin_y + bb.height,
                        )
                    )

        face_centers.sort(key=lambda fc: fc[0])  # left → right
        
        if diarization_data:
            from clipping.diarization import get_active_speakers
            active_now = get_active_speakers(diarization_data, start_clip + current_time)
        else:
            # Visual-only: base active speakers on face count
            if len(face_centers) == 1:
                active_now = ["FACE_L"]
            elif len(face_centers) >= 2:
                active_now = ["FACE_L", "FACE_R"]
            else:
                active_now = []

        n_faces = len(face_centers)
        n_active = len(active_now)

        # Pass A: collect face data and track canonical solo observations.
        # Full assignment happens in Pass B after canonical_cx is built from entire clip.
        if n_faces == 1 and n_active == 1:
            spk = active_now[0]
            speaker_solo_cxs.setdefault(spk, []).append((face_centers[0][0], face_centers[0][1]))

            solo_counts[spk] = solo_counts.get(spk, 0) + 1
        elif n_faces >= 2 and n_active >= 1:
            for spk in active_now:
                multi_counts[spk] = multi_counts.get(spk, 0) + 1

        all_frame_data.append(
            {
                "time": current_time,
                "face_centers": face_centers,
                "face_boxes": face_boxes,
                "active_now": active_now,
            }
        )

        detect_percent = (
            min(100, int((current_time / duration) * 100)) if duration > 0 else 100
        )
        if detect_percent != last_detect_percent:
            print(f"⏳ {label} - Analisa wajah: {detect_percent:3d}%", flush=True)
            last_detect_percent = detect_percent

        current_time += STEP_DETEKSI

    # Build canonical center-X per speaker from 1:1 frames (median, robust to outliers)
    import statistics as _stats

    speaker_canonical_cx: dict[str, float] = {
        spk: _stats.median([c[0] for c in cxs]) for spk, cxs in speaker_solo_cxs.items() if cxs
    }

    if not diarization_data:
        # Visual-only: assume L is left and R is right
        if "FACE_L" not in speaker_canonical_cx:
            speaker_canonical_cx["FACE_L"] = width * 0.25
        if "FACE_R" not in speaker_canonical_cx:
            speaker_canonical_cx["FACE_R"] = width * 0.75

    # Determine which speakers appear mostly in solo-scene frames
    speaker_is_solo: dict[str, bool] = {
        spk: (solo_counts.get(spk, 0) > multi_counts.get(spk, 0))
        for spk in all_speakers_in_clip
    }

    # ================================================================
    # Pass B — Canonical-guided face-to-speaker assignment
    # ================================================================
    # Re-process all stored frames using canonical_cx built from the entire clip.
    # This handles multi-scene videos correctly: e.g. in a 3-speaker podcast where
    # Speaker C's solo canonical_cx (center of Scene B) can be used to *eliminate*
    # that face from Scene A frames, leaving the correct face for Speaker A/B.
    # ================================================================
    raw_data: dict[str, list] = {spk: [] for spk in all_speakers_in_clip}

    # Helper for visual-only assignment
    def _get_canonical_x(spk):
        if diarization_data:
            return speaker_canonical_cx.get(spk, width / 2)
        else:
            return 0 if spk == "FACE_L" else width

    for fd in all_frame_data:
        fc_list = fd["face_centers"]  # list of (cx, cy), sorted left → right
        act_list = fd["active_now"]
        nf = len(fc_list)
        na = len(act_list)

        if nf == 0 or na == 0:
            continue

        if nf == 1 and na == 1:
            spk = act_list[0]
            if spk in raw_data:
                others = [fc for fc in fc_list if fc != fc_list[0]]
                d_near = min([abs(fc[0] - fc_list[0][0]) for fc in others]) if others else width
                raw_data[spk].append({"time": fd["time"], "cx": fc_list[0][0], "cy": fc_list[0][1], "dist": d_near})

        elif nf >= 2 and na >= 2:
            # Greedy canonical assignment: sort active speakers by canonical X,
            # each picks their nearest remaining face.
            remaining = list(fc_list)
            for spk in sorted(
                act_list, key=lambda s: _get_canonical_x(s)
            ):
                if not remaining or spk not in raw_data:
                    continue
                best = min(
                    remaining,
                    key=lambda fc: abs(
                        fc[0] - _get_canonical_x(spk)
                    ),
                )
                remaining.remove(best)
                others = [fc for fc in fc_list if fc != best]
                d_near = min([abs(fc[0] - best[0]) for fc in others]) if others else width
                raw_data[spk].append({"time": fd["time"], "cx": best[0], "cy": best[1], "dist": d_near})

        elif nf >= 2 and na == 1:
            spk = act_list[0]
            if spk not in raw_data:
                continue
            # Step 1: eliminate faces claimed by OTHER speakers with known canonical positions
            remaining = list(fc_list)
            for other in all_speakers_in_clip:
                if other == spk or other not in speaker_canonical_cx or not remaining:
                    continue
                claimed = min(
                    remaining, key=lambda fc: abs(fc[0] - speaker_canonical_cx[other])
                )
                remaining.remove(claimed)
            # Step 2: from remaining (or full list if exhausted), pick best for this speaker
            pool = remaining if remaining else fc_list
            if raw_data[spk]:
                last_cx = raw_data[spk][-1]["cx"]
                best = min(pool, key=lambda fc: abs(fc[0] - last_cx))
            elif spk in speaker_canonical_cx:
                best = min(pool, key=lambda fc: abs(fc[0] - speaker_canonical_cx[spk]))
            else:
                best = pool[len(pool) // 2]
            
            others = [fc for fc in fc_list if fc != best]
            d_near = min([abs(fc[0] - best[0]) for fc in others]) if others else width
            raw_data[spk].append({"time": fd["time"], "cx": best[0], "cy": best[1], "dist": d_near})

    # ---- FASE 2: SMOOTH CAMERA PER SPEAKER (Centering CX) ----
    def _smooth_positions(raw_data):
        smooth = []
        if not raw_data:
            return smooth
        cam_cx = raw_data[0]["cx"]
        cam_cy = raw_data[0]["cy"]
        
        # Reference crop width before any zoom is applied
        p_ratio = panel_w / panel_h
        if (width / height) > p_ratio:
            ref_crop_w = int(height * p_ratio)
        else:
            ref_crop_w = width

        # Auto-zoom logic setup
        def _calc_target_zoom(dist_val):
            if not cfg.split_auto_zoom:
                return 1.0
            # More aggressive separation: assume face takes some width
            # We want half-width of crop to be less than distance to neighboring face
            buffer = ref_crop_w * 0.08
            effective_dist = max(50, dist_val - buffer)
            t_zoom = (ref_crop_w / (2 * effective_dist)) * 1.4
            return max(1.0, min(t_zoom, getattr(cfg, "split_max_zoom", 2.5)))

        cam_zoom = _calc_target_zoom(raw_data[0].get("dist", width))
        
        deadzone_px = ref_crop_w * DEADZONE_RATIO
        snap_px = width * SNAP_THRESHOLD

        for d in raw_data:
            face_cx = d["cx"]
            face_cy = d["cy"]
            if abs(face_cx - cam_cx) > snap_px:
                cam_cx = face_cx
            else:
                if face_cx > cam_cx + deadzone_px:
                    cam_cx += (face_cx - (cam_cx + deadzone_px)) * SMOOTH_FACTOR
                elif face_cx < cam_cx - deadzone_px:
                    cam_cx += (face_cx - (cam_cx - deadzone_px)) * SMOOTH_FACTOR

            # Vertical smoothing
            cam_cy += (face_cy - cam_cy) * SMOOTH_FACTOR
            
            # Auto-zoom logic
            target_zoom = _calc_target_zoom(d.get("dist", width))
            cam_zoom += (target_zoom - cam_zoom) * SMOOTH_FACTOR
            
            smooth.append({"time": d["time"], "cx": cam_cx, "cy": cam_cy, "zoom": cam_zoom})
        return smooth

    smooth: dict[str, list] = {
        spk: _smooth_positions(raw_data[spk]) for spk in all_speakers_in_clip
    }

    def _get_pos_full(speaker: str, t: float) -> tuple[float, float, float]:
        sd = smooth.get(speaker, [])
        if not sd:
            return width / 2, height / 2, 1.0
        if t <= sd[0]["time"]:
            return sd[0]["cx"], sd[0]["cy"], sd[0]["zoom"]
        if t >= sd[-1]["time"]:
            return sd[-1]["cx"], sd[-1]["cy"], sd[-1]["zoom"]
        for i in range(len(sd) - 1):
            if sd[i]["time"] <= t <= sd[i + 1]["time"]:
                t1, t2 = sd[i]["time"], sd[i + 1]["time"]
                cx1, cx2 = sd[i]["cx"], sd[i + 1]["cx"]
                cy1, cy2 = sd[i]["cy"], sd[i + 1]["cy"]
                z1, z2 = sd[i]["zoom"], sd[i + 1]["zoom"]
                if t1 == t2:
                    return cx1, cy1, z1
                frac = (t - t1) / (t2 - t1)
                return (
                    cx1 + (cx2 - cx1) * frac,
                    cy1 + (cy2 - cy1) * frac,
                    z1 + (z2 - z1) * frac
                )
        return width / 2, height / 2, 1.0

    def _get_all_boxes(t):
        if not all_frame_data:
            return []
        if t <= all_frame_data[0]["time"]:
            return all_frame_data[0]["face_boxes"]
        if t >= all_frame_data[-1]["time"]:
            return all_frame_data[-1]["face_boxes"]

        for i in range(len(all_frame_data) - 1):
            if all_frame_data[i]["time"] <= t <= all_frame_data[i + 1]["time"]:
                b1s = all_frame_data[i]["face_boxes"]
                b2s = all_frame_data[i + 1]["face_boxes"]
                
                # Simple approach: if counts match, interpolate. Else just return nearest.
                if len(b1s) != len(b2s):
                    return b1s if abs(t - all_frame_data[i]["time"]) < abs(t - all_frame_data[i+1]["time"]) else b2s
                
                t1, t2 = all_frame_data[i]["time"], all_frame_data[i + 1]["time"]
                frac = (t - t1) / (t2 - t1)
                
                res = []
                for b1, b2 in zip(b1s, b2s):
                    res.append((
                        b1[0] + (b2[0] - b1[0]) * frac,
                        b1[1] + (b2[1] - b1[1]) * frac,
                        b1[2] + (b2[2] - b1[2]) * frac,
                        b1[3] + (b2[3] - b1[3]) * frac,
                    ))
                return res
        return []

    # ---- FASE 3: RENDER FRAMES ----
    # Determine outputs needed
    writer_main = None
    writer_dev = None
    
    if dual_output:
        # Two separate files need to be written simultaneously
        vid_main = output_video
        vid_dev = output_video.replace(".ts", "_dev.ts").replace(".mp4", "_dev.mp4")
        writer_main = open_ffmpeg_video_writer(vid_main, 1080, 1920, orig_fps, video_encoder)
        writer_dev = open_ffmpeg_video_writer(vid_dev, 1920, 1080, orig_fps, video_encoder)
    else:
        # Only one stream (either standard, pure dev, or merged)
        writer_main = open_ffmpeg_video_writer(output_video, out_w_final, out_h_final, orig_fps, video_encoder)

    # Pre-create overlay for inactive speaker
    dark_overlay = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    # Per-speaker frozen frame cache: stores last valid crop for each speaker
    # so any panel can fallback when its speaker is not in the current scene
    last_valid_crop: dict[str, np.ndarray] = {}

    current_layout = "split"
    current_speaker = None
    MIN_HOLD = float(getattr(cfg, "switch_hold_duration", 2.0))
    # Initialize with a large negative time to allow instant layout decision at t=0
    last_switch_time = -MIN_HOLD
    
    # Stability window for layout decisions (Majority Vote of face counts)
    LAYOUT_SMOOTH_WINDOW = getattr(cfg, "track_smooth_window", 12)
    face_count_history = []
    # (MIN_HOLD is already initialized above)
    is_dynamic = getattr(cfg, "use_dynamic_split", False)
    
    # Scene cut detection state
    prev_small_gray = None
    SCENE_CUT_THRESHOLD = getattr(cfg, "scene_cut_threshold", 18) # Lower = more sensitive to camera cuts

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_clip * 1000)
        frame_count = 0
        last_render_percent = -1

        print(f"🎬 {label} - Render split-screen {'(dynamic)' if is_dynamic else ''} dimulai...", flush=True)
        tracking_log = [] # Store (t, cx) for subtitle tracking

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t = frame_count / orig_fps
            if t > duration:
                break

            if cfg.box_face_detection:
                boxes = _get_all_boxes(t)
                for b in boxes:
                    cv2.rectangle(
                        frame,
                        (int(b[0]), int(b[1])),
                        (int(b[2]), int(b[3])),
                        (0, 255, 255),
                        3,
                    )

            # --- Scene Cut Detection ---
            # Lightweight check: if pixels change drastically, clear stability history to allow instant switch
            curr_small = _resize_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
            if prev_small_gray is not None:
                diff = cv2.absdiff(curr_small, prev_small_gray)
                avg_diff = np.mean(diff)
                if avg_diff > SCENE_CUT_THRESHOLD:
                    face_count_history.clear()
                    # Also temporarily lower MIN_HOLD requirement for this frame to cut instantly
                    last_switch_time = t - MIN_HOLD 
            prev_small_gray = curr_small

            timestamp_abs = start_clip + t
            from clipping.diarization import get_active_speakers
            active_speakers = get_active_speakers(diarization_data, timestamp_abs)
            active_speaker = get_active_speaker(diarization_data, timestamp_abs)

            # --- Layout decision logic ---
            if is_dynamic:
                if cfg.split_trigger == "face":
                    now_boxes = _get_all_boxes(t)
                    now_count = len(now_boxes)
                    face_count_history.append(now_count)
                    if len(face_count_history) > LAYOUT_SMOOTH_WINDOW:
                        face_count_history.pop(0)
                    
                    # Majority vote face count
                    if face_count_history:
                        stable_count = max(set(face_count_history), key=face_count_history.count)
                    else:
                        stable_count = now_count

                    # FAST PATH: split→full is instant (no ghost frames)
                    # When currently in split and this frame sees only 1 face,
                    # force immediate switch without majority vote or MIN_HOLD.
                    # Rationale: a split layout showing 1 person looks broken;
                    # the transition must be imperceptible.
                    if current_layout == "split" and now_count == 1:
                        current_layout = "full"
                        current_speaker = ranked[0]
                        last_switch_time = t
                        face_count_history.clear()  # Reset for clean state
                    else:
                        # Normal path: use stable_count (guarded by MIN_HOLD)
                        if stable_count == 1:
                            target_layout = "full"
                            target_speaker = ranked[0]
                        elif stable_count >= 2:
                            target_layout = "split"
                            target_speaker = ranked[0]
                        else:
                            target_layout = current_layout
                            target_speaker = current_speaker

                        if target_layout != current_layout and (t - last_switch_time) >= MIN_HOLD:
                            current_layout = target_layout
                            current_speaker = target_speaker
                            last_switch_time = t
                else:
                    if len(active_speakers) == 1:
                        target_layout = "full"
                        target_speaker = active_speakers[0]
                    elif len(active_speakers) >= 2:
                        target_layout = "split"
                        target_speaker = active_speakers[0] # Not used in split but for tracking
                    else:
                        target_layout = current_layout # Stay put
                        target_speaker = current_speaker
                
                    if target_layout != current_layout and (t - last_switch_time) >= MIN_HOLD:
                        current_layout = target_layout
                        current_speaker = target_speaker
                        last_switch_time = t
                
                if current_layout == "full":
                    # Switch speaker if audio trigger is active, or stay on tracked
                    if not cfg.split_trigger == "face":
                        if len(active_speakers) == 1 and active_speakers[0] != current_speaker:
                            if (t - last_switch_time) >= MIN_HOLD:
                                current_speaker = active_speakers[0]
                                last_switch_time = t
            else:
                current_layout = "split"

            if current_layout == "full":
                # Solo mode: Render full 9:16 crop
                spk = current_speaker or (speaker_top if speaker_top in ranked else ranked[0])
                smooth_cx, smooth_cy, _ = _get_pos_full(spk, t)
                # Calculate Top-Left X for full 9:16 crop
                x_full = int(max(0, min(smooth_cx - crop_w_full / 2, width - crop_w_full)))
                y_full = int(max(0, min(smooth_cy - crop_h_full / 2, height - crop_h_full)))
                
                crop = frame[y_full:y_full+crop_h_full, x_full : x_full + crop_w_full]
                final_frame = _resize_frame(crop, (out_w, out_h))
                # Log cx for subtitles to follow
                tracking_log.append((t, smooth_cx))
            else:
                # Split mode: existing logic
                if active_speaker and active_speaker not in (speaker_top, speaker_bottom):
                    panel_top_spk = active_speaker
                    panel_bottom_spk = speaker_bottom
                else:
                    panel_top_spk = speaker_top
                    panel_bottom_spk = speaker_bottom

                # Detect whether the current scene is a solo shot
                in_solo_scene = bool(active_speaker and speaker_is_solo.get(active_speaker, False))

                # ---- Helper: build a panel crop for a given speaker ----
                def _build_panel(spk, is_other_panel_spk=False):
                    spk_not_visible = (
                        active_speaker in all_speakers_in_clip
                        and active_speaker != spk
                        and (speaker_is_solo.get(spk, False) or (in_solo_scene and not speaker_is_solo.get(spk, False)))
                    )
                    if spk_not_visible:
                        if spk in last_valid_crop:
                            return last_valid_crop[spk].copy(), True
                        else:
                            return _resize_frame(frame[0:crop_h, default_x : default_x + crop_w], (panel_w, panel_h)), True
                    else:
                        smooth_cx, smooth_cy, s_zoom = _get_pos_full(spk, t)
                        # Combine auto-zoom with manual split-zoom
                        total_zoom = s_zoom * zoom
                        
                        # Apply dynamic zoom to the baseline crop dimensions
                        if (width / height) > (panel_w / panel_h):
                            b_crop_h, b_crop_w = height, int(height * (panel_w / panel_h))
                        else:
                            b_crop_w, b_crop_h = width, int(width / (panel_w / panel_h))
                            
                        eff_crop_w = int(b_crop_w / total_zoom)
                        eff_crop_h = int(b_crop_h / total_zoom)

                        # Calculate Top-Left X for wide panel crop
                        x_panel = int(max(0, min(smooth_cx - eff_crop_w / 2, width - eff_crop_w)))
                        # Vertical alignment with configurable bias
                        v_align = getattr(cfg, "split_v_align", 0.5)
                        y_panel = int(max(0, min(smooth_cy - (eff_crop_h * v_align), height - eff_crop_h)))
                        
                        crop = _resize_frame(frame[y_panel : y_panel + eff_crop_h, x_panel : x_panel + eff_crop_w], (panel_w, panel_h))
                        if not in_solo_scene or active_speaker == spk:
                            last_valid_crop[spk] = crop.copy()
                        return crop, False

                # ---- Top panel ----
                panel_top, is_fallback_top = _build_panel(panel_top_spk)

                # ---- Bottom panel ----
                panel_bottom, is_fallback_bottom = _build_panel(panel_bottom_spk)

                # ---- Active / inactive highlighting ----
                if active_speaker == panel_top_spk:
                    panel_bottom = cv2.addWeighted(panel_bottom, 1.0 - INACTIVE_ALPHA, dark_overlay, INACTIVE_ALPHA, 0)
                    cv2.rectangle(panel_top, (0, 0), (panel_w - 1, panel_h - 1), (0, 255, 255), ACTIVE_BORDER)
                elif active_speaker == panel_bottom_spk:
                    panel_top = cv2.addWeighted(panel_top, 1.0 - INACTIVE_ALPHA, dark_overlay, INACTIVE_ALPHA, 0)
                    cv2.rectangle(panel_bottom, (0, 0), (panel_w - 1, panel_h - 1), (0, 255, 255), ACTIVE_BORDER)
                else:
                    if is_fallback_top:
                        panel_top = cv2.addWeighted(panel_top, 1.0 - INACTIVE_ALPHA, dark_overlay, INACTIVE_ALPHA, 0)
                    if is_fallback_bottom:
                        panel_bottom = cv2.addWeighted(panel_bottom, 1.0 - INACTIVE_ALPHA, dark_overlay, INACTIVE_ALPHA, 0)

                # Compose the final frame
                divider = np.full((DIVIDER_HEIGHT, panel_w, 3), 80, dtype=np.uint8)
                final_frame = np.vstack([panel_top, divider, panel_bottom])
                # Log neutral center for split mode subtitles
                tracking_log.append((t, width / 2))


            # Ensure exact output dimensions
            if not dev_visualize:
                if final_frame.shape[0] != out_h_final or final_frame.shape[1] != out_w_final:
                    final_frame = _resize_frame(final_frame, (out_w_final, out_h_final))
            else:
                # --- DIRECTOR'S CONSOLE (DEV MODE) ---
                # UI Constants
                HUD_COLOR = (0, 255, 0)
                HUD_X, HUD_Y = 30, 50
                
                # Base frame: 1920x1080 landscape
                frame_res = _resize_frame(frame, (1920, 1080))
                frame_dev = (frame_res * 0.35).astype(np.uint8) # Dim background
                
                scale_x = 1920 / width
                scale_y = 1080 / height
                
                # PREcalculate coordinates for BOTH layouts
                # 1. solo (9:16)
                spk_solo = current_speaker or (speaker_top if speaker_top in ranked else ranked[0])
                cx_solo, cy_solo, _ = _get_pos_full(spk_solo, t)
                cx_s_scaled = int(cx_solo * scale_x)
                cy_s_scaled = int(cy_solo * scale_y)
                cw_s_scaled = int(crop_w_full * scale_x)
                ch_s_scaled = int(crop_h_full * scale_y)
                
                x1s = max(0, cx_s_scaled - cw_s_scaled // 2)
                x2s = min(1919, x1s + cw_s_scaled)
                y1s = int(max(0, min(cy_s_scaled - ch_s_scaled // 2, 1080 - ch_s_scaled)))
                y2s = y1s + ch_s_scaled
                
                # 2. split boxes (horizontal)
                cx_split = width / 2
                cx_p_scaled = int(cx_split * scale_x)
                cw_p_scaled = int(crop_w * scale_x)
                x1p = max(0, cx_p_scaled - cw_p_scaled // 2)
                x2p = min(1919, x1p + cw_p_scaled)
                mid_h = (1080 - DIVIDER_HEIGHT) // 2
                
                # --- APPLY CLEAR WINDOW (Active) ---
                if current_layout == "full":
                    # Clear solo window
                    frame_dev[y1s:y2s, x1s:x2s] = frame_res[y1s:y2s, x1s:x2s]
                else:
                    # Clear split windows
                    frame_dev[0:mid_h, x1p:x2p] = frame_res[0:mid_h, x1p:x2p]
                    frame_dev[mid_h + DIVIDER_HEIGHT:1080, x1p:x2p] = frame_res[mid_h + DIVIDER_HEIGHT:1080, x1p:x2p]

                # --- DRAW BOXES (Both) ---
                # Inactive boxes are drawn with lower opacity or thinner lines
                color_solo = (255, 255, 255) if current_layout == "full" else (100, 100, 100)
                color_split = (255, 255, 255) if current_layout == "split" else (100, 100, 100)
                thick_solo = 3 if current_layout == "full" else 1
                thick_split = 2 if current_layout == "split" else 1

                cv2.rectangle(frame_dev, (x1s, y1s), (x2s, y2s), color_solo, thick_solo)
                cv2.rectangle(frame_dev, (x1p, 0), (x2p, mid_h), color_split, thick_split)
                cv2.rectangle(frame_dev, (x1p, mid_h + DIVIDER_HEIGHT), (x2p, 1079), color_split, thick_split)

                # Labels
                if current_layout == "full":
                    cv2.putText(frame_dev, f"ACTIVE SOLO: {spk_solo}", (x1s + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    cv2.putText(frame_dev, "ACTIVE SPLIT", (x1p + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Draw all detected faces (on top of clearing)
                all_boxes = _get_all_boxes(t)
                for b in all_boxes:
                    bx1, by1, bx2, by2 = int(b[0]*scale_x), int(b[1]*scale_y), int(b[2]*scale_x), int(b[3]*scale_y)
                    cv2.rectangle(frame_dev, (bx1, by1), (bx2, by2), (0, 255, 255), 2)

                # HUD Info
                now_count = face_count_history[-1] if face_count_history else 0
                diff_val = avg_diff if 'avg_diff' in locals() else 0
                
                hud_lines = [
                    f"MODE: DYNAMIC SPLIT (DEV)",
                    f"TIME: {format_seconds(t)}",
                    f"LAYOUT: {current_layout.upper()}",
                    f"FACES (NOW): {now_count} | STABLE: {stable_count}",
                    f"SCENE DIFF: {diff_val:.1f} (Thr: {SCENE_CUT_THRESHOLD})",
                ]
                
                # Scene cut alert
                if diff_val > SCENE_CUT_THRESHOLD:
                    hud_lines[-1] += " >> RESET! <<"
                
                # Hold status
                hold_rem = max(0, MIN_HOLD - (t - last_switch_time))
                if hold_rem > 0:
                    hud_lines.append(f"SWITCH HOLD: {hold_rem:.1f}s")
                else:
                    hud_lines.append(f"SWITCH HOLD: READY")
                
                for i, line in enumerate(hud_lines):
                    cv2.putText(frame_dev, line, (HUD_X, HUD_Y + i*35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, HUD_COLOR, 2)

            # --- DUAL ROUTING LOGIC ---
            if dual_output:
                # Resize normal output as base logic does not guarantee 1080x1920 if resolution is off
                if final_frame.shape[0] != 1920 or final_frame.shape[1] != 1080:
                    frm_normal = _resize_frame(final_frame, (1080, 1920))
                else:
                    frm_normal = final_frame
                writer_main.stdin.write(frm_normal.tobytes())
                writer_dev.stdin.write(frame_dev.tobytes())
                
            elif merge_output:
                # Resize normal portrait output to fit the 1080 height evenly
                frm_normal_small = _resize_frame(final_frame, (608, 1080))
                
                # Create dark grey large canvas backdrop
                frm_merged = np.full((1220, 2648, 3), 30, dtype=np.uint8)
                
                # Title texts (Legends)
                cv2.putText(frm_merged, "DIRECTOR'S CONSOLE (16:9 RAW)", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(frm_merged, "FINAL OUTPUT (9:16 CROP)", (2000, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                # Bounding Boxes (Drawn slightly outwards to act as neat white borders)
                cv2.rectangle(frm_merged, (38, 98), (40+1920+2, 100+1080+2), (255, 255, 255), 4)
                cv2.rectangle(frm_merged, (1998, 98), (2000+608+2, 100+1080+2), (255, 255, 255), 4)
                
                # Paste the literal video frames onto the exact pixel coordinates
                frm_merged[100:1180, 40:1960] = frame_dev
                frm_merged[100:1180, 2000:2608] = frm_normal_small
                
                writer_main.stdin.write(frm_merged.tobytes())
                
            else:
                # Single Stream Handling
                output_frm = frame_dev if dev_visualize else final_frame
                if output_frm.shape[0] != out_h_final or output_frm.shape[1] != out_w_final:
                    output_frm = _resize_frame(output_frm, (out_w_final, out_h_final))
                writer_main.stdin.write(output_frm.tobytes())

            frame_count += 1

            render_percent = (
                min(100, int((t / duration) * 100)) if duration > 0 else 100
            )
            if render_percent != last_render_percent:
                print(
                    f"⏳ {label} - Render split-screen: {render_percent:3d}% | "
                    f"{format_seconds(t)} / {format_seconds(duration)}",
                    flush=True,
                )
                last_render_percent = render_percent

        if writer_main:
            writer_main.stdin.close()
            stderr_data = writer_main.stderr.read().decode("utf-8", errors="ignore")
            return_code = writer_main.wait()
            if return_code != 0:
                raise RuntimeError(f"FFmpeg writer main gagal: {stderr_data[-1000:]}")
        
        if writer_dev:
            writer_dev.stdin.close()
            stderr_data_dev = writer_dev.stderr.read().decode("utf-8", errors="ignore")
            return_code_dev = writer_dev.wait()
            if return_code_dev != 0:
                raise RuntimeError(f"FFmpeg writer dev gagal: {stderr_data_dev[-1000:]}")

        print(f"✅ {label} selesai.", flush=True)

    finally:
        cap.release()

    def get_x_final(t):
        if not tracking_log: return default_x_full
        # Interpolate for smoother subtitle tracking
        for i in range(len(tracking_log)-1):
            if tracking_log[i][0] <= t <= tracking_log[i+1][0]:
                t1, t2 = tracking_log[i][0], tracking_log[i+1][0]
                cx1, cx2 = tracking_log[i][1], tracking_log[i+1][1]
                cx = cx1 + (cx2 - cx1) * (t - t1) / (t2 - t1)
                return int(cx)
        
        return int(tracking_log[-1][1])

    return get_x_final



def buat_video_camera_switch(
    input_video,
    output_video,
    start_clip,
    end_clip,
    diarization_data,
    cfg,
    label="CameraSwitch",
):
    """
    Render a full 9:16 video with automatic camera switching based on active speaker.

    Behaviour per frame:
      - One speaker active   → full 9:16 crop centred on that speaker's face
      - Both speakers active → blurred pillarbox (16:9 wide-shot + blurred bg)
      - No speaker active    → stay on last known speaker (or pillarbox if none yet)

    A minimum hold duration (cfg.switch_hold_duration, default 2 s) prevents
    rapid flickering when speakers alternate quickly.

    Parameters
    ----------
    input_video : str
        Path to the source video.
    output_video : str
        Output video path (silent, no audio).
    start_clip, end_clip : float
        Clip time range in seconds.
    diarization_data : list[dict]
        Speaker segments from diarization.run_diarization().
    cfg : SimpleNamespace
        Config object (uses cfg.switch_hold_duration, cfg.face_detector, etc.)
    label : str
        Label for progress logging.
    """
    from clipping.diarization import get_active_speakers

    STEP_DETEKSI     = cfg.track_step if cfg.track_step is not None else 0.25
    DEADZONE_RATIO   = cfg.track_deadzone if cfg.track_deadzone is not None else 0.15
    SMOOTH_FACTOR    = cfg.track_smooth if cfg.track_smooth is not None else 0.30
    JITTER_THRESHOLD = cfg.track_jitter if cfg.track_jitter is not None else 5
    SNAP_THRESHOLD   = cfg.track_snap if cfg.track_snap is not None else 0.25
    MIN_HOLD = float(getattr(cfg, "switch_hold_duration", 2.0))
    BLUR_KERNEL = 99
    BLUR_SIGMA = 30

    video_encoder = detect_video_encoder(cfg)

    # ---------------------------------------------------------------- face detector
    yolo_model = None
    detector = None
    if cfg.face_detector == "yolo":
        if not os.path.exists(cfg.file_yolo_model):
            print(f"   📥 Mendownload YOLOv8 Face Model ({cfg.yolo_size})...")
            import urllib.request

            urllib.request.urlretrieve(cfg.url_yolo_model, cfg.file_yolo_model)
        from ultralytics import YOLO

        yolo_model = YOLO(cfg.file_yolo_model)
    else:
        detector = get_face_detector(cfg)

    cap = cv2.VideoCapture(input_video)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if math.isnan(orig_fps) or orig_fps == 0:
        orig_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = end_clip - start_clip

    out_w, out_h = 1080, 1920

    # 9:16 crop region from the source frame
    crop_ratio = out_w / out_h  # ≈ 0.5625
    if (width / height) > crop_ratio:  # source is wider  → crop width
        crop_h = height
        crop_w = int(height * crop_ratio)
    else:  # source is taller → crop height
        crop_w = width
        crop_h = int(width / crop_ratio)

    default_x = (width - crop_w) // 2

    # All unique speakers in this clip
    speakers = sorted(set(s["speaker"] for s in diarization_data))
    if not speakers:
        speakers = ["SPEAKER_00"]

    # ================================================================
    # FASE 1 — Per-speaker face profiling (diarization-guided)
    # ================================================================
    # Strategy:
    #   1 active + 1 face  → trivial: face belongs to active speaker
    #   N active + N faces → sort faces by X; sort speakers by label; assign in order
    #   1 active + N faces → use speaker's last-known position to pick nearest face
    #   otherwise          → skip (ambiguous / no data)
    print(f"🧠 {label} - Analisa wajah (camera switch) dimulai...", flush=True)

    all_frame_data: list[dict] = []  # [{time, face_centers, active_now}]
    speaker_solo_cxs: dict[str, list] = {}  # speaker → [cx, ...] from 1:1 frames
    solo_counts: dict[str, int] = {spk: 0 for spk in speakers}
    multi_counts: dict[str, int] = {spk: 0 for spk in speakers}
    current_time = 0.0
    last_detect_percent = -1

    def _clamp_x_cs(cx_center: float) -> int:
        return max(0, min(int(cx_center - crop_w / 2), width - crop_w))

    while current_time <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, (start_clip + current_time) * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        face_centers = []
        face_boxes = []

        if cfg.face_detector == "yolo":
            yolo_results = yolo_model(frame, verbose=False)
            if yolo_results and len(yolo_results[0].boxes) > 0:
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box
                    face_centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
                    face_boxes.append((x1, y1, x2, y2))
        else:
            results = detector.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
            )
            if results.detections:
                for d in results.detections:
                    bb = d.bounding_box
                    face_centers.append(
                        (
                            bb.origin_x + bb.width / 2,
                            bb.origin_y + bb.height / 2,
                        )
                    )
                    face_boxes.append(
                        (
                            bb.origin_x,
                            bb.origin_y,
                            bb.origin_x + bb.width,
                            bb.origin_y + bb.height,
                        )
                    )

        face_centers.sort(key=lambda fc: fc[0])  # left → right
        active_now = get_active_speakers(diarization_data, start_clip + current_time)
        n_faces = len(face_centers)
        n_active = len(active_now)

        # Pass A: collect face data and build canonical solo observations.
        # Full assignment happens in Pass B after canonical_cx is built from entire clip.
        if n_faces == 1 and n_active == 1:
            spk = active_now[0]
            speaker_solo_cxs.setdefault(spk, []).append((face_centers[0][0], face_centers[0][1]))
            solo_counts[spk] = solo_counts.get(spk, 0) + 1
        elif n_faces >= 2 and n_active >= 1:
            for spk in active_now:
                multi_counts[spk] = multi_counts.get(spk, 0) + 1

        all_frame_data.append(
            {
                "time": current_time,
                "face_centers": face_centers,
                "face_boxes": face_boxes,
                "active_now": active_now,
            }
        )

        detect_pct = (
            min(100, int(current_time / duration * 100)) if duration > 0 else 100
        )
        if detect_pct != last_detect_percent:
            print(f"⏳ {label} - Analisa wajah: {detect_pct:3d}%", flush=True)
            last_detect_percent = detect_pct

        current_time += STEP_DETEKSI

    # Build canonical center-X per speaker from 1:1 frames
    import statistics as _stats

    speaker_canonical_cx: dict[str, float] = {
        spk: _stats.median([c[0] for c in cxs]) for spk, cxs in speaker_solo_cxs.items() if cxs
    }

    if not diarization_data:
        # Visual-only: assume L is left and R is right
        if "FACE_L" not in speaker_canonical_cx:
            speaker_canonical_cx["FACE_L"] = width * 0.25
        if "FACE_R" not in speaker_canonical_cx:
            speaker_canonical_cx["FACE_R"] = width * 0.75

    # Determine which speakers appear mostly in solo-scene frames
    speaker_is_solo: dict[str, bool] = {
        spk: (solo_counts.get(spk, 0) > multi_counts.get(spk, 0))
        for spk in speakers
    }

    # ================================================================
    # Pass B — Canonical-guided face-to-speaker assignment
    # ================================================================
    # Re-process all stored frames using canonical_cx built from entire clip.
    # Multi-scene videos: Speaker C's solo canonical eliminates their face from
    # Scene A frames, leaving the correct face for Scene A speakers.
    # ================================================================
    raw_data: dict[str, list] = {spk: [] for spk in speakers}

    for fd in all_frame_data:
        fc_list = fd["face_centers"]  # list of (cx, cy), sorted left → right
        act_list = fd["active_now"]
        nf = len(fc_list)
        na = len(act_list)

        if nf == 0 or na == 0:
            continue

        if nf == 1 and na == 1:
            spk = act_list[0]
            if spk in raw_data:
                others = [fc for fc in fc_list if fc != fc_list[0]]
                d_near = min([abs(fc[0] - fc_list[0][0]) for fc in others]) if others else width
                raw_data[spk].append({"time": fd["time"], "cx": fc_list[0][0], "cy": fc_list[0][1], "dist": d_near})

        elif nf >= 2 and na >= 2:
            remaining = list(fc_list)
            for spk in sorted(
                act_list, key=lambda s: speaker_canonical_cx.get(s, width / 2)
            ):
                if not remaining or spk not in raw_data:
                    break
                best = min(
                    remaining,
                    key=lambda fc: abs(
                        fc[0] - speaker_canonical_cx.get(spk, width / 2)
                    ),
                )
                remaining.remove(best)
                others = [fc for fc in fc_list if fc != best]
                d_near = min([abs(fc[0] - best[0]) for fc in others]) if others else width
                raw_data[spk].append({"time": fd["time"], "cx": best[0], "cy": best[1], "dist": d_near})

        elif nf >= 2 and na == 1:
            spk = act_list[0]
            if spk not in raw_data:
                continue
            remaining = list(fc_list)
            for other in speakers:
                if other == spk or other not in speaker_canonical_cx or not remaining:
                    continue
                claimed = min(
                    remaining, key=lambda fc: abs(fc[0] - speaker_canonical_cx[other])
                )
                remaining.remove(claimed)
            pool = remaining if remaining else fc_list
            if raw_data[spk]:
                last_cx = raw_data[spk][-1]["cx"]
                best = min(pool, key=lambda fc: abs(fc[0] - last_cx))
            elif spk in speaker_canonical_cx:
                best = min(pool, key=lambda fc: abs(fc[0] - speaker_canonical_cx[spk]))
            else:
                best = pool[len(pool) // 2]
            
            others = [fc for fc in fc_list if fc != best]
            d_near = min([abs(fc[0] - best[0]) for fc in others]) if others else width
            raw_data[spk].append({"time": fd["time"], "cx": best[0], "cy": best[1], "dist": d_near})

    # ================================================================
    # FASE 2 — Smooth per-speaker camera positions
    # ================================================================
    def _smooth_positions_cs(raw_list):
        smooth_list = []
        if not raw_list:
            return smooth_list
        cam_cx = raw_list[0]["cx"]
        cam_cy = raw_list[0]["cy"]
        
        # Reference crop width before any zoom is applied
        p_ratio = crop_w / crop_h
        if (width / height) > p_ratio:
            ref_crop_w = int(height * p_ratio)
        else:
            ref_crop_w = width

        # Auto-zoom logic setup
        def _calc_target_zoom(dist_val):
            if not cfg.split_auto_zoom:
                return 1.0
            # More aggressive separation: assume face takes some width
            buffer = ref_crop_w * 0.08
            effective_dist = max(50, dist_val - buffer)
            t_zoom = (ref_crop_w / (2 * effective_dist)) * 1.4
            return max(1.0, min(t_zoom, getattr(cfg, "split_max_zoom", 2.5)))

        cam_zoom = _calc_target_zoom(raw_list[0].get("dist", width))
        
        deadzone_px = crop_w * DEADZONE_RATIO
        snap_px = width * SNAP_THRESHOLD
        
        for d in raw_list:
            face_cx = d["cx"]
            face_cy = d["cy"]
            if abs(face_cx - cam_cx) > snap_px:
                cam_cx = face_cx
            else:
                if face_cx > cam_cx + deadzone_px:
                    cam_cx += (face_cx - (cam_cx + deadzone_px)) * SMOOTH_FACTOR
                elif face_cx < cam_cx - deadzone_px:
                    cam_cx += (face_cx - (cam_cx - deadzone_px)) * SMOOTH_FACTOR
            
            # Vertical smoothing
            cam_cy += (face_cy - cam_cy) * SMOOTH_FACTOR
            
            # Auto-zoom logic
            target_zoom = _calc_target_zoom(d.get("dist", width))
            cam_zoom += (target_zoom - cam_zoom) * SMOOTH_FACTOR
            
            smooth_list.append({"time": d["time"], "cx": cam_cx, "cy": cam_cy, "zoom": cam_zoom})
        return smooth_list

    smooth: dict[str, list] = {
        spk: _smooth_positions_cs(raw_data[spk]) for spk in speakers
    }

    def _get_pos_cs(speaker, t):
        sd = smooth.get(speaker, [])
        if not sd:
            return width / 2, height / 2, 1.0
        if t <= sd[0]["time"]:
            return sd[0]["cx"], sd[0]["cy"], sd[0]["zoom"]
        if t >= sd[-1]["time"]:
            return sd[-1]["cx"], sd[-1]["cy"], sd[-1]["zoom"]
        for i in range(len(sd) - 1):
            if sd[i]["time"] <= t <= sd[i + 1]["time"]:
                t1, t2 = sd[i]["time"], sd[i + 1]["time"]
                cx1, cx2 = sd[i]["cx"], sd[i + 1]["cx"]
                cy1, cy2 = sd[i]["cy"], sd[i + 1]["cy"]
                z1, z2 = sd[i]["zoom"], sd[i + 1]["zoom"]
                if t1 == t2:
                    return cx1, cy1, z1
                frac = (t - t1) / (t2 - t1)
                return (
                    cx1 + (cx2 - cx1) * frac,
                    cy1 + (cy2 - cy1) * frac,
                    z1 + (z2 - z1) * frac
                )
        return width / 2, height / 2, 1.0

    # ----------------------------------------------------------------
    # Helper: blurred pillarbox for wide-shot / simultaneous speech
    # ----------------------------------------------------------------
    def _make_blurred_pillarbox(frame):
        h, w = frame.shape[:2]
        # Background: scale to cover 1080×1920, crop centre, then blur
        scale = max(out_w / w, out_h / h)
        new_w = max(out_w, int(w * scale))
        new_h = max(out_h, int(h * scale))
        bg = _resize_frame(frame, (new_w, new_h))
        y0 = (new_h - out_h) // 2
        x0 = (new_w - out_w) // 2
        bg = bg[y0 : y0 + out_h, x0 : x0 + out_w]
        ksize = BLUR_KERNEL if BLUR_KERNEL % 2 == 1 else BLUR_KERNEL + 1
        bg = cv2.GaussianBlur(bg, (ksize, ksize), BLUR_SIGMA)
        # Foreground: scale frame to width=out_w, preserve aspect ratio
        fg_w = out_w
        fg_h = min(out_h, int(h * out_w / w))
        fg = _resize_frame(frame, (fg_w, fg_h))
        # Composite: centre foreground vertically on blurred background
        result = bg.copy()
        y_start = (out_h - fg_h) // 2
        result[y_start : y_start + fg_h, 0:fg_w] = fg
        return result

    def _get_all_boxes(t):
        if not all_frame_data:
            return []
        if t <= all_frame_data[0]["time"]:
            return all_frame_data[0]["face_boxes"]
        if t >= all_frame_data[-1]["time"]:
            return all_frame_data[-1]["face_boxes"]

        for i in range(len(all_frame_data) - 1):
            if all_frame_data[i]["time"] <= t <= all_frame_data[i + 1]["time"]:
                b1s = all_frame_data[i]["face_boxes"]
                b2s = all_frame_data[i + 1]["face_boxes"]
                if len(b1s) != len(b2s):
                    return b1s if abs(t - all_frame_data[i]["time"]) < abs(t - all_frame_data[i+1]["time"]) else b2s
                t1, t2 = all_frame_data[i]["time"], all_frame_data[i + 1]["time"]
                frac = (t - t1) / (t2 - t1)
                res = []
                for b1, b2 in zip(b1s, b2s):
                    res.append((
                        b1[0] + (b2[0] - b1[0]) * frac,
                        b1[1] + (b2[1] - b1[1]) * frac,
                        b1[2] + (b2[2] - b1[2]) * frac,
                        b1[3] + (b2[3] - b1[3]) * frac,
                    ))
                return res
        return []

    # FASE 3: RENDER FRAME
    out_w, out_h = _get_render_dims(cfg, "9:16", source_h=height)
    
    dev_visualize = cfg.dev_mode # Assume only for 9:16 as described
    if dev_visualize:
        out_w, out_h = (1920, 1080)

    writer = open_ffmpeg_video_writer(
        output_video, out_w, out_h, orig_fps, video_encoder
    )

    current_speaker = None
    last_switch_time = 0.0

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_clip * 1000)
        frame_count = 0
        last_render_percent = -1
        tracking_log = [] # Store (t, cx) for each frame

        print(f"🎬 {label} - Render camera switch dimulai...", flush=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t = frame_count / orig_fps
            if t > duration:
                break

            if cfg.box_face_detection:
                boxes = _get_all_boxes(t)
                for b in boxes:
                    cv2.rectangle(
                        frame,
                        (int(b[0]), int(b[1])),
                        (int(b[2]), int(b[3])),
                        (0, 255, 255),
                        3,
                    )

            timestamp_abs = start_clip + t
            active_speakers = get_active_speakers(diarization_data, timestamp_abs)

            if dev_visualize:
                # Dev visualization for camera-switch
                frame_base = _resize_frame(frame, (out_w, out_h))
                frame_dev = (frame_base * 0.35).astype(np.uint8)
                
                scale_x = out_w / width
                
                # Check what state we are in (Wide or Crop)
                is_wide = False
                if len(active_speakers) >= 2:
                    all_multi_scene = all(not speaker_is_solo.get(spk, False) for spk in active_speakers)
                    if all_multi_scene:
                        is_wide = True
                elif len(active_speakers) == 0 and current_speaker is None:
                    is_wide = True
                
                if is_wide:
                    # Show full frame in dev mode (maybe slightly brightened back or with label)
                    frame_dev = (frame_base * 0.8).astype(np.uint8)
                    cv2.putText(frame_dev, "WIDE SHOT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    # It's a crop on current_speaker
                    if current_speaker is not None:
                        cx, cy, s_zoom = _get_pos_cs(current_speaker, t)
                        cx_scaled = int(cx * scale_x)
                        cw_scaled = int(crop_w * scale_x)
                        
                        # Paste bright crop
                        eff_cw = int(crop_w / s_zoom)
                        lb, rb = int(max(0, min(cx - eff_cw / 2, width - eff_cw))), int(max(0, min(cx + eff_cw / 2, width)))
                        lb_s, rb_s = int(lb * scale_x), int(rb * scale_x)
                        frame_dev[:, lb_s : rb_s] = frame_base[:, lb_s : rb_s]
                        # Vertical lines
                        cv2.line(frame_dev, (cx_scaled, 0), (cx_scaled, out_h), (255, 255, 255), 2)
                        cv2.line(frame_dev, (cx_scaled + cw_scaled, 0), (cx_scaled + cw_scaled, out_h), (255, 255, 255), 2)
                        
                        label_spk = f"TRACKING: {current_speaker}"
                        cv2.putText(frame_dev, label_spk, (cx_scaled + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Force face boxes in dev mode
                f_boxes = _get_all_boxes(t)
                for fb in f_boxes:
                    scale_y = out_h / height
                    fb1, fb2, fb3, fb4 = int(fb[0]*scale_x), int(fb[1]*scale_y), int(fb[2]*scale_x), int(fb[3]*scale_y)
                    cv2.rectangle(frame_dev, (fb1, fb2), (fb3, fb4), (0, 255, 255), 2)
                    
                    if (cfg.track_lines or cfg.dev_mode) and not is_wide:
                        # Current crop boundaries
                        if current_speaker is not None:
                            cx, cy, s_zoom = _get_pos_cs(current_speaker, t)
                            eff_cw = int(crop_w / s_zoom)
                            lb = int(max(0, min(cx - eff_cw / 2, width - eff_cw)))
                            cx_scaled = int(lb * scale_x)
                            cw_scaled = int(eff_cw * scale_x)
                            
                            mid_x = (fb1 + fb3) // 2
                            mid_y = (fb2 + fb4) // 2
                            
                            # Horizontal lines
                            cv2.line(frame_dev, (cx_scaled, mid_y), (fb1, mid_y), (0, 255, 255), 2)
                            cv2.line(frame_dev, (fb3, mid_y), (cx_scaled + cw_scaled, mid_y), (0, 255, 255), 2)
                            
                            # Vertical lines
                            cv2.line(frame_dev, (mid_x, 0), (mid_x, fb2), (0, 255, 255), 2)
                            cv2.line(frame_dev, (mid_x, fb4), (mid_x, out_h), (0, 255, 255), 2)
                
                out_frame = frame_dev

            elif len(active_speakers) >= 2:
                # ... standard logic ...
                all_multi_scene = all(
                    not speaker_is_solo.get(spk, False) for spk in active_speakers
                )
                if all_multi_scene:
                    cx = (width - crop_w) // 2
                    out_frame = _make_blurred_pillarbox(frame)
                else:
                    if current_speaker is None or current_speaker not in active_speakers:
                        current_speaker = active_speakers[0]
                        last_switch_time = t
                    cx, cy, s_zoom = _get_pos_cs(current_speaker, t)
                    eff_cw = int(crop_w / s_zoom)
                    eff_ch = int(crop_h / s_zoom)
                    x_full = int(max(0, min(cx - eff_cw / 2, width - eff_cw)))
                    y_full = int(max(0, min(cy - eff_ch / 2, height - eff_ch)))
                    crop_fr = frame[y_full : y_full + eff_ch, x_full : x_full + eff_cw]
                    out_frame = _resize_frame(crop_fr, (out_w, out_h))

            elif len(active_speakers) == 1:
                new_speaker = active_speakers[0]
                if current_speaker is None:
                    current_speaker = new_speaker
                    last_switch_time = t
                elif (
                    new_speaker != current_speaker
                    and (t - last_switch_time) >= MIN_HOLD
                ):
                    current_speaker = new_speaker
                    last_switch_time = t
                cx, cy, s_zoom = _get_pos_cs(current_speaker, t)
                eff_cw = int(crop_w / s_zoom)
                eff_ch = int(crop_h / s_zoom)
                x_full = int(max(0, min(cx - eff_cw / 2, width - eff_cw)))
                y_full = int(max(0, min(cy - eff_ch / 2, height - eff_ch)))
                crop_fr = frame[y_full : y_full + eff_ch, x_full : x_full + eff_cw]
                out_frame = _resize_frame(crop_fr, (out_w, out_h))

            else:
                if current_speaker is not None:
                    cx, cy, s_zoom = _get_pos_cs(current_speaker, t)
                    eff_cw = int(crop_w / s_zoom)
                    eff_ch = int(crop_h / s_zoom)
                    x_full = int(max(0, min(cx - eff_cw / 2, width - eff_cw)))
                    y_full = int(max(0, min(cy - eff_ch / 2, height - eff_ch)))
                    crop_fr = frame[y_full : y_full + eff_ch, x_full : x_full + eff_cw]
                    out_frame = _resize_frame(crop_fr, (out_w, out_h))
                else:
                    cx = (width - crop_w) // 2 # Center for blurred view
                    out_frame = _make_blurred_pillarbox(frame)

            tracking_log.append((t, cx))
            writer.stdin.write(out_frame.tobytes())
            frame_count += 1

            render_pct = min(100, int(t / duration * 100)) if duration > 0 else 100
            if render_pct != last_render_percent:
                print(
                    f"⏳ {label} - Render camera switch: {render_pct:3d}% | "
                    f"{format_seconds(t)} / {format_seconds(duration)}",
                    flush=True,
                )
                last_render_percent = render_pct

        writer.stdin.close()
        stderr_data = writer.stderr.read().decode("utf-8", errors="ignore")
        return_code = writer.wait()

        if return_code != 0:
            raise RuntimeError(f"FFmpeg writer gagal: {stderr_data[-1000:]}")

        print(f"✅ {label} selesai.", flush=True)

        # Helper for subtitle positioning
        def get_x_final(t):
            if not tracking_log:
                return default_x
            if t <= tracking_log[0][0]:
                return int(tracking_log[0][1])
            
            for i in range(len(tracking_log)-1):
                if tracking_log[i][0] <= t <= tracking_log[i+1][0]:
                    t1, t2 = tracking_log[i][0], tracking_log[i+1][0]
                    cx1, cx2 = tracking_log[i][1], tracking_log[i+1][1]
                    cx = cx1 + (cx2 - cx1) * (t - t1) / (t2 - t1)
                    return int(cx)
            
            return int(tracking_log[-1][1])

        return get_x_final

    finally:
        cap.release()


# ==============================================================================
# PROSES KLIP UTAMA
# ==============================================================================


def proses_klip(
    rank, clip, rasio, glitch_ts, data_segmen, cfg, video_encoder, diarization_data=None
):
    """
    Run full clip processing pipeline from render to final output files.

    Args:
        rank: Clip rank/index.
        clip: Clip metadata object.
        rasio: Target output ratio.
        glitch_ts: Optional prepared glitch transition path.
        data_segmen: Transcript segments.
        cfg: Runtime config object.
        video_encoder: Encoder descriptor dict.
        diarization_data: Optional speaker diarization metadata.

    Returns:
        Manifest dictionary describing processing result and output paths.
    """
    get_x_h = None
    get_x_main = None
    h_start = float(clip.get("hook_start_time", clip["start_time"]))
    h_end = float(
        clip.get(
            "hook_end_time",
            clip.get("hook_start_time", clip["start_time"]) + cfg.durasi_hook,
        )
    )
    
    # Custom Hook Override
    file_hook_src = cfg.file_video_asli
    custom_hook = clip.get("custom_hook_info")
    if custom_hook:
        file_hook_src = custom_hook["file_path"]
        h_start = getattr(cfg, "hook_source_start", 0.0)
        
        try:
            cap_h = cv2.VideoCapture(file_hook_src)
            fps = cap_h.get(cv2.CAP_PROP_FPS)
            frames = cap_h.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0:
                vid_duration = frames / fps
            else:
                vid_duration = float('inf')
            cap_h.release()
        except:
            vid_duration = float('inf')

        h_end = h_start + cfg.durasi_hook
        if h_end > vid_duration:
            h_end = vid_duration
    m_start = float(clip["start_time"])
    m_end = float(clip["end_time"])
    judul = clip.get("title_indonesia")
    judul_en = clip.get("title_inggris")
    
    out_vid = os.path.join(cfg.outputs_dir, f"highlight_rank_{rank}_ready.mp4")
    if getattr(cfg, "dev_mode_with_output_merge", False):
        out_vid = os.path.join(cfg.outputs_dir, f"highlight_rank_{rank}_dev_mode_merge_ready.mp4")
        
    out_thm = os.path.join(cfg.outputs_dir, f"thumbnail_rank_{rank}.jpg")

    # Ambil resolusi video asli untuk perhitungan posisi subtitle di dev-mode
    cap_asli = cv2.VideoCapture(cfg.file_video_asli)
    sw = int(cap_asli.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap_asli.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_asli.release()
    source_dim = (sw, sh)

    manifest_item = {
        "rank": rank,
        "status": "pending",
        "ratio": rasio,
        "video_path": out_vid,
        "thumbnail_path": out_thm,
        "thumbnail_text": judul_en or judul or f"Highlight {rank}",
        "youtube_title_final": clip.get(
            "youtube_title_final", clip.get("title_inggris", "")
        ),
        "youtube_description_final": clip.get("youtube_description_final", ""),
        "youtube_tags_final": clip.get("youtube_tags_final", []),
        "tiktok_caption_final": clip.get(
            "tiktok_caption_final", clip.get("hastag", "")
        ),
        "title_indonesia": clip.get("title_indonesia", ""),
        "title_inggris": clip.get("title_inggris", ""),
        "hastag": clip.get("hastag", ""),
        "start_time": m_start,
        "end_time": m_end,
        "hook_start_time": h_start,
        "hook_end_time": h_end,
        "duration": round(m_end - m_start, 2),
        "alasan": clip.get("alasan", ""),
        "broll_list": clip.get("broll_list", []),
        "typography_plan": clip.get("typography_plan", []),
    }

    print(f"\n{'=' * 70}")
    print(f"🔥 [Rank {rank}] Memproses clip")
    print(f"📝 [Judul Indo]   : '{clip.get('title_indonesia', '-')}'")
    print(f"📝 [Judul Inggris]: '{clip.get('title_inggris', '-')}'")
    print(f"#️⃣ [Hastag]      : '{clip.get('hastag', '-')}'")
    print(f"🧠 Encoder aktif  : {video_encoder['name']}")
    print(f"{'=' * 70}")

    typography_plan = clip.get("typography_plan", [])
    siapkan_font_tipografi(cfg)

    h_ts, m_ts, a_hook, a_main = (
        f"h_{rank}.ts",
        f"m_{rank}.ts",
        f"ah_{rank}.ass",
        f"am_{rank}.ass",
    )
    h_silent, m_silent = f"h_silent_{rank}.mp4", f"m_silent_{rank}.mp4"
    
    dev_dual = getattr(cfg, "dev_mode_with_output", False)
    h_ts_dev = f"h_{rank}_dev.ts"
    m_ts_dev = f"m_{rank}_dev.ts"
    
    aktif_hook = cfg.use_hook_glitch


    # Determine if we should use split-screen mode
    if getattr(cfg, "use_split_screen", False) and rasio == "9:16":
        if cfg.split_trigger == "face":
            use_split = True
        else:
            use_split = (
                diarization_data is not None
                and len(set(s["speaker"] for s in diarization_data)) >= 2
            )
    else:
        use_split = False

    # Camera-switch mode (mutually exclusive: split-screen takes precedence)
    use_camera_switch = (
        not use_split
        and getattr(cfg, "use_camera_switch", False)
        and rasio == "9:16"
        and diarization_data
        and len(set(s["speaker"] for s in diarization_data)) >= 2
    )

    broll_list = clip.get("broll_list", [])
    broll_aktif = []
    if cfg.use_broll and broll_list:
        print(f"   🎥 Mendownload {len(broll_list)} video B-Roll dari Pexels...")
        for i, br in enumerate(broll_list):
            q = br.get("search_query", "nature")
            file_broll = f"temp_broll_{rank}_{i}.mp4"
            if download_pexels_broll(q, rasio, file_broll, cfg.pexels_api_key):
                br_copy = dict(br)
                br_copy["filepath"] = file_broll
                broll_aktif.append(br_copy)

    std_p = get_ts_encode_args(video_encoder, fps=30)

    try:
        # HOOK
        if aktif_hook:
            get_x_h = None
            if use_split:
                print("   📸 [Hook] Split-screen render (Custom Hook diabaikan untuk format ini saat ini atau digabung)...")
                get_x_h = buat_video_split_screen(
                    file_hook_src,
                    h_silent,
                    h_start,
                    h_end,
                    diarization_data if not custom_hook else None,
                    cfg,
                    label=f"Rank {rank} Hook SplitScreen",
                )
            elif use_camera_switch:
                print("   📸 [Hook] Camera switch render...")
                get_x_h = buat_video_camera_switch(
                    file_hook_src,
                    h_silent,
                    h_start,
                    h_end,
                    diarization_data if not custom_hook else None,
                    cfg,
                    label=f"Rank {rank} Hook CameraSwitch",
                )
            else:
                print("   📸 [Hook] Hybrid render...")
                get_x_h = buat_video_hybrid(
                    file_hook_src,
                    h_silent,
                    h_start,
                    h_end,
                    rasio,
                    cfg,
                    label=f"Rank {rank} Hook",
                )
            
            aktif_advanced_hook = cfg.use_advanced_text_on_hook
            if not cfg.no_subs and not custom_hook:
                buat_file_ass(
                    data_segmen,
                    h_start,
                    h_end,
                    a_hook,
                    rasio,
                    cfg,
                    typography_plan=typography_plan,
                    gunakan_advanced=aktif_advanced_hook,
                    get_x_func=get_x_h,
                    source_dim=source_dim,
                )

                print("   🎬 [Hook] FFmpeg burn subtitle + audio...")
                esc_ass_hook = escape_ffmpeg_filter_value(os.path.abspath(a_hook))
                esc_fontsdir = escape_ffmpeg_filter_value(os.path.abspath(cfg.font_dir))
                vf_hook_list = [f"subtitles={esc_ass_hook}:fontsdir={esc_fontsdir}"]
            else:
                print(f"   🎬 [Hook] Skip subtitle rendering {'(Custom Hook)' if custom_hook else ''}...")
                vf_hook_list = []
            
            if cfg.video_sharpen:
                vf_hook_list.append("unsharp=5:5:0.5:5:5:0.0")
            
            vf_hook = ",".join(vf_hook_list) if vf_hook_list else None

            cmd_h_base = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "verbose",
                "-y",
                "-i",
                h_silent,
                "-ss",
                str(h_start),
                "-to",
                str(h_end),
                "-i",
                file_hook_src,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
            ]
            if vf_hook:
                cmd_h_base += ["-vf", vf_hook]
            cmd_h_base += std_p

            cmd_h = build_ffmpeg_progress_cmd(cmd_h_base, h_ts)
            rc_h, err_h = run_ffmpeg_with_progress(
                cmd_h, h_end - h_start, label=f"Rank {rank} Hook FFmpeg"
            )
            if rc_h != 0:
                raise RuntimeError("FFmpeg hook gagal:\n" + "\n".join(err_h))

        # MAIN
        if use_split:
            print("   📸 [Main] Split-screen render (Visual)...")
            get_x_main = buat_video_split_screen(
                cfg.file_video_asli,
                m_silent,
                m_start,
                m_end,
                diarization_data,
                cfg,
                label=f"Rank {rank} Main SplitScreen",
            )
        elif use_camera_switch:
            # Note: Camera Switch doesn't currently support dev_mode frames but we pass it anyway
            print("   📸 [Main] Camera switch render (Visual)...")
            get_x_main = buat_video_camera_switch(
                cfg.file_video_asli,
                m_silent,
                m_start,
                m_end,
                diarization_data,
                cfg,
                label=f"Rank {rank} Main CameraSwitch",
            )
        else:
            print("   📸 [Main] Hybrid render (Visual)...")
            get_x_main = buat_video_hybrid(
                cfg.file_video_asli,
                m_silent,
                m_start,
                m_end,
                rasio,
                cfg,
                broll_aktif,
                label=f"Rank {rank} Main",
            )

        if not cfg.no_subs:
            buat_file_ass(
                data_segmen,
                m_start,
                m_end,
                a_main,
                rasio,
                cfg,
                typography_plan=typography_plan,
                gunakan_advanced=True,
                get_x_func=get_x_main,
                source_dim=source_dim,
            )

        print(f"   🎬 [Main] FFmpeg {'skip subtitle' if cfg.no_subs else 'burn subtitle'} + audio ducking...")
        esc_ass_main = escape_ffmpeg_filter_value(os.path.abspath(a_main)) if not cfg.no_subs else ""
        esc_fontsdir = escape_ffmpeg_filter_value(os.path.abspath(cfg.font_dir))

        # SMART BGM
        aktif_bgm = cfg.use_auto_bgm
        bgm_mood = clip.get("bgm_mood", "chill")
        if bgm_mood not in cfg.bgm_pool:
            bgm_mood = "chill"
        bgm_page = cfg.bgm_pool[bgm_mood]
        file_bgm = os.path.abspath(os.path.join(cfg.base_dir, f"bgm_{bgm_mood}.mp3"))

        if aktif_bgm and not os.path.exists(file_bgm):
            print(f"   🎵 Mendownload Background Music (Mood: {bgm_mood})...")
            ok_bgm = download_bgm_from_pixabay_page(bgm_page, file_bgm)
            if not ok_bgm and bgm_mood != "chill":
                print("   🔄 Fallback ke BGM chill...")
                chill_page = cfg.bgm_pool["chill"]
                file_bgm = os.path.abspath(os.path.join(cfg.base_dir, "bgm_chill.mp3"))
                ok_bgm = download_bgm_from_pixabay_page(chill_page, file_bgm)

            if ok_bgm:
                print(f"   ✅ BGM siap: {file_bgm}")
            else:
                print("   ⚠️ Semua fallback gagal. Render lanjut tanpa BGM.")

        # --- Subtitle & BGM Encoding Loop (Handles dual output files if needed) ---
        runs = [m_silent] if not dev_dual else [m_silent, m_silent.replace(".ts", "_dev.ts")]
        out_targets = [m_ts] if not dev_dual else [m_ts, m_ts_dev]
        
        for input_silent_ts, output_final_ts in zip(runs, out_targets):
            lbl_suffix = "" if input_silent_ts == m_silent else " (DEV)"
            
            if aktif_bgm and os.path.exists(file_bgm):
                v_filter_parts = [f"subtitles={esc_ass_main}:fontsdir={esc_fontsdir}"] if not cfg.no_subs else ["null"]
                if cfg.video_sharpen:
                    v_filter_parts.append("unsharp=5:5:0.5:5:5:0.0")
                v_filter = ",".join(v_filter_parts)
                
                filter_complex = (
                    f"[0:v]{v_filter}[v_out]; "
                    f"[1:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,volume=1.2[voc]; "
                    f"[2:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,volume={cfg.bgm_base_volume}[bgm]; "
                    f"[bgm][voc]sidechaincompress=threshold=0.08:ratio=5.0:attack=100:release=1000[bgm_ducked]; "
                    f"[voc][bgm_ducked]amix=inputs=2:duration=first:weights=1 1:dropout_transition=2[a_out]"
                )

                cmd_m_base = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "verbose",
                    "-y",
                    "-i",
                    input_silent_ts,
                    "-ss",
                    str(m_start),
                    "-to",
                    str(m_end),
                    "-i",
                    cfg.file_video_asli,
                    "-stream_loop",
                    "-1",
                    "-i",
                    file_bgm,
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[v_out]",
                    "-map",
                    "[a_out]",
                    "-shortest",
                ] + std_p
            else:
                cmd_m_base = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "verbose",
                    "-y",
                    "-i",
                    input_silent_ts,
                    "-ss",
                    str(m_start),
                    "-to",
                    str(m_end),
                    "-i",
                    cfg.file_video_asli,
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                ]
                if not cfg.no_subs:
                    vf_main_parts = [f"subtitles={esc_ass_main}:fontsdir={esc_fontsdir}"]
                else:
                    vf_main_parts = []
                
                if cfg.video_sharpen:
                    vf_main_parts.append("unsharp=5:5:0.5:5:5:0.0")
                
                if vf_main_parts:
                    cmd_m_base += ["-vf", ",".join(vf_main_parts)]
                cmd_m_base += std_p

            cmd_m = build_ffmpeg_progress_cmd(cmd_m_base, output_final_ts)
            rc_m, err_m = run_ffmpeg_with_progress(
                cmd_m, m_end - m_start, label=f"Rank {rank} Main FFmpeg{lbl_suffix}"
            )
            if rc_m != 0:
                raise RuntimeError(f"FFmpeg main{lbl_suffix} gagal:\n" + "\n".join(err_m))

        # FINAL CONCAT
        print("   🔗 [Final] Menyelesaikan clip akhir...")
        
        # Calculate target dimensions for each run
        out_w_std, out_h_std = _get_render_dims(cfg, rasio, source_h=sh)
        if getattr(cfg, "dev_mode_with_output_merge", False):
            out_w_std, out_h_std = 2648, 1220
        elif getattr(cfg, "dev_mode", False) and not dev_dual:
            # Single stream pure dev mode
            out_w_std, out_h_std = 1920, 1080
            
        concat_runs = [(out_vid, m_ts, h_ts, (out_w_std, out_h_std))]
        if dev_dual:
            out_vid_dev = os.path.join(cfg.outputs_dir, f"highlight_rank_{rank}_dev_mode_ready.mp4")
            # Usually hook doesn't generate dual, so we fallback to standard hook for dev if missing
            h_dev_target = h_ts_dev if os.path.exists(h_ts_dev) else h_ts 
            concat_runs.append((out_vid_dev, m_ts_dev, h_dev_target, (1920, 1080)))
            
        for final_path, main_vid_ts, hook_vid_ts, dims in concat_runs:
            # Dynamically prepare glitch for THIS run's dimensions
            cur_glitch = None
            if aktif_hook:
                cur_glitch = siapkan_glitch_video(rasio, cfg, video_encoder, source_h=sh, custom_dims=dims)
            
            if aktif_hook and cur_glitch and os.path.exists(cur_glitch) and os.path.exists(hook_vid_ts):
                concat_str = f"concat:{hook_vid_ts}|{cur_glitch}|{main_vid_ts}"
            else:
                concat_str = f"concat:{main_vid_ts}"

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    concat_str,
                    "-c",
                    "copy",
                    "-bsf:a",
                    "aac_adtstoasc",
                    final_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        judul_thumbnail = judul_en or judul or f"Highlight {rank}"
        buat_thumbnail(out_vid, out_thm, judul_thumbnail, cfg)

        manifest_item["status"] = "success"
        manifest_item["video_exists"] = os.path.exists(out_vid)
        manifest_item["thumbnail_exists"] = os.path.exists(out_thm)

        print(f"✅ [Rank {rank}] Selesai.")
        return manifest_item

    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: FFmpeg gagal. Error: {e}")
        manifest_item["status"] = "failed"
        manifest_item["error"] = str(e)
        manifest_item["video_exists"] = os.path.exists(out_vid)
        manifest_item["thumbnail_exists"] = os.path.exists(out_thm)
        return manifest_item

    except Exception as e:
        print(f"\n❌ ERROR: Kegagalan tak terduga. Error: {e}")
        manifest_item["status"] = "failed"
        manifest_item["error"] = str(e)
        manifest_item["video_exists"] = os.path.exists(out_vid)
        manifest_item["thumbnail_exists"] = os.path.exists(out_thm)
        return manifest_item

    finally:
        files_to_remove = [h_ts, m_ts, a_hook, a_main, h_silent, m_silent]
        if dev_dual:
            files_to_remove.extend([h_ts_dev, m_ts_dev, m_silent.replace(".ts", "_dev.ts")])
            
        for br in broll_aktif:
            files_to_remove.append(br["filepath"])

        for f_path in files_to_remove:
            if os.path.exists(f_path):
                os.remove(f_path)
