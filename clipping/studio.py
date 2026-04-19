"""
clipping.studio — Video Rendering Engine

The Studio: GPU-aware hybrid renderer with face-tracking crop, B-roll,
ASS subtitles, glitch transitions, BGM ducking, and thumbnail generation.

Maps to Cell 3 (The Studio) of the notebook.
"""

import html
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


# ==============================================================================
# HELPER UMUM
# ==============================================================================


def format_seconds(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def escape_ffmpeg_filter_value(value: str) -> str:
    return str(value).replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")


# ==============================================================================
# DETEKSI ENCODER GPU
# ==============================================================================


def _ffmpeg_has_encoder(name: str) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return name in result.stdout


def _test_encoder_runtime(encoder_args):
    cmd = (
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=640x360:r=30:d=1",
        ]
        + encoder_args
        + ["-pix_fmt", "yuv420p", "-an", "-f", "null", "-"]
    )
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return result.returncode == 0, result.stderr[-1000:]


def detect_video_encoder():
    nvenc_args_fastest = [
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p1",
        "-cq",
        "25",
        "-b:v",
        "0",
    ]
    nvenc_args_legacy = [
        "-c:v",
        "h264_nvenc",
        "-preset",
        "fast",
        "-cq",
        "25",
        "-b:v",
        "0",
    ]
    cpu_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "25"]

    if _ffmpeg_has_encoder("h264_nvenc"):
        ok, _ = _test_encoder_runtime(nvenc_args_fastest)
        if ok:
            print("🚀 Pakai NVIDIA NVENC p1", flush=True)
            return {"name": "h264_nvenc", "args": nvenc_args_fastest}

        ok, _ = _test_encoder_runtime(nvenc_args_legacy)
        if ok:
            print("🚀 Pakai NVIDIA NVENC fast", flush=True)
            return {"name": "h264_nvenc", "args": nvenc_args_legacy}

    print("⚠️ Fallback ke CPU libx264", flush=True)
    return {"name": "libx264", "args": cpu_args}


def get_ts_encode_args(video_encoder, fps=30):
    return video_encoder["args"] + [
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-f",
        "mpegts",
    ]


def get_mp4_encode_args(video_encoder, fps):
    return video_encoder["args"] + [
        "-pix_fmt",
        "yuv420p",
        "-r",
        f"{fps:.06f}",
        "-movflags",
        "+faststart",
    ]


def open_ffmpeg_video_writer(output_path, width, height, fps, video_encoder):
    cmd = (
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.06f}",
            "-i",
            "-",
        ]
        + get_mp4_encode_args(video_encoder, fps)
        + [
            "-an",
            output_path,
        ]
    )

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def build_ffmpeg_progress_cmd(base_cmd, output_path):
    return base_cmd + ["-progress", "pipe:2", "-nostats", output_path]


def run_ffmpeg_with_progress(ffmpeg_cmd, total_duration, label="Render"):
    print(f"🚀 {label} dimulai...", flush=True)

    process = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    last_percent = -1
    error_lines = []

    for raw_line in process.stderr:
        line = raw_line.strip()

        if "fontselect" in line.lower() or "using font provider" in line.lower():
            print("🔎", line, flush=True)

        if line and "=" not in line:
            error_lines.append(line)
            if len(error_lines) > 50:
                error_lines = error_lines[-50:]

        if line.startswith("out_time_ms="):
            try:
                out_time_ms = int(line.split("=", 1)[1])
                current_time = out_time_ms / 1_000_000
                percent = (
                    min(100, int((current_time / total_duration) * 100))
                    if total_duration > 0
                    else 0
                )

                if percent != last_percent:
                    print(
                        f"⏳ {label}: {percent:3d}% | "
                        f"{format_seconds(current_time)} / {format_seconds(total_duration)}",
                        flush=True,
                    )
                    last_percent = percent
            except Exception:
                pass

    return_code = process.wait()
    return return_code, error_lines[-20:]


# ==============================================================================
# MEDIAPIPE FACE DETECTOR (SINGLETON)
# ==============================================================================

_FACE_DETECTOR = None


def get_face_detector(cfg):
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

    return cv2.resize(img, (target_w, target_h))


# ==============================================================================
# HYBRID VIDEO RENDERER
# ==============================================================================


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

    video_encoder = detect_video_encoder()

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
    default_x = (width - crop_w) // 2
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
                best_x = center_x - (crop_w / 2)
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
                best_x = (largest_face.origin_x + (largest_face.width / 2)) - (
                    crop_w // 2
                )
                face_box = (
                    largest_face.origin_x,
                    largest_face.origin_y,
                    largest_face.origin_x + largest_face.width,
                    largest_face.origin_y + largest_face.height,
                )

        raw_data.append(
            {
                "time": current_time,
                "x": max(0, min(best_x, width - crop_w)),
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
        cam_x = raw_data[0]["x"]
        deadzone_px = crop_w * DEADZONE_RATIO
        snap_px = width * SNAP_THRESHOLD

        for d in raw_data:
            face_x = d["x"]

            if abs(face_x - cam_x) > snap_px:
                cam_x = face_x
            else:
                if face_x > cam_x + deadzone_px:
                    cam_x += (face_x - (cam_x + deadzone_px)) * SMOOTH_FACTOR
                elif face_x < cam_x - deadzone_px:
                    cam_x += (face_x - (cam_x - deadzone_px)) * SMOOTH_FACTOR

            final_x = int(max(0, min(cam_x, width - crop_w)))
            if smooth_data and abs(final_x - smooth_data[-1]["x"]) <= JITTER_THRESHOLD:
                final_x = smooth_data[-1]["x"]

            smooth_data.append({"time": d["time"], "x": final_x})

    def get_box(t):
        if not raw_data:
            return None
        if t <= raw_data[0]["time"]:
            return raw_data[0]["box"]
        if t >= raw_data[-1]["time"]:
            return raw_data[-1]["box"]

        for i in range(len(raw_data) - 1):
            if raw_data[i]["time"] <= t <= raw_data[i + 1]["time"]:
                b1 = raw_data[i]["box"]
                b2 = raw_data[i + 1]["box"]
                if b1 is None or b2 is None:
                    return b1 if b1 else b2
                t1, t2 = raw_data[i]["time"], raw_data[i + 1]["time"]
                frac = (t - t1) / (t2 - t1)
                return (
                    b1[0] + (b2[0] - b1[0]) * frac,
                    b1[1] + (b2[1] - b1[1]) * frac,
                    b1[2] + (b2[2] - b1[2]) * frac,
                    b1[3] + (b2[3] - b1[3]) * frac,
                )
        return None

    def get_x(t):
        if not smooth_data:
            return default_x
        if t <= smooth_data[0]["time"]:
            return smooth_data[0]["x"]
        if t >= smooth_data[-1]["time"]:
            return smooth_data[-1]["x"]

        for i in range(len(smooth_data) - 1):
            if smooth_data[i]["time"] <= t <= smooth_data[i + 1]["time"]:
                t1, t2 = smooth_data[i]["time"], smooth_data[i + 1]["time"]
                x1, x2 = smooth_data[i]["x"], smooth_data[i + 1]["x"]
                if t1 == t2:
                    return x1
                return int(x1 + (x2 - x1) * (t - t1) / (t2 - t1))

        return default_x

    # FASE 3: RENDER FRAME
    out_w, out_h = (1080, 1920) if rasio == "9:16" else (1920, 1080)
    
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
                frame_base = cv2.resize(frame_utama, (out_w, out_h))
                
                # Dim background
                frame_dev = (frame_base * 0.35).astype(np.uint8)
                
                # Highlight 9:16 window
                # Current source width/height vs canvas out_w/out_h
                scale_x = out_w / width
                cx_scaled = int(cx * scale_x)
                cw_scaled = int(crop_w * scale_x)
                
                # Paste bright crop
                frame_dev[:, cx_scaled : cx_scaled + cw_scaled] = frame_base[:, cx_scaled : cx_scaled + cw_scaled]
                
                # Draw vertical border lines
                cv2.line(frame_dev, (cx_scaled, 0), (cx_scaled, out_h), (255, 255, 255), 2)
                cv2.line(frame_dev, (cx_scaled + cw_scaled, 0), (cx_scaled + cw_scaled, out_h), (255, 255, 255), 2)
                
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
                cx = get_x(t)
                cropped = frame_utama[:, cx : cx + crop_w]
                frame_utama_siap = cv2.resize(cropped, (out_w, out_h))
                frame_terpilih = frame_utama_siap
            else:
                frame_utama_siap = cv2.resize(frame_utama, (out_w, out_h))
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

    play_res_x, play_res_y = (1080, 1920) if rasio == "9:16" else (1920, 1080)
    align = cfg.ass_align_916 if rasio == "9:16" else cfg.ass_align_169
    margin_v = cfg.ass_margin_916 if rasio == "9:16" else cfg.ass_margin_169
    font_sz = cfg.ass_font_916 if rasio == "9:16" else cfg.ass_font_169
    margin_lr = 60 if rasio == "9:16" else 40

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
                    # Calculate center of 9:16 window in source pixels
                    crop_w_src = sh * 9 // 16
                    # Reference time: midpoint of the current segment/line
                    t_ref = line["words"][0]["start"] + start_clip
                    cx = get_x_func(t_ref)
                    center_x_src = cx + (crop_w_src / 2)
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


def siapkan_glitch_video(rasio, cfg, video_encoder):
    glitch_ts = f"glitch_ready_{rasio.replace(':', '')}.ts"
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

    filter_g = (
        "crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920,setsar=1"
        if rasio == "9:16"
        else "scale=1920:1080,setsar=1"
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
    from .diarization import get_active_speaker, get_active_speakers

    STEP_DETEKSI = 0.5
    DEADZONE_RATIO = 0.25
    SMOOTH_FACTOR = 0.15
    JITTER_THRESHOLD = 5
    SNAP_THRESHOLD = 0.25
    DIVIDER_HEIGHT = 4  # px, divider between panels
    INACTIVE_ALPHA = 0.15  # darkening for inactive speaker panel
    ACTIVE_BORDER = 3  # px, highlight border for active speaker

    video_encoder = detect_video_encoder()

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

    # Output dimensions: 1080x1920 for 9:16
    out_w, out_h = 1080, 1920
    panel_h = (out_h - DIVIDER_HEIGHT) // 2  # height per speaker panel
    panel_w = out_w

    # Crop dimensions for each panel from the source frame
    # Each panel crops a region from the full frame suitable for the panel aspect ratio
    panel_ratio = panel_w / panel_h
    if (width / height) > panel_ratio:
        # source is wider — crop width
        crop_h = height
        crop_w = int(height * panel_ratio)
    else:
        # source is taller — crop height
        crop_w = width
        crop_h = int(width / panel_ratio)

    default_x = (width - crop_w) // 2
    default_y = (height - crop_h) // 2

    # ----------------------------------------------------------------
    # Determine top/bottom panel speakers by speaking time in this clip
    # ----------------------------------------------------------------
    all_speakers_in_clip = sorted(set(s["speaker"] for s in diarization_data))
    if not all_speakers_in_clip:
        all_speakers_in_clip = ["SPEAKER_00", "SPEAKER_01"]

    speaking_time: dict[str, float] = {spk: 0.0 for spk in all_speakers_in_clip}
    for seg in diarization_data:
        eff_start = max(seg["start"], start_clip)
        eff_end = min(seg["end"], end_clip)
        if eff_end > eff_start:
            speaking_time[seg["speaker"]] += eff_end - eff_start

    # Top-2 speakers by speaking time own the permanent panel slots
    ranked = sorted(all_speakers_in_clip, key=lambda s: speaking_time[s], reverse=True)
    speaker_top = ranked[0]
    speaker_bottom = ranked[1] if len(ranked) > 1 else ranked[0]
    extra_speakers = ranked[2:]  # 3rd, 4th, … speakers

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
            yolo_results = yolo_model(frame, verbose=False)
            if yolo_results and len(yolo_results[0].boxes) > 0:
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
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
        active_now = get_active_speakers(diarization_data, start_clip + current_time)
        n_faces = len(face_centers)
        n_active = len(active_now)

        # Pass A: collect face data and track canonical solo observations.
        # Full assignment happens in Pass B after canonical_cx is built from entire clip.
        if n_faces == 1 and n_active == 1:
            spk = active_now[0]
            speaker_solo_cxs.setdefault(spk, []).append(face_centers[0][0])
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
        spk: _stats.median(cxs) for spk, cxs in speaker_solo_cxs.items() if cxs
    }

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
                raw_data[spk].append({"time": fd["time"], "x": _clamp_x(fc_list[0][0])})

        elif nf >= 2 and na >= 2:
            # Greedy canonical assignment: sort active speakers by canonical X,
            # each picks their nearest remaining face.
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
                raw_data[spk].append({"time": fd["time"], "x": _clamp_x(best[0])})

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
                last_cx = raw_data[spk][-1]["x"] + crop_w / 2
                best = min(pool, key=lambda fc: abs(fc[0] - last_cx))
            elif spk in speaker_canonical_cx:
                best = min(pool, key=lambda fc: abs(fc[0] - speaker_canonical_cx[spk]))
            else:
                best = pool[len(pool) // 2]
            raw_data[spk].append({"time": fd["time"], "x": _clamp_x(best[0])})

    # ---- FASE 2: SMOOTH CAMERA PER SPEAKER ----
    def _smooth_positions(raw_data):
        smooth = []
        if not raw_data:
            return smooth
        cam_x = raw_data[0]["x"]
        deadzone_px = crop_w * DEADZONE_RATIO
        snap_px = width * SNAP_THRESHOLD

        for d in raw_data:
            face_x = d["x"]
            if abs(face_x - cam_x) > snap_px:
                cam_x = face_x
            else:
                if face_x > cam_x + deadzone_px:
                    cam_x += (face_x - (cam_x + deadzone_px)) * SMOOTH_FACTOR
                elif face_x < cam_x - deadzone_px:
                    cam_x += (face_x - (cam_x - deadzone_px)) * SMOOTH_FACTOR

            final_x = int(max(0, min(cam_x, width - crop_w)))
            if smooth and abs(final_x - smooth[-1]["x"]) <= JITTER_THRESHOLD:
                final_x = smooth[-1]["x"]
            smooth.append({"time": d["time"], "x": final_x})
        return smooth

    smooth: dict[str, list] = {
        spk: _smooth_positions(raw_data[spk]) for spk in all_speakers_in_clip
    }

    def _get_x(speaker: str, t: float) -> int:
        sd = smooth.get(speaker, [])
        if not sd:
            return default_x
        if t <= sd[0]["time"]:
            return sd[0]["x"]
        if t >= sd[-1]["time"]:
            return sd[-1]["x"]
        for i in range(len(sd) - 1):
            if sd[i]["time"] <= t <= sd[i + 1]["time"]:
                t1, t2 = sd[i]["time"], sd[i + 1]["time"]
                x1, x2 = sd[i]["x"], sd[i + 1]["x"]
                if t1 == t2:
                    return x1
                return int(x1 + (x2 - x1) * (t - t1) / (t2 - t1))
        return default_x

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
    writer = open_ffmpeg_video_writer(
        output_video, out_w, out_h, orig_fps, video_encoder
    )

    # Pre-create overlay for inactive speaker
    dark_overlay = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    # Per-speaker frozen frame cache: stores last valid crop for each speaker
    # so any panel can fallback when its speaker is not in the current scene
    last_valid_crop: dict[str, np.ndarray] = {}

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_clip * 1000)
        frame_count = 0
        last_render_percent = -1

        print(f"🎬 {label} - Render split-screen dimulai...", flush=True)

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
            active_speaker = get_active_speaker(diarization_data, timestamp_abs)

            # Determine which speakers populate each panel this frame
            # Extra speakers (3rd, 4th…) temporarily take over one panel
            if active_speaker and active_speaker not in (speaker_top, speaker_bottom):
                # 3rd-party speaker is active → show them in top panel,
                # keep the OTHER primary speaker (not the active extra) in bottom
                panel_top_spk = active_speaker
                panel_bottom_spk = speaker_bottom
            else:
                panel_top_spk = speaker_top
                panel_bottom_spk = speaker_bottom

            # Detect whether the current scene is a solo shot
            in_solo_scene = bool(
                active_speaker and speaker_is_solo.get(active_speaker, False)
            )

            # ---- Helper: build a panel crop for a given speaker ----
            def _build_panel(spk, is_other_panel_spk=False):
                """Return (panel_img, used_fallback) for the given speaker."""
                # Check if this speaker is NOT visible in the current frame
                spk_not_visible = (
                    active_speaker in all_speakers_in_clip
                    and active_speaker != spk
                    and (
                        speaker_is_solo.get(spk, False)
                        or (
                            in_solo_scene
                            and not speaker_is_solo.get(spk, False)
                        )
                    )
                )
                if spk_not_visible:
                    # Use frozen frame for this speaker
                    if spk in last_valid_crop:
                        return last_valid_crop[spk].copy(), True
                    else:
                        return cv2.resize(
                            frame[0:crop_h, default_x : default_x + crop_w],
                            (panel_w, panel_h),
                        ), True
                else:
                    cx = _get_x(spk, t)
                    crop = cv2.resize(
                        frame[0:crop_h, cx : cx + crop_w],
                        (panel_w, panel_h),
                    )
                    # Update frozen frame cache when speaker is actually visible
                    if not in_solo_scene or active_speaker == spk:
                        last_valid_crop[spk] = crop.copy()
                    return crop, False

            # ---- Top panel ----
            panel_top, is_fallback_top = _build_panel(panel_top_spk)

            # ---- Bottom panel ----
            panel_bottom, is_fallback_bottom = _build_panel(panel_bottom_spk)

            # ---- Active / inactive highlighting ----
            if active_speaker == panel_top_spk:
                panel_bottom = cv2.addWeighted(
                    panel_bottom, 1.0 - INACTIVE_ALPHA, dark_overlay, INACTIVE_ALPHA, 0
                )
                cv2.rectangle(
                    panel_top,
                    (0, 0),
                    (panel_w - 1, panel_h - 1),
                    (0, 255, 255),
                    ACTIVE_BORDER,
                )
            elif active_speaker == panel_bottom_spk:
                panel_top = cv2.addWeighted(
                    panel_top, 1.0 - INACTIVE_ALPHA, dark_overlay, INACTIVE_ALPHA, 0
                )
                cv2.rectangle(
                    panel_bottom,
                    (0, 0),
                    (panel_w - 1, panel_h - 1),
                    (0, 255, 255),
                    ACTIVE_BORDER,
                )
            else:
                # Dim any fallback panels so the active panel stands out
                if is_fallback_top:
                    panel_top = cv2.addWeighted(
                        panel_top, 1.0 - INACTIVE_ALPHA, dark_overlay, INACTIVE_ALPHA, 0
                    )
                if is_fallback_bottom:
                    panel_bottom = cv2.addWeighted(
                        panel_bottom, 1.0 - INACTIVE_ALPHA, dark_overlay, INACTIVE_ALPHA, 0
                    )

            # Compose the final frame: top panel + divider + bottom panel
            divider = np.full(
                (DIVIDER_HEIGHT, panel_w, 3), 80, dtype=np.uint8
            )  # gray divider
            final_frame = np.vstack([panel_top, divider, panel_bottom])

            # Ensure exact output dimensions
            if final_frame.shape[0] != out_h or final_frame.shape[1] != out_w:
                final_frame = cv2.resize(final_frame, (out_w, out_h))

            writer.stdin.write(final_frame.tobytes())
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

        writer.stdin.close()
        stderr_data = writer.stderr.read().decode("utf-8", errors="ignore")
        return_code = writer.wait()

        if return_code != 0:
            raise RuntimeError(f"FFmpeg writer gagal: {stderr_data[-1000:]}")

        print(f"✅ {label} selesai.", flush=True)

    finally:
        cap.release()


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
    from .diarization import get_active_speakers

    STEP_DETEKSI = 0.5
    DEADZONE_RATIO = 0.25
    SMOOTH_FACTOR = 0.15
    JITTER_THRESHOLD = 5
    SNAP_THRESHOLD = 0.25
    MIN_HOLD = float(getattr(cfg, "switch_hold_duration", 2.0))
    BLUR_KERNEL = 99
    BLUR_SIGMA = 30

    video_encoder = detect_video_encoder()

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
            speaker_solo_cxs.setdefault(spk, []).append(face_centers[0][0])
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
        spk: _stats.median(cxs) for spk, cxs in speaker_solo_cxs.items() if cxs
    }

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
        fc_list = fd["face_centers"]
        act_list = fd["active_now"]
        nf = len(fc_list)
        na = len(act_list)

        if nf == 0 or na == 0:
            continue

        if nf == 1 and na == 1:
            spk = act_list[0]
            if spk in raw_data:
                raw_data[spk].append(
                    {"time": fd["time"], "x": _clamp_x_cs(fc_list[0][0])}
                )

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
                raw_data[spk].append({"time": fd["time"], "x": _clamp_x_cs(best[0])})

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
                last_cx = raw_data[spk][-1]["x"] + crop_w / 2
                best = min(pool, key=lambda fc: abs(fc[0] - last_cx))
            elif spk in speaker_canonical_cx:
                best = min(pool, key=lambda fc: abs(fc[0] - speaker_canonical_cx[spk]))
            else:
                best = pool[len(pool) // 2]
            raw_data[spk].append({"time": fd["time"], "x": _clamp_x_cs(best[0])})

    # ================================================================
    # FASE 2 — Smooth per-speaker camera positions
    # ================================================================
    def _smooth_positions(raw_list):
        smooth = []
        if not raw_list:
            return smooth
        cam_x = raw_list[0]["x"]
        deadzone_px = crop_w * DEADZONE_RATIO
        snap_px = width * SNAP_THRESHOLD
        for d in raw_list:
            face_x = d["x"]
            if abs(face_x - cam_x) > snap_px:
                cam_x = face_x
            else:
                if face_x > cam_x + deadzone_px:
                    cam_x += (face_x - (cam_x + deadzone_px)) * SMOOTH_FACTOR
                elif face_x < cam_x - deadzone_px:
                    cam_x += (face_x - (cam_x - deadzone_px)) * SMOOTH_FACTOR
            final_x = int(max(0, min(cam_x, width - crop_w)))
            if smooth and abs(final_x - smooth[-1]["x"]) <= JITTER_THRESHOLD:
                final_x = smooth[-1]["x"]
            smooth.append({"time": d["time"], "x": final_x})
        return smooth

    smooth: dict[str, list] = {
        spk: _smooth_positions(raw_data[spk]) for spk in speakers
    }

    def _get_x(speaker, t):
        sd = smooth.get(speaker, [])
        if not sd:
            return default_x
        if t <= sd[0]["time"]:
            return sd[0]["x"]
        if t >= sd[-1]["time"]:
            return sd[-1]["x"]
        for i in range(len(sd) - 1):
            if sd[i]["time"] <= t <= sd[i + 1]["time"]:
                t1, t2 = sd[i]["time"], sd[i + 1]["time"]
                x1, x2 = sd[i]["x"], sd[i + 1]["x"]
                if t1 == t2:
                    return x1
                return int(x1 + (x2 - x1) * (t - t1) / (t2 - t1))
        return default_x

    # ----------------------------------------------------------------
    # Helper: blurred pillarbox for wide-shot / simultaneous speech
    # ----------------------------------------------------------------
    def _make_blurred_pillarbox(frame):
        h, w = frame.shape[:2]
        # Background: scale to cover 1080×1920, crop centre, then blur
        scale = max(out_w / w, out_h / h)
        new_w = max(out_w, int(w * scale))
        new_h = max(out_h, int(h * scale))
        bg = cv2.resize(frame, (new_w, new_h))
        y0 = (new_h - out_h) // 2
        x0 = (new_w - out_w) // 2
        bg = bg[y0 : y0 + out_h, x0 : x0 + out_w]
        ksize = BLUR_KERNEL if BLUR_KERNEL % 2 == 1 else BLUR_KERNEL + 1
        bg = cv2.GaussianBlur(bg, (ksize, ksize), BLUR_SIGMA)
        # Foreground: scale frame to width=out_w, preserve aspect ratio
        fg_w = out_w
        fg_h = min(out_h, int(h * out_w / w))
        fg = cv2.resize(frame, (fg_w, fg_h))
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
    out_w, out_h = (1080, 1920) # Assume 9:16 target
    
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
                frame_base = cv2.resize(frame, (out_w, out_h))
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
                        cx = _get_x(current_speaker, t)
                        cx_scaled = int(cx * scale_x)
                        cw_scaled = int(crop_w * scale_x)
                        
                        # Paste bright crop
                        frame_dev[:, cx_scaled : cx_scaled + cw_scaled] = frame_base[:, cx_scaled : cx_scaled + cw_scaled]
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
                            cx = _get_x(current_speaker, t)
                            cx_scaled = int(cx * scale_x)
                            cw_scaled = int(crop_w * scale_x)
                            
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
                    cx = _get_x(current_speaker, t)
                    crop_fr = frame[0:crop_h, cx : cx + crop_w]
                    out_frame = cv2.resize(crop_fr, (out_w, out_h))

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
                cx = _get_x(current_speaker, t)
                crop_fr = frame[0:crop_h, cx : cx + crop_w]
                out_frame = cv2.resize(crop_fr, (out_w, out_h))

            else:
                if current_speaker is not None:
                    cx = _get_x(current_speaker, t)
                    crop_fr = frame[0:crop_h, cx : cx + crop_w]
                    out_frame = cv2.resize(crop_fr, (out_w, out_h))
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

    finally:
        cap.release()


# ==============================================================================
# PROSES KLIP UTAMA
# ==============================================================================


def proses_klip(
    rank, clip, rasio, glitch_ts, data_segmen, cfg, video_encoder, diarization_data=None
):
    get_x_h = None
    get_x_main = None
    h_start = float(clip.get("hook_start_time", clip["start_time"]))
    h_end = float(
        clip.get(
            "hook_end_time",
            clip.get("hook_start_time", clip["start_time"]) + cfg.durasi_hook,
        )
    )
    m_start = float(clip["start_time"])
    m_end = float(clip["end_time"])
    judul = clip.get("title_indonesia")
    judul_en = clip.get("title_inggris")
    out_vid = os.path.join(cfg.outputs_dir, f"highlight_rank_{rank}_ready.mp4")
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

    aktif_hook = cfg.use_hook_glitch

    # Determine if we should use split-screen mode
    use_split = (
        getattr(cfg, "use_split_screen", False)
        and rasio == "9:16"
        and diarization_data
        and len(set(s["speaker"] for s in diarization_data)) >= 2
    )

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
                print("   📸 [Hook] Split-screen render...")
                buat_video_split_screen(
                    cfg.file_video_asli,
                    h_silent,
                    h_start,
                    h_end,
                    diarization_data,
                    cfg,
                    label=f"Rank {rank} Hook SplitScreen",
                )
            elif use_camera_switch:
                print("   📸 [Hook] Camera switch render...")
                get_x_h = buat_video_camera_switch(
                    cfg.file_video_asli,
                    h_silent,
                    h_start,
                    h_end,
                    diarization_data,
                    cfg,
                    label=f"Rank {rank} Hook CameraSwitch",
                )
            else:
                print("   📸 [Hook] Hybrid render...")
                get_x_h = buat_video_hybrid(
                    cfg.file_video_asli,
                    h_silent,
                    h_start,
                    h_end,
                    rasio,
                    cfg,
                    label=f"Rank {rank} Hook",
                )
            
            aktif_advanced_hook = cfg.use_advanced_text_on_hook
            if not cfg.no_subs:
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
                vf_hook = f"subtitles={esc_ass_hook}:fontsdir={esc_fontsdir}"
            else:
                print("   🎬 [Hook] Skip subtitle rendering...")
                vf_hook = None

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
                cfg.file_video_asli,
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
            buat_video_split_screen(
                cfg.file_video_asli,
                m_silent,
                m_start,
                m_end,
                diarization_data,
                cfg,
                label=f"Rank {rank} Main SplitScreen",
            )
        elif use_camera_switch:
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

        if aktif_bgm and os.path.exists(file_bgm):
            v_filter = f"subtitles={esc_ass_main}:fontsdir={esc_fontsdir}" if not cfg.no_subs else "null"
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
                m_silent,
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
                m_silent,
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
                vf_main = f"subtitles={esc_ass_main}:fontsdir={esc_fontsdir}"
                cmd_m_base += ["-vf", vf_main]
            cmd_m_base += std_p

        cmd_m = build_ffmpeg_progress_cmd(cmd_m_base, m_ts)
        rc_m, err_m = run_ffmpeg_with_progress(
            cmd_m, m_end - m_start, label=f"Rank {rank} Main FFmpeg"
        )
        if rc_m != 0:
            raise RuntimeError("FFmpeg main gagal:\n" + "\n".join(err_m))

        # FINAL CONCAT
        print("   🔗 [Final] Menyelesaikan clip akhir...")
        if aktif_hook and glitch_ts and os.path.exists(glitch_ts):
            concat_str = f"concat:{h_ts}|{glitch_ts}|{m_ts}"
        else:
            concat_str = f"concat:{m_ts}"

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
                out_vid,
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
        for br in broll_aktif:
            files_to_remove.append(br["filepath"])

        for f_path in files_to_remove:
            if os.path.exists(f_path):
                os.remove(f_path)
