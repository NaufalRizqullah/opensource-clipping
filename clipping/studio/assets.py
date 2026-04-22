"""
clipping.studio_assets — Asset Downloads (Font, BGM, B-Roll)

Berisi fungsi-fungsi untuk mendownload dan menyiapkan aset eksternal:
- Font dari Google Fonts (+ registrasi untuk libass)
- Background Music dari Pixabay
- B-Roll video dari Pexels API

Dipindahkan dari studio.py tanpa perubahan logic.
"""

import html
import json
import os
import random
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request

import cv2
import requests

from .utils import FIREFOX_UA

# ==============================================================================
# FONT DOWNLOADER & REGISTRATION
# ==============================================================================


def download_google_font(
    url, output_filename, font_dir, max_retry=10, min_valid_size=1000
):
    """Download font dari Google Fonts / Fontsource.

    Parameters
    ----------
    url : str
        URL download font.
    output_filename : str
        Nama file output di ``font_dir``.
    font_dir : str
        Direktori penyimpanan font.
    max_retry : int
        Jumlah percobaan maksimal.
    min_valid_size : int
        Ukuran minimum file valid (bytes).

    Returns
    -------
    bool
        True jika berhasil.
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
    """Copy fonts to system font dir and refresh cache (Linux only).

    Parameters
    ----------
    font_dir : str
        Direktori sumber font.
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
    """Siapkan font tipografi (download + registrasi).

    Parameters
    ----------
    cfg : SimpleNamespace
        Konfigurasi berisi ``daftar_font``, ``gaya_font_aktif``, ``font_dir``.

    Raises
    ------
    RuntimeError
        Jika font gagal disiapkan.
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
    """Resolve direct audio URL dari halaman Pixabay.

    Parameters
    ----------
    page_url : str
        URL halaman Pixabay music.
    timeout : int
        Timeout HTTP request.

    Returns
    -------
    str
        Direct download URL untuk file audio.

    Raises
    ------
    RuntimeError
        Jika URL tidak ditemukan di halaman.
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
    """Download BGM dari halaman Pixabay.

    Parameters
    ----------
    page_url : str
        URL halaman Pixabay music.
    output_path : str
        Path file output MP3.
    max_retry : int
        Jumlah percobaan maksimal.
    min_valid_size : int
        Ukuran minimum file valid (bytes).

    Returns
    -------
    bool
        True jika berhasil.
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
    """Download video B-Roll dari Pexels API.

    Parameters
    ----------
    query : str
        Kata kunci pencarian.
    rasio : str
        Rasio video (``"9:16"`` atau ``"16:9"``).
    output_filename : str
        Path file output.
    pexels_api_key : str
        API key Pexels.

    Returns
    -------
    bool
        True jika berhasil.
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
    """Crop dan resize gambar B-Roll ke ukuran target (center crop).

    Parameters
    ----------
    img : numpy.ndarray
        Frame gambar input.
    target_w, target_h : int
        Ukuran target.

    Returns
    -------
    numpy.ndarray
        Frame yang sudah di-crop dan resize.
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

    return cv2.resize(img, (target_w, target_h))
