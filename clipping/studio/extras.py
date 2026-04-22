"""
clipping.studio_extras — Glitch Transition & Thumbnail Generator

Berisi fungsi untuk menyiapkan video glitch transisi (download + crop)
dan membuat thumbnail dari frame video dengan overlay teks.

Dipindahkan dari studio.py tanpa perubahan logic.
"""

import os
import subprocess
import textwrap
import urllib.request

import cv2
from PIL import Image, ImageDraw, ImageFont
from yt_dlp import YoutubeDL

from .utils import get_ts_encode_args


# ==============================================================================
# GLITCH & THUMBNAIL
# ==============================================================================


def siapkan_glitch_video(rasio, cfg, video_encoder):
    """Siapkan video glitch transisi (download dari YouTube jika belum ada).

    Parameters
    ----------
    rasio : str
        Rasio video (``"9:16"`` atau ``"16:9"``).
    cfg : SimpleNamespace
        Konfigurasi berisi ``url_glitch_video``.
    video_encoder : dict
        Hasil dari ``detect_video_encoder()``.

    Returns
    -------
    str
        Path file glitch transition (``.ts``).
    """
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
    """Buat thumbnail dari frame video dengan overlay teks.

    Parameters
    ----------
    video_path : str
        Path file video sumber.
    output_image_path : str
        Path file gambar output.
    teks : str
        Teks yang akan ditampilkan di thumbnail.
    cfg : SimpleNamespace
        Konfigurasi berisi ``file_font_thumbnail``, ``url_font_thumbnail``.
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
