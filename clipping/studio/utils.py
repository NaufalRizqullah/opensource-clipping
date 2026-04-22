"""
clipping.studio_utils — Helper Umum & FFmpeg Utilities

Berisi konstanta umum, fungsi format waktu, dan semua utility terkait FFmpeg
(deteksi encoder GPU, encode args, video writer, progress tracking).

Modul ini adalah dependensi dasar untuk seluruh subsistem studio.
Dipindahkan dari studio.py tanpa perubahan logic.
"""

import subprocess

# ==============================================================================
# KONSTANTA
# ==============================================================================

FIREFOX_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0"
)


# ==============================================================================
# HELPER UMUM
# ==============================================================================


def format_seconds(seconds):
    """Format detik menjadi string HH:MM:SS.

    Parameters
    ----------
    seconds : int | float
        Jumlah detik.

    Returns
    -------
    str
        String terformat ``"HH:MM:SS"``.
    """
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def escape_ffmpeg_filter_value(value: str) -> str:
    """Escape karakter khusus untuk FFmpeg filter value.

    Parameters
    ----------
    value : str
        Nilai yang akan di-escape.

    Returns
    -------
    str
        Nilai yang sudah di-escape untuk FFmpeg.
    """
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
    """Deteksi encoder video terbaik yang tersedia (NVENC > CPU).

    Returns
    -------
    dict
        ``{"name": str, "args": list}`` — nama encoder dan argumen FFmpeg.
    """
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
    """Bangun argumen FFmpeg untuk output MPEG-TS.

    Parameters
    ----------
    video_encoder : dict
        Hasil dari ``detect_video_encoder()``.
    fps : int
        Frame rate output.

    Returns
    -------
    list[str]
        Daftar argumen FFmpeg.
    """
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
    """Bangun argumen FFmpeg untuk output MP4.

    Parameters
    ----------
    video_encoder : dict
        Hasil dari ``detect_video_encoder()``.
    fps : float
        Frame rate output.

    Returns
    -------
    list[str]
        Daftar argumen FFmpeg.
    """
    return video_encoder["args"] + [
        "-pix_fmt",
        "yuv420p",
        "-r",
        f"{fps:.06f}",
        "-movflags",
        "+faststart",
    ]


def open_ffmpeg_video_writer(output_path, width, height, fps, video_encoder):
    """Buka FFmpeg subprocess sebagai video writer (raw BGR24 stdin → MP4).

    Parameters
    ----------
    output_path : str
        Path file output video.
    width, height : int
        Resolusi output.
    fps : float
        Frame rate output.
    video_encoder : dict
        Hasil dari ``detect_video_encoder()``.

    Returns
    -------
    subprocess.Popen
        Proses FFmpeg yang menerima frame via stdin.
    """
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
    """Tambahkan flag progress ke command FFmpeg.

    Parameters
    ----------
    base_cmd : list[str]
        Command FFmpeg dasar.
    output_path : str
        Path file output.

    Returns
    -------
    list[str]
        Command FFmpeg lengkap dengan progress flag.
    """
    return base_cmd + ["-progress", "pipe:2", "-nostats", output_path]


def run_ffmpeg_with_progress(ffmpeg_cmd, total_duration, label="Render"):
    """Jalankan FFmpeg dengan progress tracking ke stdout.

    Parameters
    ----------
    ffmpeg_cmd : list[str]
        Command FFmpeg lengkap.
    total_duration : float
        Durasi total dalam detik (untuk kalkulasi persentase).
    label : str
        Label untuk log progress.

    Returns
    -------
    tuple[int, list[str]]
        (return_code, error_lines) — kode return dan baris error terakhir.
    """
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
