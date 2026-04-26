"""
FFmpeg and encoder utilities for Studio rendering pipeline.
"""

import subprocess

def format_seconds(seconds):
    """
    Format a duration in seconds into HH:MM:SS.

    Args:
        seconds: Numeric duration in seconds.

    Returns:
        Duration string in `HH:MM:SS` format, clamped to non-negative.
    """
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _ffmpeg_has_encoder(name: str) -> bool:
    """
    Check whether a specific FFmpeg encoder is available.

    Args:
        name: Encoder name, for example `h264_nvenc`.

    Returns:
        True if encoder exists in local FFmpeg build.
    """
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return name in result.stdout


def _test_encoder_runtime(encoder_args):
    """
    Validate encoder arguments by running a short synthetic FFmpeg encode.

    Args:
        encoder_args: FFmpeg encoder argument list to test.

    Returns:
        Tuple `(ok, stderr_tail)` where `ok` is True on success.
    """
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


def _get_auto_bitrate(height: int) -> str:
    """Determine a safe target bitrate based on output height."""
    if height >= 2160: return "20M"
    if height >= 1440: return "12M"
    if height >= 1080: return "8M"
    return "4M"


def detect_video_encoder(cfg=None, target_h=1080):
    """
    Select the best available video encoder with conservative fallback.
    Now supports dynamic bitrate scaling for TikTok optimization.
    """
    nvenc_preset_fast = "p1"
    nvenc_preset_legacy = "fast"
    nvenc_cq = 25
    cpu_preset = "veryfast"
    cpu_crf = 25
    
    target_bitrate = "auto"

    if cfg is not None:
        nvenc_cq = int(getattr(cfg, "video_quality_cq", nvenc_cq))
        cpu_crf = int(getattr(cfg, "video_quality_crf", cpu_crf))
        target_bitrate = str(getattr(cfg, "video_bitrate", "auto")).lower()
        preset_override = str(getattr(cfg, "video_preset", "auto")).lower()
        if preset_override != "auto":
            nvenc_preset_fast = preset_override
            nvenc_preset_legacy = preset_override
            cpu_preset = preset_override

    if target_bitrate == "auto":
        target_bitrate = _get_auto_bitrate(target_h)

    nvenc_args_fastest = [
        "-c:v", "h264_nvenc",
        "-preset", nvenc_preset_fast,
        "-rc", "vbr",
        "-cq", str(nvenc_cq),
        "-b:v", target_bitrate,
        "-maxrate", f"{int(float(target_bitrate.replace('M', '')) * 1.5)}M",
        "-bufsize", f"{int(float(target_bitrate.replace('M', '')) * 2)}M",
    ]
    nvenc_args_legacy = [
        "-c:v", "h264_nvenc",
        "-preset", nvenc_preset_legacy,
        "-rc", "vbr",
        "-cq", str(nvenc_cq),
        "-b:v", target_bitrate,
        "-maxrate", f"{int(float(target_bitrate.replace('M', '')) * 1.5)}M",
        "-bufsize", f"{int(float(target_bitrate.replace('M', '')) * 2)}M",
    ]
    cpu_args = [
        "-c:v", "libx264",
        "-preset", cpu_preset,
        "-crf", str(cpu_crf),
        "-maxrate", target_bitrate,
        "-bufsize", f"{int(float(target_bitrate.replace('M', '')) * 2)}M",
    ]

    if _ffmpeg_has_encoder("h264_nvenc"):
        ok, _ = _test_encoder_runtime(nvenc_args_fastest)
        if ok:
            print(f"🚀 Pakai NVIDIA NVENC {nvenc_preset_fast} (Bitrate {target_bitrate}, CQ {nvenc_cq})", flush=True)
            return {"name": "h264_nvenc", "args": nvenc_args_fastest}

        ok, _ = _test_encoder_runtime(nvenc_args_legacy)
        if ok:
            print(f"🚀 Pakai NVIDIA NVENC {nvenc_preset_legacy} (Bitrate {target_bitrate}, CQ {nvenc_cq})", flush=True)
            return {"name": "h264_nvenc", "args": nvenc_args_legacy}

    print(f"⚠️ Fallback ke CPU libx264 ({cpu_preset}, CRF {cpu_crf}, Max {target_bitrate})", flush=True)
    return {"name": "libx264", "args": cpu_args}


def get_ts_encode_args(video_encoder, fps=30):
    """
    Build FFmpeg encode args for MPEG-TS output.

    Args:
        video_encoder: Encoder descriptor from `detect_video_encoder`.
        fps: Target frame rate.

    Returns:
        FFmpeg argument list for TS output.
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
    """
    Build FFmpeg encode args for MP4 output.

    Args:
        video_encoder: Encoder descriptor from `detect_video_encoder`.
        fps: Target frame rate.

    Returns:
        FFmpeg argument list for MP4 output.
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
    """
    Start an FFmpeg process that accepts raw BGR frames via stdin.

    Args:
        output_path: Target MP4 path.
        width: Video width in pixels.
        height: Video height in pixels.
        fps: Output frame rate.
        video_encoder: Encoder descriptor from `detect_video_encoder`.

    Returns:
        Running `subprocess.Popen` object.
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
    """
    Extend an FFmpeg command so progress can be parsed from stderr.

    Args:
        base_cmd: Base FFmpeg argument list.
        output_path: Output media path.

    Returns:
        FFmpeg command with `-progress pipe:2 -nostats`.
    """
    return base_cmd + ["-progress", "pipe:2", "-nostats", output_path]


def run_ffmpeg_with_progress(ffmpeg_cmd, total_duration, label="Render"):
    """
    Execute FFmpeg command while printing progress updates.

    Args:
        ffmpeg_cmd: Full FFmpeg command list.
        total_duration: Expected output duration in seconds.
        label: Human-readable label for logs.

    Returns:
        Tuple `(return_code, recent_errors)` where `recent_errors` contains
        the latest non-progress stderr lines.
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

