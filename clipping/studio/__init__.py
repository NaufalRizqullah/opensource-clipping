"""
clipping.studio — Modular Studio Coordinator

This package contains the modularized rendering pipeline for the Clipping app.
The __init__.py file acts as the coordinator/entry point, re-exporting
all public functions and implementing the orchestrator `proses_klip`.

Organization:
  - utils: FFmpeg and general helpers
  - face: Face detection (MediaPipe/YOLO)
  - assets: Asset management (Fonts, BGM, B-Roll)
  - subtitle: ASS subtitle builder
  - hybrid: Single-speaker rendering pipeline
  - split: Split-screen rendering pipeline
  - camera: Camera-switch rendering pipeline
  - extras: Glitch and thumbnails
"""

import os
import time

# Re-export everything for backward compatibility with studio.py imports
from .utils import (
    FIREFOX_UA,
    format_seconds,
    escape_ffmpeg_filter_value,
    detect_video_encoder,
    get_ts_encode_args,
    get_mp4_encode_args,
    open_ffmpeg_video_writer,
    build_ffmpeg_progress_cmd,
    run_ffmpeg_with_progress,
)

from .face import (
    get_face_detector,
    estimate_speaker_count_from_video,
)

from .assets import (
    download_google_font,
    register_fonts_for_libass,
    siapkan_font_tipografi,
    resolve_pixabay_audio_url,
    download_bgm_from_pixabay_page,
    USED_PEXELS_IDS,
    download_pexels_broll,
    crop_center_broll,
)

from .subtitle import buat_file_ass

from .hybrid import buat_video_hybrid

from .split import buat_video_split_screen

from .camera import buat_video_camera_switch

from .extras import siapkan_glitch_video, buat_thumbnail


def proses_klip(
    rank,
    metadata,
    rasio,
    file_glitch_ts,
    data_segmen,
    cfg,
    video_encoder,
    diarization_data=None,
):
    """Orchestrator utama untuk memproses satu unit klip video.

    Parameters
    ----------
    rank : int
        Peringkat klip (1, 2, 3...).
    metadata : dict
        Metadata klip dari Gemini (judul, deskripsi, start, end, dll.).
    rasio : str
        Rasio video ("9:16" atau "16:9").
    file_glitch_ts : str | None
        Path ke file transisi glitch (.ts).
    data_segmen : list[dict]
        Data transkrip per kata untuk subtitle.
    cfg : SimpleNamespace
        Konfigurasi lengkap.
    video_encoder : str
        Encoder video yang terdeteksi (h264_nvenc, libx264, dll.).
    diarization_data : list[dict] | None
        Data speaker diarization (untuk mode split/camera).
    """
    t_start_render = time.time()
    
    start_c = metadata["start_time"]
    end_c = metadata["end_time"]
    label = f"Klip #{rank}"

    out_prefix = os.path.join(cfg.outputs_dir, f"klip_{rank}")
    out_video_silent = f"{out_prefix}_silent.mp4"
    out_ass = f"{out_prefix}.ass"
    out_final_mp4 = f"{out_prefix}.mp4"
    out_final_ts = f"{out_prefix}.ts"

    # 1. Pilih Pipeline Renderer
    use_split = getattr(cfg, "use_split_screen", False) and diarization_data and rasio == "9:16"
    use_camera = getattr(cfg, "use_camera_switch", False) and diarization_data and rasio == "9:16"
    
    # Deteksi apakah butuh B-Roll data
    broll_data = metadata.get("broll_list", [])
    
    get_x_func = None
    
    if use_split:
        # Pipeline Split-Screen
        get_x_func = buat_video_split_screen(
            cfg.file_video_asli, out_video_silent, start_c, end_c, 
            diarization_data, cfg, label=f"{label} (Split)"
        )
    elif use_camera:
        # Pipeline Camera-Switch
        get_x_func = buat_video_camera_switch(
            cfg.file_video_asli, out_video_silent, start_c, end_c, 
            diarization_data, cfg, label=f"{label} (Camera)"
        )
    else:
        # Pipeline Hybrid (Default)
        get_x_func = buat_video_hybrid(
            cfg.file_video_asli, out_video_silent, start_c, end_c, 
            rasio, cfg, broll_data=broll_data, label=label
        )

    # 2. Build Subtitle
    buat_file_ass(
        data_segmen, start_c, end_c, out_ass, rasio, cfg,
        typography_plan=metadata.get("typography_plan", []),
        get_x_func=get_x_func,
        source_dim=getattr(cfg, "source_dim", (1920, 1080))
    )

    # 3. Final Assembly (Silent Video + Audio Source + Subtitle)
    # Filter komplex FFmpeg untuk menggabungkan video render dengan audio asli (cut) dan subtitle
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(max(0, start_c)), "-t", str(end_c - start_c),
        "-i", cfg.file_video_asli,  # Input 1: Audio source
        "-i", out_video_silent,      # Input 2: Video silent (9:16)
        "-vf", f"ass={escape_ffmpeg_filter_value(out_ass)}",
    ]
    
    # Tambahkan parameter encoding
    ffmpeg_cmd += get_mp4_encode_args(video_encoder)
    ffmpeg_cmd.append(out_final_mp4)

    # Jalankan assembly
    import subprocess
    subprocess.run(ffmpeg_cmd, check=True)

    # 4. Burn to TS (untuk concat transisi glitch)
    if file_glitch_ts and os.path.exists(file_glitch_ts):
        ts_args = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", out_final_mp4,
        ]
        ts_args += get_ts_encode_args(video_encoder)
        ts_args.append(out_final_ts)
        subprocess.run(ts_args, check=True)

    # 5. Generate Thumbnail (opsional)
    if getattr(cfg, "generate_thumbnails", True):
        buat_thumbnail(
            out_final_mp4, 
            f"{out_prefix}_thumb.jpg", 
            metadata.get("title_indonesia", "Klip"), 
            cfg
        )

    durasi_render = time.time() - t_start_render
    print(f"📦 {label} Berhasil! (Render: {durasi_render:.1f}s)")
    
    return {
        "rank": rank,
        "video": out_final_mp4,
        "ts": out_final_ts if file_glitch_ts else None,
        "metadata": metadata
    }
