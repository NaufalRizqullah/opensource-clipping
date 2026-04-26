"""
clipping.config — Master Configuration (Dashboard)

Menyimpan semua default value dan membangun config dari CLI args.
"""

import argparse
import os
from types import SimpleNamespace

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ==============================================================================
# DEFAULT VALUES  (sama persis dengan Cell 0 notebook)
# ==============================================================================

BASE_DIR = os.getcwd()
FONT_DIR = os.path.abspath(os.path.join(BASE_DIR, "custom_fonts"))

# 1. PENGATURAN UTAMA
JUMLAH_CLIP = 7
PILIHAN_RASIO = "9:16"

# 2. PENGATURAN KONTEN & HOOK
MAX_KATA_PER_SUBTITLE = 5
DURASI_HOOK = 3
USE_BROLL = True
USE_HOOK_GLITCH = True
USE_SPLIT_SCREEN = False
USE_CAMERA_SWITCH = False
DIARIZATION_NUM_SPEAKERS = "auto"
SWITCH_HOLD_DURATION = 2.0

# 3. PENGATURAN SUBTITLE & TIPOGRAFI (ASS STYLE)
USE_ADVANCED_TEXT = False
USE_ADVANCED_TEXT_ON_HOOK = False
USE_KARAOKE_EFFECT = True

GAYA_FONT_AKTIF = "HORMOZI"

DAFTAR_FONT = {
    "DEFAULT": {
        "utama": {
            "nama": "Montserrat Black",
            "file": "Montserrat-Black.ttf",
            "url": "https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-Black.ttf",
            "bold": 1,
        },
        "khusus": {
            "nama": "Montserrat Medium",
            "file": "Montserrat-Medium.ttf",
            "url": "https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-Medium.ttf",
            "bold": 0,
        },
    },
    "STORYTELLER": {
        "utama": {
            "nama": "Inter",
            "file": "Inter-Regular.ttf",
            "url": "https://cdn.jsdelivr.net/fontsource/fonts/inter@latest/latin-400-normal.ttf",
            "bold": 0,
        },
        "khusus": {
            "nama": "Lora",
            "file": "Lora-Bold.ttf",
            "url": "https://cdn.jsdelivr.net/fontsource/fonts/lora@latest/latin-700-normal.ttf",
            "bold": 1,
        },
    },
    "HORMOZI": {
        "utama": {
            "nama": "Montserrat",
            "file": "Montserrat-Regular.ttf",
            "url": "https://cdn.jsdelivr.net/fontsource/fonts/montserrat@latest/latin-400-normal.ttf",
            "bold": 0,
        },
        "khusus": {
            "nama": "Anton",
            "file": "Anton-Regular.ttf",
            "url": "https://cdn.jsdelivr.net/fontsource/fonts/anton@latest/latin-400-normal.ttf",
            "bold": 0,
        },
    },
    "CINEMATIC": {
        "utama": {
            "nama": "Roboto",
            "file": "Roboto-Regular.ttf",
            "url": "https://cdn.jsdelivr.net/fontsource/fonts/roboto@latest/latin-400-normal.ttf",
            "bold": 0,
        },
        "khusus": {
            "nama": "Bebas Neue",
            "file": "BebasNeue-Regular.ttf",
            "url": "https://cdn.jsdelivr.net/fontsource/fonts/bebas-neue@latest/latin-400-normal.ttf",
            "bold": 0,
        },
    },
}

# Khusus 9:16 (Vertikal)
ASS_ALIGN_916 = 2
ASS_MARGIN_916 = 450
ASS_FONT_916 = 90
SCALE_KATA_KHUSUS_916 = ASS_FONT_916 + 120

# Khusus 16:9 (Horizontal)
ASS_ALIGN_169 = 2
ASS_MARGIN_169 = 70
ASS_FONT_169 = 80
SCALE_KATA_KHUSUS_169 = ASS_FONT_169 + 120

# Warna Kata Khusus  (Format ASS: BGR -> &H[Blue][Green][Red]&)
WARNA_KATA_KHUSUS = "&HFFFFFF&"

# 4. PENGATURAN ASSET EKSTERNAL
NAMA_FONT_THUMBNAIL = "Montserrat-Black.ttf"
URL_FONT_THUMBNAIL = (
    "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Black.ttf"
)

URL_GLITCH_VIDEO = "https://www.youtube.com/watch?v=5nBcNRYmjs0"
URL_MEDIAPIPE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"

# 5. PENGATURAN Auto-BGM & Audio Ducking
USE_AUTO_BGM = True
BGM_BASE_VOLUME = 0.25

BGM_POOL = {
    "chill": "https://pixabay.com/music/beats-lofi-study-calm-peaceful-chill-hop-112191/",
    "epic": "https://pixabay.com/music/main-title-epic-epic-trailer-music-325357/",
    "sad": "https://pixabay.com/music/modern-classical-sad-sorrowful-piano-music-330768/",
    "upbeat": "https://pixabay.com/music/upbeat-pop-upbeat-pop-music-307007/",
    "suspense": "https://pixabay.com/music/ambient-dark-ambient-background-music-342437/",
}

# Whisper
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
DOWNLOAD_SOURCE_HEIGHT = "max"
VIDEO_QUALITY_CQ = 23
VIDEO_QUALITY_CRF = 20
VIDEO_PRESET = "auto"
VIDEO_SCALE_ALGO = "lanczos"
RENDER_OUTPUT_HEIGHT = 1080

# AI Provider
AI_PROVIDER = "gemini"
NVIDIA_MODEL = "deepseek-ai/deepseek-v4-pro"
GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash"


# ==============================================================================
# CLI PARSER
# ==============================================================================


def _parse_speakers(val: str) -> str | int:
    if val.lower() == "auto":
        return "auto"
    try:
        return int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{val}' is not a valid integer or 'auto'")


def _parse_download_height(val: str) -> str | int:
    """
    Parse desired download source height.

    Accepts:
    - `max` to always prefer the highest available quality.
    - positive integers like 1080, 1440, 2160 to cap source resolution.
    """
    if val.lower() == "max":
        return "max"
    try:
        parsed = int(val)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"'{val}' is not valid. Use 'max' or an integer height (e.g. 1080, 1440, 2160)."
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Download source height must be a positive integer.")
    return parsed

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="🎬 OpenSource Clipping — AI Auto-Clipper & Teaser Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Pengaturan utama ---
    p.add_argument(
        "--url", "-u", required=True, help="YouTube video URL to process"
    )
    p.add_argument(
        "--clips",
        "-n",
        type=int,
        default=JUMLAH_CLIP,
        help="Number of highlight clips to generate",
    )
    p.add_argument(
        "--ratio",
        "-r",
        default=PILIHAN_RASIO,
        choices=["9:16", "16:9"],
        help="Output aspect ratio",
    )
    p.add_argument(
        "--source-height",
        type=_parse_download_height,
        default=DOWNLOAD_SOURCE_HEIGHT,
        help="Preferred source download max height. Use 'max' to fetch highest available quality.",
    )
    p.add_argument(
        "--render-height",
        default=str(RENDER_OUTPUT_HEIGHT),
        help="Target output height for the render. Use 'source' to match the source video height, or a number (e.g. 1080, 1440).",
    )

    # --- Konten & Hook ---
    p.add_argument(
        "--words-per-sub",
        type=int,
        default=MAX_KATA_PER_SUBTITLE,
        help="Max words per karaoke subtitle group",
    )
    p.add_argument(
        "--hook-duration",
        type=int,
        default=DURASI_HOOK,
        help="Hook teaser duration in seconds",
    )
    p.add_argument(
        "--hook-source",
        default=None,
        help="Google Drive URL or local path for a single custom hook video (.mp4)",
    )
    p.add_argument(
        "--hook-source-start",
        type=float,
        default=0.0,
        help="Start time in seconds for the custom hook video",
    )
    p.add_argument("--no-broll", action="store_true", help="Disable B-roll footage")
    p.add_argument("--no-hook", action="store_true", help="Disable hook glitch teaser")
    p.add_argument("--no-bgm", action="store_true", help="Disable background music")
    p.add_argument(
        "--no-karaoke",
        action="store_true",
        help="Disable karaoke highlight effect (use clean text instead)",
    )
    p.add_argument(
        "--split-screen",
        action="store_true",
        default=USE_SPLIT_SCREEN,
        help="Enable split-screen mode for podcast with 2 speakers (9:16 only, requires HF_TOKEN for Pyannote)",
    )
    p.add_argument(
        "--diarization-speakers",
        type=_parse_speakers,
        default=DIARIZATION_NUM_SPEAKERS,
        help="Number of speakers for diarization, or 'auto' to auto-detect visually (used with --split-screen or --camera-switch)",
    )
    p.add_argument(
        "--camera-switch",
        action="store_true",
        default=USE_CAMERA_SWITCH,
        help="Enable camera-switch mode for podcast (9:16 only, requires HF_TOKEN). "
        "Mutually exclusive with --split-screen; split-screen takes precedence if both are set.",
    )
    p.add_argument(
        "--switch-hold-duration",
        type=float,
        default=SWITCH_HOLD_DURATION,
        help="Minimum seconds to hold on the current speaker before switching cameras (camera-switch mode only)",
    )
    p.add_argument(
        "--no-subs",
        action="store_true",
        help="Disable all subtitle rendering (useful if you only want the video without text)",
    )
    p.add_argument(
        "--dynamic-split",
        action="store_true",
        help="Automatically switch between full-screen (1 speaker) and split-screen (2 speakers) based on who is talking. Only active with --split-screen.",
    )
    p.add_argument(
        "--split-trigger",
        choices=["diarization", "face"],
        default="diarization",
        help="The trigger used to decide when to split the screen. 'diarization' uses audio (who is talking), 'face' uses video (how many faces are visible).",
    )


    # --- Split Screen Optimizations ---
    p.add_argument(
        "--split-zoom",
        type=float,
        default=1.0,
        help="Zoom factor for split-screen panels (e.g. 1.2, 1.5). Default is 1.0 (no zoom).",
    )
    p.add_argument(
        "--split-v-align",
        type=float,
        default=0.5,
        help="Vertical alignment for split-screen panels (0.0=top, 0.5=center, 1.0=bottom). Default is 0.5 (center).",
    )
    p.add_argument(
        "--split-auto-zoom",
        action="store_true",
        help="Automatically zoom in each split-screen panel until only one person is visible in each frame.",
    )
    p.add_argument(
        "--split-max-zoom",
        type=float,
        default=2.5,
        help="Maximum zoom factor allowed for auto-zoom (default: 2.5).",
    )


    # --- Subtitle & Tipografi ---
    p.add_argument(
        "--font-style",
        default=GAYA_FONT_AKTIF,
        choices=["DEFAULT", "STORYTELLER", "HORMOZI", "CINEMATIC"],
        help="Font style preset",
    )
    p.add_argument(
        "--advanced-text",
        action="store_true",
        default=USE_ADVANCED_TEXT,
        help="Enable advanced kinetic typography",
    )
    p.add_argument(
        "--advanced-text-hook",
        action="store_true",
        default=USE_ADVANCED_TEXT_ON_HOOK,
        help="Enable advanced typography on hook",
    )

    # --- Whisper ---
    p.add_argument(
        "--use-dlp-subs",
        action="store_true",
        help="Use yt-dlp to download auto/manual subtitles to speed up process (skipping Whisper if found)",
    )
    p.add_argument(
        "--whisper-model", default=WHISPER_MODEL, help="Faster-Whisper model size"
    )
    p.add_argument(
        "--whisper-device",
        default=WHISPER_DEVICE,
        choices=["cuda", "cpu", "auto"],
        help="Device for Whisper inference",
    )
    p.add_argument(
        "--whisper-compute-type",
        default=WHISPER_COMPUTE_TYPE,
        help="Compute type for Whisper (float16, int8, etc.)",
    )

    # --- Gemini & Face Detection ---
    p.add_argument(
        "--face-detector",
        choices=["mediapipe", "yolo"],
        default="mediapipe",
        help="AI model for face tracking (mediapipe is CPU, yolo uses GPU if available)",
    )
    p.add_argument(
        "--yolo-size",
        choices=["8n", "8s", "8m", "8n_v2", "9c"],
        default="8m",
        help="YOLO face model version/size (8n, 8s, 8m, 8n_v2, 9c). Only active if --face-detector yolo",
    )
    p.add_argument(
        "--ai-provider",
        choices=["gemini", "nvidia"],
        default=AI_PROVIDER,
        help="AI provider for video analysis (gemini or nvidia).",
    )
    p.add_argument(
        "--nvidia-model",
        default=NVIDIA_MODEL,
        help="Model name for NVIDIA NIM API (e.g. deepseek-ai/deepseek-v3).",
    )
    p.add_argument("--gemini-model", default=GEMINI_MODEL, help="Gemini model name")
    p.add_argument(
        "--gemini-fallback-model",
        default=GEMINI_FALLBACK_MODEL,
        help="Gemini fallback model name if main model fails",
    )
    p.add_argument(
        "--load-gemini-json",
        action="store_true",
        help="Load the saved gemini_response.json from outputs dir to bypass the AI generation step (useful for debugging)",
    )
    p.add_argument(
        "--box-face-detection",
        action="store_true",
        help="Draw a yellow bounding box around the detected face for debugging/tracking visualization",
    )
    p.add_argument(
        "--dev-mode",
        action="store_true",
        help="Enable developer visualization mode for 9:16 tracking (shows stabilization box and dimmed background)",
    )
    p.add_argument(
        "--dev-mode-with-output",
        action="store_true",
        help="Render BOTH the Dev Mode visualization AND the standard output video simultaneously.",
    )
    p.add_argument(
        "--dev-mode-with-output-merge",
        action="store_true",
        help="Render a merged side-by-side video of both Dev Mode and standard output.",
    )
    p.add_argument(
        "--track-lines",
        action="store_true",
        help="Draw crosshair tracking lines extending from the face box to the boundaries",
    )

    # --- Smart Auto-Framing / Tracking ---
    p.add_argument(
        "--track-step",
        type=float,
        default=None,
        help="Face detection frequency in seconds (default: 0.25)",
    )
    p.add_argument(
        "--track-deadzone",
        type=float,
        default=None,
        help="Camera deadzone ratio (default: 0.15)",
    )
    p.add_argument(
        "--track-smooth",
        type=float,
        default=None,
        help="Camera smoothing speed (default: 0.30)",
    )
    p.add_argument(
        "--track-jitter",
        type=int,
        default=None,
        help="Pixel jitter threshold (default: 5)",
    )
    p.add_argument(
        "--track-snap",
        type=float,
        default=None,
        help="Face jump snap threshold (default: 0.25)",
    )
    p.add_argument(
        "--track-conf",
        type=float,
        default=0.55,
        help="[Experimental] Higher confidence threshold for face detection to prevent ghosts (default: 0.55)",
    )
    p.add_argument(
        "--track-smooth-window",
        type=int,
        default=12,
        help="[Experimental] Majority-vote window for layout stability (default: 12 frames)",
    )
    p.add_argument(
        "--scene-cut-threshold",
        type=int,
        default=18,
        help="[Experimental] Visibility change threshold to detect camera cuts and reset layout history (default: 18)",
    )
    p.add_argument(
        "--track-iou-threshold",
        type=float,
        default=0.2,
        help="[Experimental] Box overlap threshold to merge duplicate detections (default: 0.2)",
    )
    p.add_argument(
        "--video-bitrate",
        default="auto",
        help="Target video bitrate (e.g. 8M, 12M, auto). 'auto' scales based on resolution.",
    )
    p.add_argument(
        "--video-sharpen",
        action="store_true",
        help="Apply a subtle sharpening filter for clearer output.",
    )
    p.add_argument(
        "--video-cq",
        type=int,
        default=VIDEO_QUALITY_CQ,
        help="NVENC constant quality value (lower is sharper, bigger file).",
    )
    p.add_argument(
        "--video-crf",
        type=int,
        default=VIDEO_QUALITY_CRF,
        help="libx264 CRF value (lower is sharper, bigger file).",
    )
    p.add_argument(
        "--video-preset",
        default=VIDEO_PRESET,
        help="Override encoder preset for NVENC/libx264, or 'auto' to keep defaults.",
    )
    p.add_argument(
        "--video-scale-algo",
        choices=["lanczos", "bicubic", "bilinear", "area"],
        default=VIDEO_SCALE_ALGO,
        help="Resize algorithm for OpenCV scaling steps during rendering.",
    )

    return p


def build_config(argv: list[str] | None = None) -> SimpleNamespace:
    """Parse CLI args and merge with defaults into a config namespace."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    base_dir = os.getcwd()
    outputs_dir = os.path.abspath(os.path.join(base_dir, "outputs"))
    os.makedirs(outputs_dir, exist_ok=True)
    font_dir = os.path.abspath(os.path.join(base_dir, "custom_fonts"))
    os.makedirs(font_dir, exist_ok=True)

    cfg = SimpleNamespace(
        # Paths
        base_dir=base_dir,
        outputs_dir=outputs_dir,
        font_dir=font_dir,
        file_video_asli=os.path.abspath(os.path.join(base_dir, "video_asli.mp4")),
        file_font_thumbnail=os.path.abspath(
            os.path.join(base_dir, NAMA_FONT_THUMBNAIL)
        ),
        file_mediapipe_model=os.path.abspath(
            os.path.join(base_dir, "blaze_face_full_range.tflite")
        ),
        # YOLO configs
        face_detector=args.face_detector,
        yolo_size=args.yolo_size,
        url_yolo_model=f"https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov{args.yolo_size}.pt",
        file_yolo_model=os.path.abspath(
            os.path.join(base_dir, f"face_yolov{args.yolo_size}.pt")
        ),
        # API keys (from env)
        api_key_gemini=os.environ.get("GOOGLE_API_KEY", ""),
        hf_token=os.environ.get("HF_TOKEN", ""),
        pexels_api_key=os.environ.get("PEXELS_API_KEY", ""),
        # Pengaturan utama
        url_youtube=args.url,
        jumlah_clip=args.clips,
        pilihan_rasio=args.ratio,
        download_source_height=args.source_height,
        render_output_height=args.render_height,
        # Konten & Hook
        max_kata_per_subtitle=args.words_per_sub,
        durasi_hook=args.hook_duration,
        hook_source=args.hook_source,
        hook_source_start=args.hook_source_start,
        use_broll=not args.no_broll,
        use_hook_glitch=not args.no_hook,
        use_auto_bgm=not args.no_bgm,
        use_karaoke_effect=not args.no_karaoke,
        use_split_screen=args.split_screen,
        use_dynamic_split=args.dynamic_split,
        split_trigger=args.split_trigger,
        use_camera_switch=args.camera_switch,
        diarization_num_speakers=args.diarization_speakers,
        switch_hold_duration=args.switch_hold_duration,
        split_zoom=args.split_zoom,
        split_v_align=args.split_v_align,
        split_auto_zoom=args.split_auto_zoom,
        split_max_zoom=args.split_max_zoom,
        # Subtitle & Tipografi
        no_subs=args.no_subs,
        gaya_font_aktif=args.font_style,
        daftar_font=DAFTAR_FONT,
        use_advanced_text=args.advanced_text,
        use_advanced_text_on_hook=args.advanced_text_hook,
        # ASS position values
        ass_align_916=ASS_ALIGN_916,
        ass_margin_916=ASS_MARGIN_916,
        ass_font_916=ASS_FONT_916,
        scale_kata_khusus_916=SCALE_KATA_KHUSUS_916,
        ass_align_169=ASS_ALIGN_169,
        ass_margin_169=ASS_MARGIN_169,
        ass_font_169=ASS_FONT_169,
        scale_kata_khusus_169=SCALE_KATA_KHUSUS_169,
        warna_kata_khusus=WARNA_KATA_KHUSUS,
        # Asset URLs
        url_font_thumbnail=URL_FONT_THUMBNAIL,
        url_glitch_video=URL_GLITCH_VIDEO,
        url_mediapipe_model=URL_MEDIAPIPE_MODEL,
        # BGM
        bgm_base_volume=BGM_BASE_VOLUME,
        bgm_pool=BGM_POOL,
        # Whisper
        use_dlp_subs=args.use_dlp_subs,
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        whisper_compute_type=args.whisper_compute_type,
        # AI
        ai_provider=args.ai_provider,
        api_key_nvidia=os.environ.get("NVIDIA_API_KEY", ""),
        nvidia_model=args.nvidia_model,
        gemini_model=args.gemini_model,
        gemini_fallback_model=args.gemini_fallback_model,
        load_gemini_json=args.load_gemini_json,
        # Tracking Tuning
        track_step=args.track_step,
        track_deadzone=args.track_deadzone,
        track_smooth=args.track_smooth,
        track_jitter=args.track_jitter,
        track_snap=args.track_snap,
        track_conf=args.track_conf,
        track_smooth_window=args.track_smooth_window,
        scene_cut_threshold=args.scene_cut_threshold,
        track_iou_threshold=args.track_iou_threshold,
        video_quality_cq=args.video_cq,
        video_quality_crf=args.video_crf,
        video_bitrate=args.video_bitrate,
        video_sharpen=args.video_sharpen,
        video_preset=args.video_preset,
        video_scale_algo=args.video_scale_algo,
        box_face_detection=args.box_face_detection,
        dev_mode=args.dev_mode,
        dev_mode_with_output=args.dev_mode_with_output,
        dev_mode_with_output_merge=args.dev_mode_with_output_merge,
        track_lines=args.track_lines,
    )

    return cfg
