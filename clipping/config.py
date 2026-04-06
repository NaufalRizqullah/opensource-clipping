"""
clipping.config — Master Configuration (Dashboard)

Menyimpan semua default value dan membangun config dari CLI args.
"""

import os
import argparse
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
URL_YOUTUBE     = "https://www.youtube.com/watch?v=Ip6pCjWp4lk"
JUMLAH_CLIP     = 7
PILIHAN_RASIO   = "9:16"

# 2. PENGATURAN KONTEN & HOOK
MAX_KATA_PER_SUBTITLE = 5
DURASI_HOOK           = 3
USE_BROLL             = True
USE_HOOK_GLITCH       = True
USE_SPLIT_SCREEN      = False
DIARIZATION_NUM_SPEAKERS = 2

# 3. PENGATURAN SUBTITLE & TIPOGRAFI (ASS STYLE)
USE_ADVANCED_TEXT          = False
USE_ADVANCED_TEXT_ON_HOOK  = False
USE_KARAOKE_EFFECT         = True

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
ASS_ALIGN_916   = 2
ASS_MARGIN_916  = 450
ASS_FONT_916    = 90
SCALE_KATA_KHUSUS_916 = ASS_FONT_916 + 120

# Khusus 16:9 (Horizontal)
ASS_ALIGN_169   = 2
ASS_MARGIN_169  = 70
ASS_FONT_169    = 80
SCALE_KATA_KHUSUS_169 = ASS_FONT_169 + 120

# Warna Kata Khusus  (Format ASS: BGR -> &H[Blue][Green][Red]&)
WARNA_KATA_KHUSUS = "&HFFFFFF&"

# 4. PENGATURAN ASSET EKSTERNAL
NAMA_FONT_THUMBNAIL = "Montserrat-Black.ttf"
URL_FONT_THUMBNAIL  = "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Black.ttf"

URL_GLITCH_VIDEO    = "https://www.youtube.com/watch?v=5nBcNRYmjs0"
URL_MEDIAPIPE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"

# 5. PENGATURAN Auto-BGM & Audio Ducking
USE_AUTO_BGM      = True
BGM_BASE_VOLUME   = 0.25

BGM_POOL = {
    "chill":    "https://pixabay.com/music/beats-lofi-study-calm-peaceful-chill-hop-112191/",
    "epic":     "https://pixabay.com/music/main-title-epic-epic-trailer-music-325357/",
    "sad":      "https://pixabay.com/music/modern-classical-sad-sorrowful-piano-music-330768/",
    "upbeat":   "https://pixabay.com/music/upbeat-pop-upbeat-pop-music-307007/",
    "suspense": "https://pixabay.com/music/ambient-dark-ambient-background-music-342437/",
}

# Whisper
WHISPER_MODEL  = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

# Gemini
GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash"


# ==============================================================================
# CLI PARSER
# ==============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="🎬 OpenSource Clipping — AI Auto-Clipper & Teaser Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Pengaturan utama ---
    p.add_argument("--url", "-u", default=URL_YOUTUBE,
                    help="YouTube video URL to process")
    p.add_argument("--clips", "-n", type=int, default=JUMLAH_CLIP,
                    help="Number of highlight clips to generate")
    p.add_argument("--ratio", "-r", default=PILIHAN_RASIO,
                    choices=["9:16", "16:9"],
                    help="Output aspect ratio")

    # --- Konten & Hook ---
    p.add_argument("--words-per-sub", type=int, default=MAX_KATA_PER_SUBTITLE,
                    help="Max words per karaoke subtitle group")
    p.add_argument("--hook-duration", type=int, default=DURASI_HOOK,
                    help="Hook teaser duration in seconds")
    p.add_argument("--no-broll", action="store_true",
                    help="Disable B-roll footage")
    p.add_argument("--no-hook", action="store_true",
                    help="Disable hook glitch teaser")
    p.add_argument("--no-bgm", action="store_true",
                    help="Disable background music")
    p.add_argument("--no-karaoke", action="store_true",
                    help="Disable karaoke highlight effect (use clean text instead)")
    p.add_argument("--split-screen", action="store_true", default=USE_SPLIT_SCREEN,
                    help="Enable split-screen mode for podcast with 2 speakers (9:16 only, requires HF_TOKEN for Pyannote)")
    p.add_argument("--diarization-speakers", type=int, default=DIARIZATION_NUM_SPEAKERS,
                    help="Number of speakers for diarization (used with --split-screen)")

    # --- Subtitle & Tipografi ---
    p.add_argument("--font-style", default=GAYA_FONT_AKTIF,
                    choices=["DEFAULT", "STORYTELLER", "HORMOZI", "CINEMATIC"],
                    help="Font style preset")
    p.add_argument("--advanced-text", action="store_true", default=USE_ADVANCED_TEXT,
                    help="Enable advanced kinetic typography")
    p.add_argument("--advanced-text-hook", action="store_true", default=USE_ADVANCED_TEXT_ON_HOOK,
                    help="Enable advanced typography on hook")

    # --- Whisper ---
    p.add_argument("--use-dlp-subs", action="store_true",
                    help="Use yt-dlp to download auto/manual subtitles to speed up process (skipping Whisper if found)")
    p.add_argument("--whisper-model", default=WHISPER_MODEL,
                    help="Faster-Whisper model size")
    p.add_argument("--whisper-device", default=WHISPER_DEVICE,
                    choices=["cuda", "cpu", "auto"],
                    help="Device for Whisper inference")
    p.add_argument("--whisper-compute-type", default=WHISPER_COMPUTE_TYPE,
                    help="Compute type for Whisper (float16, int8, etc.)")

    # --- Gemini ---
    p.add_argument("--face-detector", choices=["mediapipe", "yolo"], default="mediapipe",
                   help="AI model for face tracking (mediapipe is CPU, yolo uses GPU if available)")
    p.add_argument("--yolo-size", choices=["8n", "8s", "8m", "8n_v2", "9c"], default="8m",
                   help="YOLO face model version/size (8n, 8s, 8m, 8n_v2, 9c). Only active if --face-detector yolo")
    p.add_argument("--gemini-model", default=GEMINI_MODEL,
                    help="Gemini model name")
    p.add_argument("--gemini-fallback-model", default=GEMINI_FALLBACK_MODEL,
                    help="Gemini fallback model name if main model fails")

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
        file_font_thumbnail=os.path.abspath(os.path.join(base_dir, NAMA_FONT_THUMBNAIL)),
        file_mediapipe_model=os.path.abspath(os.path.join(base_dir, "blaze_face_full_range.tflite")),
        
        # YOLO configs
        face_detector=args.face_detector,
        yolo_size=args.yolo_size,
        url_yolo_model=f"https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov{args.yolo_size}.pt",
        file_yolo_model=os.path.abspath(os.path.join(base_dir, f"face_yolov{args.yolo_size}.pt")),

        # API keys (from env)
        api_key_gemini=os.environ.get("GOOGLE_API_KEY", ""),
        hf_token=os.environ.get("HF_TOKEN", ""),
        pexels_api_key=os.environ.get("PEXELS_API_KEY", ""),

        # Pengaturan utama
        url_youtube=args.url,
        jumlah_clip=args.clips,
        pilihan_rasio=args.ratio,

        # Konten & Hook
        max_kata_per_subtitle=args.words_per_sub,
        durasi_hook=args.hook_duration,
        use_broll=not args.no_broll,
        use_hook_glitch=not args.no_hook,
        use_auto_bgm=not args.no_bgm,
        use_karaoke_effect=not args.no_karaoke,
        use_split_screen=args.split_screen,
        diarization_num_speakers=args.diarization_speakers,

        # Subtitle & Tipografi
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

        # Gemini
        gemini_model=args.gemini_model,
        gemini_fallback_model=args.gemini_fallback_model,
    )

    return cfg
