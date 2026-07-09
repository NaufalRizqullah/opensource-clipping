"""
web.api.config_adapter — Bridge between API JSON payload and the CLI config.

Converts a ``JobCreateRequest`` (or raw dict) into the same
``SimpleNamespace`` object that ``clipping.config.build_config()`` produces,
so the existing pipeline code works unchanged.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

# Import defaults from existing config module
from clipping.config import (
    ASS_ALIGN_169,
    ASS_ALIGN_916,
    ASS_FONT_169,
    ASS_FONT_916,
    ASS_MARGIN_169,
    ASS_MARGIN_916,
    BGM_BASE_VOLUME,
    BGM_POOL,
    DAFTAR_FONT,
    GEMINI_FALLBACK_MODEL,
    NAMA_FONT_THUMBNAIL,
    RENDER_OUTPUT_HEIGHT,
    SCALE_KATA_KHUSUS_169,
    SCALE_KATA_KHUSUS_916,
    URL_FONT_THUMBNAIL,
    URL_GLITCH_VIDEO,
    URL_MEDIAPIPE_MODEL,
    VIDEO_PRESET,
    VIDEO_QUALITY_CQ,
    VIDEO_QUALITY_CRF,
    VIDEO_SCALE_ALGO,
    WARNA_KATA_KHUSUS,
)


def build_config_from_payload(
    payload: dict,
    job_id: str,
    *,
    env_overrides: dict | None = None,
) -> SimpleNamespace:
    """
    Convert an API request payload into a ``SimpleNamespace`` config
    compatible with the existing clipping pipeline.

    Parameters
    ----------
    payload : dict
        The job creation payload (from ``JobCreateRequest.model_dump()``).
    job_id : str
        Unique job identifier — used for per-job output directory.
    env_overrides : dict, optional
        Runtime overrides for API keys (e.g. from stored settings).

    Returns
    -------
    SimpleNamespace
        Fully populated config ready for ``run_pipeline(cfg)``.
    """
    env = env_overrides or {}

    base_dir = os.getcwd()
    outputs_dir = os.path.abspath(os.path.join(base_dir, "outputs", job_id))
    os.makedirs(outputs_dir, exist_ok=True)
    font_dir = os.path.abspath(os.path.join(base_dir, "custom_fonts"))
    os.makedirs(font_dir, exist_ok=True)

    # Resolve source platform
    source_platform = payload.get("source", "youtube")

    # Resolve face detector model
    face_detector = payload.get("face_detector", "mediapipe")
    yolo_size = payload.get("yolo_size", "8m")

    # Resolve AI provider
    ai_provider = payload.get("ai_provider", "gemini")

    # Resolve render height
    render_height = payload.get("render_height", str(RENDER_OUTPUT_HEIGHT))

    # Resolve source height
    source_height = payload.get("source_height", "max")
    if source_height != "max":
        try:
            source_height = int(source_height)
        except (ValueError, TypeError):
            source_height = "max"

    # Determine video input path
    upload_filename = payload.get("upload_filename")
    if upload_filename:
        file_video_asli = os.path.abspath(
            os.path.join(base_dir, "uploads", upload_filename)
        )
    else:
        file_video_asli = os.path.abspath(
            os.path.join(outputs_dir, "video_asli.mp4")
        )

    cfg = SimpleNamespace(
        # Paths
        base_dir=base_dir,
        outputs_dir=outputs_dir,
        font_dir=font_dir,
        file_video_asli=file_video_asli,
        file_font_thumbnail=os.path.abspath(
            os.path.join(base_dir, NAMA_FONT_THUMBNAIL)
        ),
        file_mediapipe_model=os.path.abspath(
            os.path.join(base_dir, "blaze_face_full_range.tflite")
        ),
        # YOLO configs
        face_detector=face_detector,
        yolo_size=yolo_size,
        url_yolo_model=f"https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov{yolo_size}.pt",
        file_yolo_model=os.path.abspath(
            os.path.join(base_dir, f"face_yolov{yolo_size}.pt")
        ),
        # API keys — prefer env_overrides, then os.environ
        api_key_gemini=env.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", "")),
        hf_token=env.get("HF_TOKEN", os.environ.get("HF_TOKEN", "")),
        pexels_api_key=env.get("PEXELS_API_KEY", os.environ.get("PEXELS_API_KEY", "")),
        # Pengaturan utama
        source_platform=source_platform,
        url_youtube=payload.get("url"),
        jumlah_clip=payload.get("clips", 7),
        pilihan_rasio=payload.get("ratio", "9:16"),
        download_source_height=source_height,
        render_output_height=render_height,
        # Konten & Hook
        max_kata_per_subtitle=payload.get("words_per_sub", 5),
        durasi_hook=payload.get("hook_duration", 3),
        hook_source=None,
        hook_source_start=0.0,
        # Hook V2 & Segment Trimming
        hook_v2=payload.get("hook_v2", False),
        hook_v2_items=payload.get("hook_v2_items", 3),
        hook_v2_style=payload.get("hook_v2_style", "controversial_fast_glitch"),
        white_flash_duration=payload.get("white_flash_duration", 0.12),
        no_segment_trim=payload.get("no_segment_trim", False),
        silence_trim=payload.get("silence_trim", False),
        use_broll=payload.get("use_broll", True),
        use_hook_glitch=payload.get("use_hook_glitch", True),
        use_auto_bgm=payload.get("use_auto_bgm", True),
        use_karaoke_effect=payload.get("use_karaoke_effect", True),
        use_split_screen=payload.get("use_split_screen", False),
        use_dynamic_split=payload.get("use_dynamic_split", False),
        split_trigger=payload.get("split_trigger", "diarization"),
        use_camera_switch=payload.get("use_camera_switch", False),
        diarization_num_speakers=payload.get("diarization_speakers", "auto"),
        switch_hold_duration=payload.get("switch_hold_duration", 2.0),
        switch_blend_duration=payload.get("switch_blend_duration", 0.0),
        split_zoom=payload.get("split_zoom", 1.0),
        split_v_align=payload.get("split_v_align", 0.5),
        split_auto_zoom=payload.get("split_auto_zoom", False),
        split_max_zoom=payload.get("split_max_zoom", 2.5),
        # Subtitle & Tipografi
        no_subs=payload.get("no_subs", False),
        gaya_font_aktif=payload.get("font_style", "HORMOZI"),
        daftar_font=DAFTAR_FONT,
        use_advanced_text=payload.get("advanced_text", False),
        use_advanced_text_on_hook=payload.get("advanced_text_hook", False),
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
        use_dlp_subs=payload.get("use_dlp_subs", False),
        whisper_model=payload.get("whisper_model", "large-v3"),
        whisper_device=payload.get("whisper_device", "cuda"),
        whisper_compute_type=payload.get("whisper_compute_type", "float16"),
        # AI
        ai_provider=ai_provider,
        api_key_nvidia=env.get("NVIDIA_API_KEY", os.environ.get("NVIDIA_API_KEY", "")),
        nvidia_model=payload.get("nvidia_model", "deepseek-ai/deepseek-v4-pro"),
        gemini_model=payload.get("gemini_model", "gemini-3-flash-preview"),
        gemini_fallback_model=payload.get("gemini_fallback_model", GEMINI_FALLBACK_MODEL),
        load_gemini_json=payload.get("load_gemini_json", False),
        # Tracking Tuning (use defaults for web GUI)
        track_step=None,
        track_deadzone=None,
        track_smooth=None,
        track_jitter=None,
        track_snap=None,
        track_conf=0.55,
        track_smooth_window=12,
        scene_cut_threshold=18,
        track_iou_threshold=0.2,
        video_quality_cq=payload.get("video_cq", VIDEO_QUALITY_CQ),
        video_quality_crf=payload.get("video_crf", VIDEO_QUALITY_CRF),
        video_bitrate=payload.get("video_bitrate", "auto"),
        video_sharpen=payload.get("video_sharpen", False),
        video_preset=payload.get("video_preset", VIDEO_PRESET),
        video_scale_algo=payload.get("video_scale_algo", VIDEO_SCALE_ALGO),
        box_face_detection=False,
        dev_mode=False,
        dev_mode_with_output=False,
        dev_mode_with_output_merge=False,
        track_lines=False,
        static_crop=payload.get("static_crop", False),
        # Story Clip Mode (not supported via web yet)
        story_mode=False,
        story_recipe_path=None,
        sources_json_path=None,
        story_output_dir=None,
        skip_download=False,
    )

    return cfg
