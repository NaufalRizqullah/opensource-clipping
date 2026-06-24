"""
web.api.worker - Background task runner for the clipping pipeline.

Wraps ``clipping.runner.run_pipeline()`` in an asyncio task with
progress reporting via the job store.
"""

from __future__ import annotations

import asyncio
import glob
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .config_adapter import build_config_from_payload
from .models import ClipDetail, JobStatus
from . import store

# Semaphore to control max concurrent jobs
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "1"))
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)

# Store settings overrides (API keys etc.) in memory
_settings_env: dict[str, str] = {}


def set_settings_env(env: dict[str, str]) -> None:
    """Update runtime settings environment."""
    global _settings_env
    _settings_env.update(env)


def get_settings_env() -> dict[str, str]:
    """Get current settings environment."""
    return dict(_settings_env)


def _run_pipeline_sync(job_id: str, payload: dict) -> None:
    """
    Run the clipping pipeline synchronously (called from thread pool).

    This function updates the job store at each pipeline step so the
    frontend can poll or receive SSE progress updates.
    """
    try:
        # Build config from API payload
        cfg = build_config_from_payload(
            payload, job_id, env_overrides=_settings_env
        )

        # Validate API key
        if not cfg.api_key_gemini:
            store.set_error(
                job_id,
                "GOOGLE_API_KEY not found. Set it via Settings or a .env file.",
            )
            return

        # --- Step 1: Download ---
        store.set_status(job_id, JobStatus.DOWNLOADING)
        store.update_progress(
            job_id,
            step="download",
            step_number=1,
            total_steps=7,
            message="Downloading video...",
            percent=5.0,
        )

        from clipping import engine

        source_platform = getattr(cfg, "source_platform", "youtube")

        # If upload file, skip download
        if payload.get("upload_filename"):
            upload_path = os.path.join(os.getcwd(), "uploads", payload["upload_filename"])
            if not os.path.exists(upload_path):
                store.set_error(job_id, f"Uploaded file not found: {payload['upload_filename']}")
                return
            cfg.file_video_asli = upload_path
            store.update_progress(
                job_id,
                step="download",
                step_number=1,
                total_steps=7,
                message="Using uploaded file.",
                percent=14.0,
            )
        else:
            if not cfg.url_youtube:
                if os.path.exists(cfg.file_video_asli):
                    store.update_progress(
                        job_id,
                        step="download",
                        step_number=1,
                        total_steps=7,
                        message="Skipping download: using existing video.",
                        percent=14.0,
                    )
                else:
                    store.set_error(job_id, "Original video not found for that Job ID. The file may have been deleted.")
                    return
            else:
                engine.download_video(
                    cfg.url_youtube,
                    cfg.file_video_asli,
                    getattr(cfg, "use_dlp_subs", False),
                    getattr(cfg, "download_source_height", "max"),
                    source_platform=source_platform,
                )
                store.update_progress(
                    job_id,
                    step="download",
                    step_number=1,
                    total_steps=7,
                    message="Video downloaded successfully.",
                    percent=14.0,
                )

        # --- Step 2: Transcribe ---
        store.set_status(job_id, JobStatus.TRANSCRIBING)
        store.update_progress(
            job_id,
            step="transcribe",
            step_number=2,
            total_steps=7,
            message="Starting transcription...",
            percent=15.0,
        )

        transkrip_lengkap = ""
        data_segmen = []

        # Try YouTube JSON3 subs first
        json3_files = glob.glob(cfg.file_video_asli.replace(".mp4", ".*.json3"))
        file_json3 = json3_files[0] if json3_files else None

        if source_platform == "youtube" and getattr(cfg, "use_dlp_subs", False) and file_json3 and os.path.exists(file_json3):
            transkrip_lengkap, data_segmen = engine.parse_youtube_json3_subs(
                file_json3, max_words_per_subtitle=cfg.max_kata_per_subtitle
            )

        if not transkrip_lengkap or not data_segmen:
            transkrip_lengkap, data_segmen = engine.transcribe_video(
                cfg.file_video_asli,
                max_words_per_subtitle=cfg.max_kata_per_subtitle,
                model_size=cfg.whisper_model,
                device=cfg.whisper_device,
                compute_type=cfg.whisper_compute_type,
            )

        store.update_progress(
            job_id,
            step="transcribe",
            step_number=2,
            total_steps=7,
            message="Transcription complete.",
            percent=35.0,
        )

        # --- Step 3: AI Analysis ---
        store.set_status(job_id, JobStatus.ANALYZING)
        store.update_progress(
            job_id,
            step="analyze",
            step_number=3,
            total_steps=7,
            message="Analyzing with AI...",
            percent=36.0,
        )

        import json

        gemini_output_path = os.path.join(cfg.outputs_dir, "gemini_response.json")

        if getattr(cfg, "load_gemini_json", False) and os.path.exists(gemini_output_path):
            with open(gemini_output_path, "r", encoding="utf-8") as f:
                hasil_json = json.load(f)
        else:
            hasil_json = engine.analyze_with_ai(transkrip_lengkap, cfg)
            with open(gemini_output_path, "w", encoding="utf-8") as f:
                json.dump(hasil_json, f, indent=4, ensure_ascii=False)

        store.update_progress(
            job_id,
            step="analyze",
            step_number=3,
            total_steps=7,
            message=f"AI found {len(hasil_json)} viral clips.",
            percent=50.0,
        )

        # --- Step 4: Metadata ---
        from clipping import metadata

        hasil_json = metadata.normalize_and_validate(hasil_json)
        metadata_path = os.path.join(cfg.outputs_dir, "metadata_preview.json")
        metadata.save_metadata_preview(hasil_json, path=metadata_path)

        store.update_progress(
            job_id,
            step="metadata",
            step_number=4,
            total_steps=7,
            message="Metadata normalized.",
            percent=55.0,
        )

        # --- Step 5: Diarization (optional) ---
        diarization_data = None
        from clipping import studio, diarization as diarization_mod

        if (
            (getattr(cfg, "use_split_screen", False) and cfg.split_trigger == "diarization")
            or getattr(cfg, "use_camera_switch", False)
        ) and studio._is_vertical_ratio(cfg.pilihan_rasio):
            try:
                store.update_progress(
                    job_id,
                    step="diarization",
                    step_number=5,
                    total_steps=7,
                    message="Running speaker diarization...",
                    percent=56.0,
                )
                audio_path = cfg.file_video_asli.replace(".mp4", "_audio.wav")
                diarization_mod.extract_audio(cfg.file_video_asli, audio_path)
                num_speakers_arg = getattr(cfg, "diarization_num_speakers", 2)
                min_spk = None
                max_spk = None

                if str(num_speakers_arg).lower() == "auto":
                    max_faces = studio.estimate_speaker_count_from_video(cfg.file_video_asli, cfg)
                    num_speakers_arg = "auto"
                    min_spk = max(1, max_faces)
                    max_spk = min_spk + 2

                diarization_data = diarization_mod.run_diarization(
                    audio_path,
                    hf_token=cfg.hf_token,
                    num_speakers=num_speakers_arg,
                    min_speakers=min_spk,
                    max_speakers=max_spk,
                )
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                store.update_progress(
                    job_id,
                    step="diarization",
                    step_number=5,
                    total_steps=7,
                    message=f"Diarization failed: {e}. Falling back to normal mode.",
                    percent=58.0,
                )
                diarization_data = None

        # --- Step 6: Render Preparation ---
        store.set_status(job_id, JobStatus.RENDERING)
        store.update_progress(
            job_id,
            step="render",
            step_number=6,
            total_steps=7,
            message="Preparing rendering...",
            percent=60.0,
        )

        os.environ["OSC_VIDEO_SCALE_ALGO"] = str(getattr(cfg, "video_scale_algo", "lanczos"))

        import cv2

        cap_e = cv2.VideoCapture(cfg.file_video_asli)
        src_h_e = int(cap_e.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_e.release()

        target_w_e, target_h_e = studio._get_render_dims(cfg, cfg.pilihan_rasio, source_h=src_h_e)
        video_encoder = studio.detect_video_encoder(cfg, target_h=target_h_e)

        file_glitch_ts = None
        if cfg.use_hook_glitch:
            cap_g = cv2.VideoCapture(cfg.file_video_asli)
            source_h_g = int(cap_g.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_g.release()
            file_glitch_ts = studio.siapkan_glitch_video(
                cfg.pilihan_rasio, cfg, video_encoder, source_h=source_h_g
            )

        # --- Step 7: Render Each Clip ---
        from clipping import hook_manager

        render_manifest: list[dict] = []
        total_clips = len(hasil_json)

        custom_hook_path = None
        if getattr(cfg, "hook_source", None):
            custom_hook_path = hook_manager.download_custom_hook(cfg)

        for idx, klip in enumerate(sorted(hasil_json, key=lambda x: x["rank"])):
            clip_num = idx + 1
            store.update_progress(
                job_id,
                step="render",
                step_number=6,
                total_steps=7,
                message=f"Rendering clip {clip_num}/{total_clips}...",
                percent=60.0 + (35.0 * clip_num / total_clips),
            )

            if custom_hook_path:
                klip["custom_hook_info"] = {"file_path": custom_hook_path}

            hasil_render = studio.proses_klip(
                klip["rank"],
                klip,
                cfg.pilihan_rasio,
                file_glitch_ts,
                data_segmen,
                cfg,
                video_encoder,
                diarization_data=diarization_data,
            )
            if hasil_render:
                render_manifest.append(hasil_render)

        # --- Save manifest ---
        manifest_path = os.path.join(cfg.outputs_dir, "render_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(render_manifest, f, ensure_ascii=False, indent=2)

        # --- Build clip details for the job store ---
        clips: list[ClipDetail] = []
        for entry in render_manifest:
            filename = os.path.basename(entry.get("output_file") or entry.get("video_path") or "")
            clips.append(
                ClipDetail(
                    rank=entry.get("rank", 0),
                    viral_score=entry.get("viral_score"),
                    title=entry.get("title_inggris", "") or entry.get("youtube_title_final", ""),
                    title_en=entry.get("title_inggris", ""),
                    filename=filename,
                    duration=entry.get("duration"),
                    start_time=entry.get("start_time"),
                    end_time=entry.get("end_time"),
                    download_url=f"/api/outputs/{job_id}/{filename}",
                    metadata=entry,
                )
            )

        store.set_clips(job_id, clips)
        store.update_progress(
            job_id,
            step="done",
            step_number=7,
            total_steps=7,
            message=f"Done! {len(clips)} clips rendered successfully.",
            percent=100.0,
        )

    except Exception as exc:
        tb = traceback.format_exc()
        error_msg = f"{type(exc).__name__}: {exc}"
        store.set_error(job_id, error_msg)
        store.update_progress(
            job_id,
            step="error",
            step_number=0,
            total_steps=7,
            message=f"Pipeline failed: {error_msg}",
            percent=0.0,
        )
        print(f"[Worker] Job {job_id} failed:\n{tb}", file=sys.stderr)


async def submit_job(job_id: str, payload: dict) -> None:
    """
    Submit a job to the background worker queue.

    Uses a semaphore to limit concurrency and runs the pipeline
    in a thread pool to avoid blocking the async event loop.
    """
    async def _run():
        async with _semaphore:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                _executor, _run_pipeline_sync, job_id, payload
            )

    asyncio.create_task(_run())
