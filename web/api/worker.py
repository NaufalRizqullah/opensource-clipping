"""
web.api.worker — Background task runner for the clipping pipeline.

Wraps ``clipping.runner.run_pipeline()`` in an asyncio task with
progress reporting via the job store.
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

from .config_adapter import build_config_from_payload
from .models import ClipDetail, JobStatus
from . import store

# The shared renderer uses process-wide FFmpeg scratch names and settings.
# Keep API jobs serial until the renderer itself supports concurrent execution.
MAX_CONCURRENT_JOBS = 1
if os.environ.get("MAX_CONCURRENT_JOBS", "1") != "1":
    logging.getLogger(__name__).warning(
        "API jobs run one at a time because the renderer shares temporary files."
    )
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)
_tasks: dict[str, asyncio.Task] = {}
_running_jobs: set[str] = set()

# Store settings overrides (API keys etc.) in memory
_settings_env: dict[str, str] = {}


def set_settings_env(env: dict[str, str]) -> None:
    """Update runtime settings environment."""
    global _settings_env
    _settings_env.update(env)


def get_settings_env() -> dict[str, str]:
    """Get current settings environment."""
    return dict(_settings_env)


class JobCancelled(Exception):
    """Internal signal raised at safe pipeline boundaries."""


def _check_cancelled(job_id: str) -> None:
    if store.is_cancel_requested(job_id):
        raise JobCancelled()


def _validate_analysis_config(cfg) -> None:
    """Validate only the provider that will actually be called."""
    if cfg.load_gemini_json:
        cached = Path(cfg.outputs_dir) / "gemini_response.json"
        if not cached.is_file():
            raise ValueError("Cached AI JSON tidak ditemukan. Nonaktifkan Bypass AI untuk analisis baru.")
        return
    if cfg.ai_provider == "nvidia" and cfg.api_key_nvidia:
        return
    if not cfg.api_key_gemini:
        key_name = "NVIDIA_API_KEY atau GOOGLE_API_KEY" if cfg.ai_provider == "nvidia" else "GOOGLE_API_KEY"
        raise ValueError(f"{key_name} tidak ditemukan. Set via Settings atau environment.")


def is_job_active(job_id: str) -> bool:
    """Include the small interval between terminal state and thread completion."""
    task = _tasks.get(job_id)
    return job_id in _running_jobs or (task is not None and not task.done())


def cancel_job(job_id: str) -> bool:
    """Cancel a queued task, or request cancellation of a running thread."""
    if not store.request_cancel(job_id):
        return False
    if job_id not in _running_jobs:
        task = _tasks.get(job_id)
        if task is not None:
            task.cancel()
        store.finish_job(job_id)
    return True


async def cancel_all_jobs() -> None:
    """Request shutdown without pretending active native operations stopped."""
    for job_id in list(_tasks):
        cancel_job(job_id)
    # Let cancelled queued tasks release their semaphore waiters/callbacks.
    await asyncio.sleep(0)


def _clips_from_manifest(job_id: str, manifest: list[dict]) -> list[ClipDetail]:
    """Expose only successful, existing renderer outputs to the Studio."""
    clips = []
    for entry in manifest:
        if entry.get("status") != "success" or not entry.get("video_exists", False):
            continue
        try:
            filename = os.path.basename(entry.get("output_file") or entry.get("video_path") or "")
            if not filename:
                continue
            clips.append(ClipDetail(
                rank=entry.get("rank", 0), viral_score=entry.get("viral_score"),
                title=entry.get("title_indonesia", ""), title_en=entry.get("title_inggris", ""),
                filename=filename, duration=entry.get("duration"),
                start_time=entry.get("start_time"), end_time=entry.get("end_time"),
                download_url=f"/api/outputs/{quote(job_id, safe='')}/{quote(filename, safe='')}",
                metadata=entry,
            ))
        except (ValueError, TypeError) as exc:
            # This helper also runs on the error path. Invalid metadata must
            # never prevent the worker from publishing its final failed state.
            logging.getLogger(__name__).warning("Invalid renderer metadata (%s)", type(exc).__name__)
    return clips


def _run_pipeline_sync(job_id: str, payload: dict) -> None:
    """
    Run the clipping pipeline synchronously (called from thread pool).

    This function updates the job store at each pipeline step so the
    frontend can poll or receive SSE progress updates.
    """
    render_manifest: list[dict] = []
    try:
        _check_cancelled(job_id)
        # Build config from API payload
        cfg = build_config_from_payload(
            payload, job_id, env_overrides=_settings_env
        )

        _validate_analysis_config(cfg)

        # --- Step 1: Download ---
        store.set_status(job_id, JobStatus.DOWNLOADING)
        store.update_progress(
            job_id,
            step="download",
            step_number=1,
            total_steps=7,
            message="Mengunduh video...",
            percent=5.0,
        )

        from clipping import engine

        source_platform = getattr(cfg, "source_platform", "youtube")

        # If upload file, skip download
        if payload.get("upload_filename"):
            upload_path = cfg.file_video_asli
            if not os.path.exists(upload_path):
                raise ValueError("File upload tidak ditemukan.")
            cfg.file_video_asli = upload_path
            store.update_progress(
                job_id,
                step="download",
                step_number=1,
                total_steps=7,
                message="Menggunakan file upload.",
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
                        message="Bypass download: menggunakan video lama.",
                        percent=14.0,
                    )
                else:
                    raise ValueError("Video asli tidak ditemukan di Job ID tersebut. File mungkin sudah terhapus.")
            else:
                engine.download_video(
                    cfg.url_youtube,
                    cfg.file_video_asli,
                    getattr(cfg, "use_dlp_subs", False),
                    getattr(cfg, "download_source_height", "max"),
                    source_platform=source_platform,
                    yt_cookies=getattr(cfg, "yt_cookies", None),
                )
                store.update_progress(
                    job_id,
                    step="download",
                    step_number=1,
                    total_steps=7,
                    message="Video berhasil diunduh.",
                    percent=14.0,
                )

        _check_cancelled(job_id)
        # --- Step 2: Transcribe ---
        store.set_status(job_id, JobStatus.TRANSCRIBING)
        store.update_progress(
            job_id,
            step="transcribe",
            step_number=2,
            total_steps=7,
            message="Memulai transkripsi...",
            percent=15.0,
        )

        transkrip_lengkap = ""
        data_segmen = []

        # Try YouTube JSON3 subs first
        json3_files = glob.glob(str(Path(cfg.file_video_asli).with_suffix("")) + ".*.json3")
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
            message="Transkripsi selesai.",
            percent=35.0,
        )

        _check_cancelled(job_id)
        # --- Step 3: AI Analysis ---
        store.set_status(job_id, JobStatus.ANALYZING)
        store.update_progress(
            job_id,
            step="analyze",
            step_number=3,
            total_steps=7,
            message="Menganalisis dengan AI... Provider dapat melakukan retry otomatis; tahap ini bisa memerlukan beberapa menit.",
            percent=36.0,
        )

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
            message=f"AI menemukan {len(hasil_json)} klip viral.",
            percent=50.0,
        )

        _check_cancelled(job_id)
        # --- Step 4: Metadata ---
        from clipping import metadata

        hasil_json = metadata.normalize_and_validate(hasil_json)
        if not hasil_json:
            raise ValueError("AI tidak menghasilkan klip yang valid untuk dirender.")
        metadata_path = os.path.join(cfg.outputs_dir, "metadata_preview.json")
        metadata.save_metadata_preview(hasil_json, path=metadata_path)

        store.update_progress(
            job_id,
            step="metadata",
            step_number=4,
            total_steps=7,
            message="Metadata dinormalisasi.",
            percent=55.0,
        )

        _check_cancelled(job_id)
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
                    message="Menjalankan speaker diarization...",
                    percent=56.0,
                )
                audio_path = str(Path(cfg.outputs_dir) / "diarization_audio.wav")
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
                    message=f"Diarization gagal: {e}. Fallback ke mode biasa.",
                    percent=58.0,
                )
                diarization_data = None

        _check_cancelled(job_id)
        # --- Step 6: Render Preparation ---
        store.set_status(job_id, JobStatus.RENDERING)
        store.update_progress(
            job_id,
            step="render",
            step_number=6,
            total_steps=7,
            message="Menyiapkan rendering...",
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

        total_clips = len(hasil_json)

        custom_hook_path = None
        if getattr(cfg, "hook_source", None):
            custom_hook_path = hook_manager.download_custom_hook(cfg)

        for idx, klip in enumerate(sorted(hasil_json, key=lambda x: x["rank"])):
            _check_cancelled(job_id)
            clip_num = idx + 1
            store.update_progress(
                job_id,
                step="render",
                step_number=6,
                total_steps=7,
                message=f"Merender klip {clip_num}/{total_clips}...",
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
            if not isinstance(hasil_render, dict):
                hasil_render = {"rank": klip["rank"], "status": "failed", "error": "Renderer returned no result"}
            render_manifest.append(hasil_render)
            # Keep completed work visible even if cancellation happens next.
            manifest_path = os.path.join(cfg.outputs_dir, "render_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(render_manifest, f, ensure_ascii=False, indent=2)
            _check_cancelled(job_id)

        # --- Save manifest ---
        manifest_path = os.path.join(cfg.outputs_dir, "render_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(render_manifest, f, ensure_ascii=False, indent=2)

        # --- Build clip details for the job store ---
        clips = _clips_from_manifest(job_id, render_manifest)

        failed_count = total_clips - len(clips)
        error = None
        if failed_count:
            error = f"{failed_count}/{total_clips} klip gagal dirender. Lihat render_manifest.json untuk detail."
        store.finish_job(job_id, clips=clips, error=error)

    except JobCancelled:
        store.finish_job(job_id, clips=_clips_from_manifest(job_id, render_manifest))
    except Exception as exc:
        tb = traceback.format_exc()
        error_msg = f"{type(exc).__name__}: {exc}"
        store.finish_job(job_id, clips=_clips_from_manifest(job_id, render_manifest), error=error_msg)
        print(f"[Worker] Job {job_id} failed:\n{tb}", file=sys.stderr)


async def submit_job(job_id: str, payload: dict) -> None:
    """
    Submit a job to the background worker queue.

    Uses a semaphore to limit concurrency and runs the pipeline
    in a thread pool to avoid blocking the async event loop.
    """
    if is_job_active(job_id):
        raise ValueError("Job is already running")

    async def _run():
        try:
            async with _semaphore:
                _running_jobs.add(job_id)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    _executor, _run_pipeline_sync, job_id, payload
                )
        finally:
            _running_jobs.discard(job_id)

    task = asyncio.create_task(_run())
    _tasks[job_id] = task

    def _forget(done_task):
        if _tasks.get(job_id) is done_task:
            _tasks.pop(job_id, None)

    task.add_done_callback(_forget)
