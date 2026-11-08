"""
web.api.routes.settings — Settings management endpoints.
"""

from __future__ import annotations

import os
import asyncio
from functools import lru_cache
import shutil
import subprocess

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..models import SettingsRequest, SettingsResponse, SystemHealthResponse
from .. import store as job_store
from .. import worker
from ..readiness import get_readiness

router = APIRouter(tags=["settings"])


@lru_cache(maxsize=1)
def _check_gpu() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


@router.get("/api/settings")
async def get_settings() -> SettingsResponse:
    """Get current settings (API keys are masked)."""
    env = worker.get_settings_env()

    # Check which keys are set (from worker env or os.environ)
    google_key = env.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
    pexels_key = env.get("PEXELS_API_KEY", os.environ.get("PEXELS_API_KEY", ""))
    hf_token = env.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
    nvidia_key = env.get("NVIDIA_API_KEY", os.environ.get("NVIDIA_API_KEY", ""))

    return SettingsResponse(
        google_api_key_set=bool(google_key),
        pexels_api_key_set=bool(pexels_key),
        hf_token_set=bool(hf_token),
        nvidia_api_key_set=bool(nvidia_key),
        default_clips=int(env.get("DEFAULT_CLIPS", "7")),
        default_ratio=env.get("DEFAULT_RATIO", "9:16"),
        default_font_style=env.get("DEFAULT_FONT_STYLE", "HORMOZI"),
        default_whisper_model=env.get("DEFAULT_WHISPER_MODEL", "large-v3"),
        default_whisper_device=env.get("DEFAULT_WHISPER_DEVICE", "cuda"),
        default_ai_provider=env.get("DEFAULT_AI_PROVIDER", "gemini"),
        gpu_available=await asyncio.to_thread(_check_gpu),
    )


@router.put("/api/settings")
async def update_settings(req: SettingsRequest) -> SettingsResponse:
    """Update settings (API keys and defaults)."""
    env_updates: dict[str, str] = {}

    if req.google_api_key is not None:
        env_updates["GOOGLE_API_KEY"] = req.google_api_key
    if req.pexels_api_key is not None:
        env_updates["PEXELS_API_KEY"] = req.pexels_api_key
    if req.hf_token is not None:
        env_updates["HF_TOKEN"] = req.hf_token
    if req.nvidia_api_key is not None:
        env_updates["NVIDIA_API_KEY"] = req.nvidia_api_key
    if req.default_clips is not None:
        env_updates["DEFAULT_CLIPS"] = str(req.default_clips)
    if req.default_ratio is not None:
        env_updates["DEFAULT_RATIO"] = req.default_ratio.value if hasattr(req.default_ratio, "value") else req.default_ratio
    if req.default_font_style is not None:
        env_updates["DEFAULT_FONT_STYLE"] = req.default_font_style.value if hasattr(req.default_font_style, "value") else req.default_font_style
    if req.default_whisper_model is not None:
        env_updates["DEFAULT_WHISPER_MODEL"] = req.default_whisper_model
    if req.default_whisper_device is not None:
        env_updates["DEFAULT_WHISPER_DEVICE"] = req.default_whisper_device.value if hasattr(req.default_whisper_device, "value") else req.default_whisper_device
    if req.default_ai_provider is not None:
        env_updates["DEFAULT_AI_PROVIDER"] = req.default_ai_provider.value if hasattr(req.default_ai_provider, "value") else req.default_ai_provider

    worker.set_settings_env(env_updates)

    return await get_settings()


@router.get("/api/health")
async def health_check() -> SystemHealthResponse:
    """API liveness and system information; use /api/ready for pipeline readiness."""
    gpu_available, ffmpeg_available = await asyncio.gather(
        asyncio.to_thread(_check_gpu), asyncio.to_thread(_check_ffmpeg),
    )
    return SystemHealthResponse(
        status="ok",
        version="1.12.0",
        gpu_available=gpu_available,
        ffmpeg_available=ffmpeg_available,
        jobs_running=job_store.get_running_count(),
        jobs_queued=job_store.get_queued_count(),
    )


@router.get("/api/ready")
async def readiness_check() -> JSONResponse:
    """Fail before starting paid AI work when native renderer imports are broken."""
    result = await asyncio.to_thread(get_readiness)
    return JSONResponse(result, status_code=200 if result["ready"] else 503)
