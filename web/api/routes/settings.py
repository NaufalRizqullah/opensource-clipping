"""
web.api.routes.settings - Settings management endpoints.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from fastapi import APIRouter

from ..models import SettingsRequest, SettingsResponse, SystemHealthResponse
from .. import store as job_store
from .. import worker

router = APIRouter(tags=["settings"])


def _check_gpu() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
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
    openai_key = env.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    anthropic_key = env.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))

    return SettingsResponse(
        google_api_key_set=bool(google_key),
        pexels_api_key_set=bool(pexels_key),
        hf_token_set=bool(hf_token),
        nvidia_api_key_set=bool(nvidia_key),
        openai_api_key_set=bool(openai_key),
        anthropic_api_key_set=bool(anthropic_key),
        default_clips=int(env.get("DEFAULT_CLIPS", "7")),
        default_ratio=env.get("DEFAULT_RATIO", "9:16"),
        default_font_style=env.get("DEFAULT_FONT_STYLE", "HORMOZI"),
        default_whisper_model=env.get("DEFAULT_WHISPER_MODEL", "large-v3"),
        default_whisper_device=env.get("DEFAULT_WHISPER_DEVICE", "cuda"),
        default_ai_provider=env.get("DEFAULT_AI_PROVIDER", "gemini"),
        gpu_available=_check_gpu(),
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
    if req.openai_api_key is not None:
        env_updates["OPENAI_API_KEY"] = req.openai_api_key
    if req.anthropic_api_key is not None:
        env_updates["ANTHROPIC_API_KEY"] = req.anthropic_api_key
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
    """System health check."""
    return SystemHealthResponse(
        status="ok",
        version="1.0.7",
        gpu_available=_check_gpu(),
        ffmpeg_available=_check_ffmpeg(),
        jobs_running=job_store.get_running_count(),
        jobs_queued=job_store.get_queued_count(),
    )
