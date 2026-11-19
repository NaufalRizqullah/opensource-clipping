"""Shared execution-path preflight and local-only temporal validation."""

import math
from pathlib import Path
import re

from .base import InvalidOutputError, ProviderConfigError


def validate_analysis_config(cfg, *, probe=False, check_cache=True):
    cached = getattr(cfg, "load_gemini_json", False)
    voiceover = getattr(cfg, "voiceover", False)
    provider = getattr(cfg, "ai_provider", "gemini")
    if cached and check_cache and not (Path(cfg.outputs_dir) / "gemini_response.json").is_file():
        raise ProviderConfigError("Cached AI JSON tidak ditemukan. Nonaktifkan Bypass AI untuk analisis baru.")
    if cached and not voiceover:
        return
    if provider == "vllm":
        from .config import vllm_config_values
        vllm_config_values(vars(cfg))
        if probe:
            from .vllm import VllmProvider
            with VllmProvider.from_cfg(cfg) as local:
                local.validate()
        return
    if provider not in {"gemini", "nvidia"}:
        raise ProviderConfigError("AI provider tidak dikenal.")
    if voiceover and not getattr(cfg, "api_key_gemini", ""):
        raise ProviderConfigError("GOOGLE_API_KEY diperlukan untuk voiceover provider cloud.")
    if cached:
        return
    if provider == "nvidia" and getattr(cfg, "api_key_nvidia", ""):
        return
    if not getattr(cfg, "api_key_gemini", ""):
        key = "NVIDIA_API_KEY atau GOOGLE_API_KEY" if provider == "nvidia" else "GOOGLE_API_KEY"
        raise ProviderConfigError(f"{key} tidak ditemukan. Set via Settings atau environment.")


_STAMP = re.compile(r"^\[(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\]\s*", re.MULTILINE)


def transcript_end(transcript):
    # Do not silently shorten the source when a trailing timestamp is malformed.
    for line in transcript.splitlines():
        if line.lstrip().startswith("[") and _STAMP.match(line) is None:
            raise ProviderConfigError("Transkrip vLLM memiliki baris timestamp yang tidak valid.")
    pairs = [(float(m[1]), float(m[2])) for m in _STAMP.finditer(transcript)]
    if not pairs or any(not math.isfinite(s + e) or s < 0 or e < s for s, e in pairs):
        raise ProviderConfigError("Transkrip vLLM harus memiliki timestamp [start - end] yang valid.")
    end = max(e for _, e in pairs)
    if end <= 0:
        raise ProviderConfigError("Durasi transkrip harus lebih dari nol.")
    return end


def _number(value):
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise InvalidOutputError("Timestamp/skor klip harus angka finite.")
    return value


def _interval(item, low, high, *, start="start_time", end="end_time"):
    try:
        a, b = _number(item[start]), _number(item[end])
    except (KeyError, TypeError):
        raise InvalidOutputError("Field timestamp klip/segmen tidak lengkap.") from None
    if not low <= a < b <= high:
        raise InvalidOutputError("Timestamp klip/hook/segmen di luar rentang sumber atau interval terbalik.")
    return a, b


def _ordered_intervals(items, low, high):
    last = low
    for item in items:
        start, end = _interval(item, low, high)
        if start < last:
            raise InvalidOutputError("Segmen harus terurut dan tidak saling overlap.")
        last = end


def validate_clips(clips, *, source_end, requested_clips, min_duration=20, max_duration=179):
    if not clips or len(clips) > requested_clips:
        raise InvalidOutputError("Jumlah klip kosong atau melebihi jumlah yang diminta.")
    intervals = []
    for item in clips:
        a, b = _interval(item, 0, source_end)
        if not min_duration <= b - a <= max_duration:
            raise InvalidOutputError(f"Durasi klip harus {min_duration}-{max_duration} detik.")
        if not 1 <= _number(item["viral_score"]) <= 100:
            raise InvalidOutputError("Viral score harus 1-100.")
        _interval(item, a, b, start="hook_start_time", end="hook_end_time")
        _ordered_intervals(item["keep_segments"], a, b)
        _ordered_intervals(item["broll_list"], a, b)
        # Hook montage items may deliberately be re-ordered, but stay in source clip.
        for hook in item["hook_v2"]["items"]:
            _interval(hook, a, b)
        if item["hook_v2"]["enabled"] and not item["hook_v2"]["items"]:
            raise InvalidOutputError("Hook V2 aktif tetapi tidak mempunyai segmen.")
        intervals.append((a, b))
    ordered = sorted(intervals)
    if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
        raise InvalidOutputError("Klip saling overlap; pilih momen yang berbeda.")
    return clips
