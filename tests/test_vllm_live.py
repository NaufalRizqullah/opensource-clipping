"""Explicit Kaggle smoke test; unavailable configured servers must fail."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.local_llm
@pytest.mark.skipif(os.environ.get("RUN_LOCAL_LLM_TESTS") != "1", reason="Set RUN_LOCAL_LLM_TESTS=1 in Kaggle to opt in")
def test_kaggle_vllm_full_clip_contract():
    from clipping.engine import analyze_with_vllm
    from clipping.providers.config import vllm_config_values

    cfg = SimpleNamespace(
        ai_provider="vllm", jumlah_clip=1, durasi_hook=3,
        **vllm_config_values(),
    )
    transcript = (Path(__file__).parent / "fixtures/eval_transcript_short.txt").read_text(encoding="utf-8")
    result = analyze_with_vllm(transcript, cfg)
    assert len(result) == 1
    assert cfg.ai_run_meta["actual_provider"] == "vllm"
    assert cfg.ai_run_meta["fallback_used"] is False
    assert cfg.ai_run_meta["status"] == "completed"
