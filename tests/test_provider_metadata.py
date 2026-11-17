"""Run/cache provenance contracts, authored for execution in Kaggle."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from clipping.providers import metadata
from clipping.providers.base import ProviderUnavailableError


def _cfg(tmp_path, **overrides):
    values = dict(outputs_dir=str(tmp_path), ai_provider="vllm")
    values.update(overrides)
    return SimpleNamespace(**values)


def _read(tmp_path):
    return json.loads((tmp_path / "ai_run_meta.json").read_text(encoding="utf-8"))


def _completed(model="local-model"):
    return dict(actual_provider="vllm", model=model, status="completed", attempts=1)


def test_completed_run_hashes_saved_response_and_repeated_cache_preserves_origin(tmp_path):
    response = b'[{"rank": 1}]'
    (tmp_path / "gemini_response.json").write_bytes(response)
    cfg = _cfg(tmp_path, ai_run_meta=_completed())
    metadata.save_run_metadata(cfg)
    digest = hashlib.sha256(response).hexdigest()
    assert _read(tmp_path)["response_sha256"] == digest

    cfg.ai_provider = "gemini"
    for _ in range(2):
        metadata.save_run_metadata(cfg, cached=True)
        saved = _read(tmp_path)
        assert saved["actual_provider"] == "vllm"
        assert saved["model"] == "local-model"
        assert saved["requested_provider"] == "gemini"
        assert saved["response_sha256"] == digest
        assert saved["cache_hit"] is True
        assert saved["attempts"] == saved["duration_seconds"] == 0


def test_modified_cache_loses_old_provider_attribution(tmp_path):
    response = tmp_path / "gemini_response.json"
    response.write_text('[{"rank": 1}]', encoding="utf-8")
    cfg = _cfg(tmp_path, ai_run_meta=_completed())
    metadata.save_run_metadata(cfg)
    response.write_text('[{"rank": 2}]', encoding="utf-8")

    metadata.save_run_metadata(cfg, cached=True)
    saved = _read(tmp_path)
    assert saved["actual_provider"] is None
    assert saved["model"] is None
    assert saved["response_sha256"] == hashlib.sha256(response.read_bytes()).hexdigest()


def test_failed_rerun_cannot_claim_previous_response(tmp_path):
    response = tmp_path / "gemini_response.json"
    response.write_text('[{"rank": 1}]', encoding="utf-8")
    cfg = _cfg(tmp_path, ai_run_meta=_completed("previous-model"))
    metadata.save_run_metadata(cfg)
    cfg.ai_run_meta = dict(
        actual_provider="vllm", model="failed-model", status="failed", attempts=3,
        error_type="InvalidOutputError", response_sha256=_read(tmp_path)["response_sha256"],
    )
    metadata.save_run_metadata(cfg)
    failed = _read(tmp_path)
    assert failed["status"] == "failed"
    assert failed["model"] == "failed-model"
    assert "response_sha256" not in failed
    assert response.read_text(encoding="utf-8") == '[{"rank": 1}]'

    metadata.save_run_metadata(cfg, cached=True)
    assert _read(tmp_path)["actual_provider"] is None
    assert _read(tmp_path)["model"] is None


@pytest.mark.parametrize("previous", [None, "broken JSON", "[]", '{"actual_provider":"vllm","model":"legacy"}'])
def test_cache_without_verifiable_metadata_has_unknown_origin(tmp_path, previous):
    (tmp_path / "gemini_response.json").write_text("[]", encoding="utf-8")
    if previous is not None:
        (tmp_path / "ai_run_meta.json").write_text(previous, encoding="utf-8")
    metadata.save_run_metadata(_cfg(tmp_path), cached=True)
    saved = _read(tmp_path)
    assert saved["actual_provider"] is None
    assert saved["model"] is None
    assert saved["cache_hit"] is True


def test_failed_metadata_is_rejected_even_if_its_hash_matches_cache(tmp_path):
    response = b"[]"
    (tmp_path / "gemini_response.json").write_bytes(response)
    previous = dict(_completed(), status="failed", response_sha256=hashlib.sha256(response).hexdigest())
    (tmp_path / "ai_run_meta.json").write_text(json.dumps(previous), encoding="utf-8")
    metadata.save_run_metadata(_cfg(tmp_path), cached=True)
    assert _read(tmp_path)["actual_provider"] is None


def test_legacy_provider_success_does_not_invent_model_or_fallback_origin(tmp_path):
    (tmp_path / "gemini_response.json").write_text("[]", encoding="utf-8")
    metadata.save_run_metadata(_cfg(tmp_path, ai_provider="nvidia"))
    saved = _read(tmp_path)
    assert saved["requested_provider"] == "nvidia"
    assert saved["actual_provider"] is None
    assert saved["model"] is None
    assert saved["instrumented"] is False
    assert saved["status"] == "completed"


def test_completed_metadata_without_cache_has_no_hash(tmp_path):
    metadata.save_run_metadata(_cfg(tmp_path, ai_run_meta=_completed()))
    assert "response_sha256" not in _read(tmp_path)


@pytest.fixture
def failed_worker(monkeypatch, tmp_path):
    import clipping
    from web.api import worker

    video = tmp_path / "video_asli.mp4"
    video.write_bytes(b"synthetic video; never decoded")
    cfg = _cfg(
        tmp_path, file_video_asli=str(video), url_youtube=None,
        max_kata_per_subtitle=5, whisper_model="base", whisper_device="cpu",
        whisper_compute_type="int8", load_gemini_json=False,
    )
    finished = []
    events = []

    def fail_analysis(transcript, config):
        config.ai_run_meta = dict(
            actual_provider="vllm", model="failed-model", status="failed",
            attempts=3, error_type="ProviderUnavailableError",
        )
        raise ProviderUnavailableError("Synthetic inference failure")

    engine = SimpleNamespace(
        transcribe_video=lambda *args, **kwargs: ("transcript", []),
        analyze_with_ai=fail_analysis,
    )
    monkeypatch.setattr(clipping, "engine", engine, raising=False)
    monkeypatch.setitem(sys.modules, "clipping.engine", engine)
    monkeypatch.setattr(worker, "build_config_from_payload", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(worker, "_validate_analysis_config", lambda value: None)
    monkeypatch.setattr(worker, "log_event", lambda name, **fields: events.append((name, fields)))
    monkeypatch.setattr(worker, "store", SimpleNamespace(
        is_cancel_requested=lambda job_id: False,
        set_status=lambda *args, **kwargs: None,
        update_progress=lambda *args, **kwargs: None,
        finish_job=lambda *args, **kwargs: finished.append(kwargs),
    ))
    return SimpleNamespace(worker=worker, finished=finished, events=events, root=tmp_path)


def test_worker_saves_failed_analysis_provenance_without_changing_old_cache(failed_worker):
    response = failed_worker.root / "gemini_response.json"
    response.write_text("[]", encoding="utf-8")
    failed_worker.worker._run_pipeline_sync("synthetic-job", {})
    saved = _read(failed_worker.root)
    assert saved["status"] == "failed"
    assert saved["attempts"] == 3
    assert "response_sha256" not in saved
    assert response.read_text(encoding="utf-8") == "[]"
    assert "Synthetic inference failure" in failed_worker.finished[0]["error"]


def test_worker_preserves_inference_error_when_metadata_write_fails(failed_worker, monkeypatch):
    def cannot_write(*args, **kwargs):
        raise OSError("Synthetic sidecar write failure")

    monkeypatch.setattr(metadata, "save_run_metadata", cannot_write)
    failed_worker.worker._run_pipeline_sync("synthetic-job", {})
    assert "Synthetic inference failure" in failed_worker.finished[0]["error"]
    assert "sidecar write failure" not in failed_worker.finished[0]["error"]
    assert any(name == "analysis_metadata_failed" for name, _ in failed_worker.events)
