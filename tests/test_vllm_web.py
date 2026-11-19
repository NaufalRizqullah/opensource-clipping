"""Offline web contracts for the Kaggle test run; no server or GPU required."""

import asyncio
import os
import socket
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from web.api import config_adapter, worker
from web.api.models import AIProvider, JobCreateRequest, SettingsRequest
from web.api.routes import settings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", {"PYTHON_DOTENV_DISABLED": "1"})
    monkeypatch.setattr(worker, "_settings_env", {})
    monkeypatch.setattr(settings, "_check_gpu", lambda: False)

    def forbidden(*args, **kwargs):
        pytest.fail("Web contract tests must not connect to an inference server")

    real_connect = socket.socket.connect
    real_socketpair = socket.socketpair
    in_socketpair = False

    def guarded_connect(sock, *args, **kwargs):
        if not in_socketpair:
            forbidden()
        return real_connect(sock, *args, **kwargs)

    def socketpair(*args, **kwargs):
        # On Windows, socket.socketpair() (used by asyncio's event loop) is built
        # from a loopback connect; that local self-pipe is not an inference call.
        nonlocal in_socketpair
        in_socketpair = True
        try:
            return real_socketpair(*args, **kwargs)
        finally:
            in_socketpair = False

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "socketpair", socketpair)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_vllm_request_accepts_alias_and_explicit_output_budget():
    request = JobCreateRequest(ai_provider="vllm", vllm_model="clipping-local", local_max_tokens=2048)
    assert request.ai_provider is AIProvider.VLLM
    assert request.vllm_model == "clipping-local"
    assert request.local_max_tokens == 2048
    defaults = JobCreateRequest()
    assert defaults.ai_provider is AIProvider.GEMINI
    assert defaults.vllm_model is None
    assert defaults.local_max_tokens is None


@pytest.mark.parametrize("budget", [0, -1, 131073, 1.5, True, "4096"])
def test_invalid_output_budget_rejected(budget):
    with pytest.raises(ValidationError):
        JobCreateRequest(ai_provider="vllm", local_max_tokens=budget)


@pytest.mark.parametrize("model", ["", "bad model", "bad\nmodel", "x" * 257])
def test_invalid_model_rejected(model):
    with pytest.raises(ValidationError):
        JobCreateRequest(ai_provider="vllm", vllm_model=model)


def test_adapter_uses_operator_endpoint_and_key_not_job_values(monkeypatch, tmp_path):
    monkeypatch.setattr(config_adapter, "PROJECT_ROOT", tmp_path)
    os.environ.update({
        "VLLM_BASE_URL": "http://127.0.0.1:8001",
        "VLLM_API_KEY": "synthetic-env-key",
        "VLLM_MODEL": "operator-model",
        "VLLM_MAX_TOKENS": "3072",
    })
    payload = JobCreateRequest(ai_provider="vllm", vllm_model="job-model", local_max_tokens=1024).model_dump()
    payload.update(vllm_base_url="http://169.254.169.254", api_key_vllm="injected", VLLM_API_KEY="injected")
    cfg = config_adapter.build_config_from_payload(payload, "vllm-job", env_overrides={"VLLM_API_KEY": "synthetic-override"})
    assert cfg.vllm_model == "job-model"
    assert cfg.local_max_tokens == 1024
    assert cfg.api_key_vllm == "synthetic-override"
    assert cfg.vllm_base_url.rstrip("/").removesuffix("/v1") == "http://127.0.0.1:8001"
    assert cfg.api_key_gemini == ""


def test_omitted_job_model_and_budget_use_operator_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(config_adapter, "PROJECT_ROOT", tmp_path)
    cfg = config_adapter.build_config_from_payload(
        JobCreateRequest(ai_provider="vllm").model_dump(), "defaults",
        env_overrides={"VLLM_MODEL": "served-alias", "VLLM_MAX_TOKENS": "2048"},
    )
    assert cfg.vllm_model == "served-alias"
    assert cfg.local_max_tokens == 2048


def test_cloud_job_does_not_parse_unused_local_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(config_adapter, "PROJECT_ROOT", tmp_path)
    cfg = config_adapter.build_config_from_payload(
        {"ai_provider": "gemini"}, "cloud", env_overrides={"VLLM_MAX_TOKENS": "invalid"},
    )
    assert cfg.ai_provider == "gemini"


def test_cached_job_ignores_unavailable_local_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(config_adapter, "PROJECT_ROOT", tmp_path)
    cfg = config_adapter.build_config_from_payload(
        JobCreateRequest(ai_provider="vllm", load_gemini_json=True).model_dump(), "cached",
        env_overrides={"VLLM_MAX_TOKENS": "invalid", "VLLM_BASE_URL": "removed-server"},
    )
    (tmp_path / "outputs" / "cached" / "gemini_response.json").write_text("[]", encoding="utf-8")
    worker._validate_analysis_config(cfg)
    assert cfg.ai_provider == "vllm"


def test_vllm_cpu_whisper_accepts_studio_int8_setting(monkeypatch, tmp_path):
    monkeypatch.setattr(config_adapter, "PROJECT_ROOT", tmp_path)
    cfg = config_adapter.build_config_from_payload(
        JobCreateRequest(
            ai_provider="vllm", whisper_device="cpu", whisper_compute_type="int8",
        ).model_dump(), "cpu-whisper",
    )
    assert cfg.whisper_device == "cpu"
    assert cfg.whisper_compute_type == "int8"


def test_worker_delegates_without_network_probe(monkeypatch):
    from clipping.providers import validation

    calls = []
    cfg = SimpleNamespace(ai_provider="vllm")
    monkeypatch.setattr(validation, "validate_analysis_config", lambda value, *, probe: calls.append((value, probe)))
    worker._validate_analysis_config(cfg)
    assert calls == [(cfg, False)]


def test_cached_vllm_analysis_needs_no_server_or_cloud_key(tmp_path):
    (tmp_path / "gemini_response.json").write_text("[]", encoding="utf-8")
    cfg = SimpleNamespace(
        ai_provider="vllm", load_gemini_json=True, outputs_dir=str(tmp_path),
        api_key_gemini="", api_key_nvidia="", voiceover=False,
    )
    worker._validate_analysis_config(cfg)


def test_settings_expose_local_defaults_without_secret_or_endpoint():
    worker.set_settings_env({
        "VLLM_BASE_URL": "http://private-inference:8001",
        "VLLM_API_KEY": "synthetic-secret",
        "VLLM_MODEL": "served-alias",
        "VLLM_MAX_TOKENS": "2048",
    })
    response = asyncio.run(settings.get_settings()).model_dump()
    assert response["vllm_api_key_set"] is True
    assert response["vllm_model"] == "served-alias"
    assert response["local_max_tokens"] == 2048
    assert "synthetic-secret" not in str(response)
    assert "private-inference" not in str(response)
    assert "vllm_api_key" not in response


def test_public_settings_cannot_change_local_endpoint_or_key():
    worker.set_settings_env({"VLLM_API_KEY": "operator-key", "VLLM_BASE_URL": "http://127.0.0.1:8001"})
    request = SettingsRequest(
        default_ai_provider="vllm", vllm_api_key="untrusted-key", vllm_base_url="http://169.254.169.254",
    )
    response = asyncio.run(settings.update_settings(request))
    assert response.default_ai_provider == "vllm"
    assert worker.get_settings_env()["VLLM_API_KEY"] == "operator-key"
    assert worker.get_settings_env()["VLLM_BASE_URL"] == "http://127.0.0.1:8001"


def test_bad_unused_local_budget_does_not_break_settings():
    worker.set_settings_env({"VLLM_MAX_TOKENS": "not-a-number"})
    assert asyncio.run(settings.get_settings()).local_max_tokens == 4096
