"""CLI configuration contracts; no video, model, server, or network is started."""

import os
import socket

import pytest

from clipping.config import build_config
from clipping.providers.base import ProviderConfigError
from clipping.providers.config import DEFAULT_LOCAL_MAX_TOKENS, DEFAULT_VLLM_MODEL
from clipping.providers.validation import validate_analysis_config


@pytest.fixture(autouse=True)
def isolated_cli_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "environ", {"PYTHON_DOTENV_DISABLED": "1"})

    def forbidden(*args, **kwargs):
        pytest.fail("Configuration tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def config(*args):
    return build_config(["--url", "https://example.invalid/video", *args])


def test_vllm_defaults_need_no_cloud_key():
    cfg = config("--ai-provider", "vllm")
    assert cfg.vllm_model == DEFAULT_VLLM_MODEL
    assert cfg.vllm_base_url == "http://127.0.0.1:8001"
    assert cfg.local_max_tokens == DEFAULT_LOCAL_MAX_TOKENS
    assert cfg.api_key_vllm == cfg.api_key_gemini == cfg.api_key_nvidia == ""
    validate_analysis_config(cfg)


def test_environment_sets_local_defaults():
    os.environ.update({
        "VLLM_MODEL": "operator-alias", "VLLM_BASE_URL": "http://192.168.1.10:8001/v1",
        "VLLM_API_KEY": "synthetic-key", "VLLM_MAX_TOKENS": "2048",
        "VLLM_CONTEXT_LIMIT": "8192", "VLLM_TIMEOUT": "300.5",
    })
    cfg = config("--ai-provider", "vllm")
    assert cfg.vllm_model == "operator-alias"
    assert cfg.vllm_base_url == "http://192.168.1.10:8001"
    assert cfg.api_key_vllm == "synthetic-key"
    assert cfg.local_max_tokens == 2048
    assert cfg.vllm_context_limit == 8192
    assert cfg.vllm_timeout == 300.5


def test_explicit_cli_flags_override_local_environment():
    os.environ.update({
        "VLLM_MODEL": "env-model", "VLLM_BASE_URL": "http://localhost:8001",
        "VLLM_MAX_TOKENS": "4096", "VLLM_CONTEXT_LIMIT": "16384", "VLLM_TIMEOUT": "900",
    })
    cfg = config(
        "--ai-provider", "vllm", "--vllm-model", "requested-alias",
        "--local-base-url", "https://server.invalid/inference/v1/",
        "--local-max-tokens", "1024", "--local-context-limit", "4096",
        "--local-timeout", "60.5", "--whisper-device", "cpu", "--whisper-compute-type", "int8",
    )
    assert cfg.vllm_model == "requested-alias"
    assert cfg.vllm_base_url == "https://server.invalid/inference"
    assert cfg.local_max_tokens == 1024
    assert cfg.vllm_context_limit == 4096
    assert cfg.vllm_timeout == 60.5
    assert cfg.whisper_device == "cpu" and cfg.whisper_compute_type == "int8"


def test_default_cloud_provider_ignores_unused_bad_local_environment():
    os.environ.update({"VLLM_BASE_URL": "file:///invalid", "VLLM_MAX_TOKENS": "invalid"})
    cfg = config()
    assert cfg.ai_provider == "gemini"


def test_cache_only_cli_ignores_unavailable_local_settings(tmp_path):
    os.environ.update({"VLLM_BASE_URL": "file:///invalid", "VLLM_MAX_TOKENS": "invalid"})
    cfg = config("--ai-provider", "vllm", "--load-gemini-json")
    (tmp_path / "outputs" / "gemini_response.json").write_text("[]", encoding="utf-8")
    validate_analysis_config(cfg, probe=True)


def test_cache_with_new_voiceover_requires_valid_local_settings():
    os.environ["VLLM_BASE_URL"] = "file:///invalid"
    with pytest.raises(ProviderConfigError):
        config("--ai-provider", "vllm", "--load-gemini-json", "--voiceover")


@pytest.mark.parametrize("flag,value", [
    ("--local-max-tokens", "0"), ("--local-context-limit", "-1"),
    ("--local-timeout", "nan"), ("--local-base-url", "file:///invalid"),
    ("--vllm-model", "bad model"),
])
def test_invalid_explicit_local_options_fail_before_execution(flag, value):
    with pytest.raises(ProviderConfigError):
        config("--ai-provider", "vllm", flag, value)


def test_nvidia_cli_accepts_nvidia_key_without_google_key():
    os.environ["NVIDIA_API_KEY"] = "synthetic-key"
    cfg = config("--ai-provider", "nvidia")
    validate_analysis_config(cfg)


def test_unknown_provider_is_rejected_by_cli():
    with pytest.raises(SystemExit) as captured:
        config("--ai-provider", "unsupported")
    assert captured.value.code == 2
