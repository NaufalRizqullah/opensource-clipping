"""Cloud contracts frozen before provider extraction.

Schema fixtures were extracted from the source AST at commit
ac2945549b02185e033b81e86a31bb30d237d51e, never from the new schema builders.
The real Gemini SDK routing/retry coverage remains in test_gemini_routing.py.
"""

import importlib.util
import json
from pathlib import Path
import socket
import sys
from types import ModuleType, SimpleNamespace

import pytest

from clipping.providers.schema import build_clips_schema, build_gemini_schema


ROOT = Path(__file__).resolve().parents[1]
CLIPS = [{"rank": 1, "start_time": 10.0, "end_time": 40.0}]


def _snapshot(provider):
    path = ROOT / "tests" / "fixtures" / f"provider_{provider}_schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Cloud contract tests must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def engine(monkeypatch):
    # Load the facade by path, as supported by the existing Gemini test suite.
    for name, symbol in (("yt_dlp", "YoutubeDL"), ("faster_whisper", "WhisperModel")):
        stub = ModuleType(name)
        setattr(stub, symbol, object)
        monkeypatch.setitem(sys.modules, name, stub)
    spec = importlib.util.spec_from_file_location(
        "_provider_contract_engine", ROOT / "clipping" / "engine.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cfg():
    # Deliberately omit ai_provider: direct legacy functions accept minimal cfg.
    return SimpleNamespace(
        api_key_gemini="synthetic-gemini-key",
        gemini_model="gemini-3-flash-preview",
        gemini_fallback_model="gemini-2.5-flash",
        api_key_nvidia="synthetic-nvidia-key",
        nvidia_model="test/nim-model",
        jumlah_clip=1,
        durasi_hook=5,
    )


@pytest.mark.parametrize(
    ("provider", "builder"),
    [("gemini", build_gemini_schema), ("nvidia", build_clips_schema)],
)
def test_schema_matches_pre_refactor_snapshot(engine, provider, builder):
    actual = builder(engine.TARGET_ACCOUNTS)
    expected = _snapshot(provider)
    assert actual == expected
    # Preserve property order as well as required/enum list ordering.
    assert json.dumps(actual) == json.dumps(expected)


def test_schema_uses_current_accounts_without_sharing_mutable_state():
    accounts = {"Learning": {}, "Science": {}}
    first = build_clips_schema(accounts)
    account_schema = first["items"]["properties"]["klasifikasi_akun"]["properties"]
    assert account_schema["tipe_akun"]["enum"] == ["Learning", "Science"]
    assert account_schema["alternatif_akun"]["properties"]["tipe_akun"]["enum"] == [
        "Learning", "Science",
    ]
    account_schema["tipe_akun"]["enum"].append("Unexpected")
    first["items"]["required"].clear()
    second = build_clips_schema(accounts)
    assert second["items"]["required"]
    assert second["items"]["properties"]["klasifikasi_akun"]["properties"]["tipe_akun"]["enum"] == [
        "Learning", "Science",
    ]
    assert list(accounts) == ["Learning", "Science"]


@pytest.fixture
def nim_sdk(monkeypatch):
    state = SimpleNamespace(constructors=[], requests=[], content=json.dumps(CLIPS))

    def create(**kwargs):
        state.requests.append(kwargs)
        return SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=state.content)),
        ])

    def client(**kwargs):
        state.constructors.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    stub = ModuleType("openai")
    stub.OpenAI = client
    monkeypatch.setitem(sys.modules, "openai", stub)
    return state


def test_nim_facade_preserves_full_request_and_prompt_patch(engine, cfg, nim_sdk, monkeypatch):
    prompts = []

    def prompt(transcript, count, hook, *, cfg):
        prompts.append((transcript, count, hook, cfg))
        return "Patched analysis prompt"

    monkeypatch.setattr(engine, "get_analysis_prompt", prompt)
    assert engine.analyze_with_nvidia("Test transcript", cfg) == CLIPS
    assert prompts == [("Test transcript", 1, 5, cfg)]
    # Do not add application retry/timeout overrides to the legacy SDK defaults.
    assert nim_sdk.constructors == [{
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": "synthetic-nvidia-key",
    }]
    assert nim_sdk.requests == [{
        "model": "test/nim-model",
        "messages": [
            {
                "role": "system",
                "content": "You are a professional video editor and strategist. Return JSON only. Follow the provided JSON schema exactly.",
            },
            {"role": "user", "content": "Patched analysis prompt"},
        ],
        "temperature": 0.5,
        "top_p": 1,
        "max_tokens": 16384,
        "extra_body": {
            "chat_template_kwargs": {"thinking": False},
            "nvext": {"guided_json": _snapshot("nvidia")},
        },
    }]


@pytest.mark.parametrize("wrapper", ["array", "fence", "clips", "data", "highlights", "object"])
def test_nim_legacy_response_shapes(cfg, nim_sdk, wrapper):
    from clipping.providers.nvidia import NvidiaNimProvider

    if wrapper == "fence":
        nim_sdk.content = f"```json\n{json.dumps(CLIPS)}\n```"
    elif wrapper == "object":
        nim_sdk.content = json.dumps(CLIPS[0])
    elif wrapper != "array":
        nim_sdk.content = json.dumps({wrapper: CLIPS})
    result = NvidiaNimProvider.from_cfg(cfg).generate_clips("Test prompt", _snapshot("nvidia"))
    assert result == CLIPS
    assert len(nim_sdk.requests) == 1


def test_gemini_facade_keeps_live_prompt_retry_and_timeout_patch_points(engine, cfg, monkeypatch):
    from clipping.providers import gemini

    received = {}
    result = object()

    def prompt(transcript, count, hook, *, cfg):
        received["prompt_args"] = (transcript, count, hook, cfg)
        return "Patched Gemini prompt"

    def generate_json(**kwargs):
        return result

    class CapturedProvider:
        def __init__(self, cfg, **kwargs):
            received["cfg"] = cfg
            received.update(kwargs)

        def generate_clips(self, prompt, schema):
            received["prompt"] = prompt
            received["schema"] = schema
            return received["generate_json"]()

    monkeypatch.setattr(engine, "get_analysis_prompt", prompt)
    monkeypatch.setattr(engine, "_generate_json_with_retry", generate_json)
    monkeypatch.setattr(engine, "REQUEST_TIMEOUT_MS", 12345)
    monkeypatch.setattr(gemini, "GeminiProvider", CapturedProvider)

    assert engine.analyze_with_gemini("Test transcript", cfg) is result
    assert received["cfg"] is cfg
    assert received["prompt_args"] == ("Test transcript", 1, 5, cfg)
    assert received["prompt"] == "Patched Gemini prompt"
    assert received["generate_json"] is generate_json
    assert received["timeout_ms"] == 12345
    assert received["schema"] == _snapshot("gemini")


@pytest.mark.parametrize(
    ("provider", "has_nvidia_key", "nvidia_fails", "expected_calls"),
    [
        (None, True, False, ["gemini"]),
        ("gemini", True, False, ["gemini"]),
        ("nvidia", True, False, ["nvidia"]),
        ("nvidia", False, False, ["gemini"]),
        ("nvidia", True, True, ["nvidia", "gemini"]),
        ("vllm", False, False, ["vllm"]),
    ],
)
def test_dispatch_preserves_cloud_fallback(
    engine, cfg, monkeypatch, provider, has_nvidia_key, nvidia_fails, expected_calls,
):
    if provider is not None:
        cfg.ai_provider = provider
    if not has_nvidia_key:
        cfg.api_key_nvidia = ""
    calls = []

    def implementation(name):
        def analyze(transcript, config):
            assert transcript == "Test transcript"
            assert config is cfg
            calls.append(name)
            if name == "nvidia" and nvidia_fails:
                raise RuntimeError("Synthetic NIM error")
            return CLIPS
        return analyze

    for name in ("gemini", "nvidia", "vllm"):
        monkeypatch.setattr(engine, f"analyze_with_{name}", implementation(name))
    assert engine.analyze_with_ai("Test transcript", cfg) is CLIPS
    assert calls == expected_calls


def test_local_failure_never_enters_legacy_cloud_fallback(engine, cfg, monkeypatch):
    cfg.ai_provider = "vllm"
    error = RuntimeError("Synthetic local server failure")

    def fail_local(*args):
        raise error

    def cloud_forbidden(*args):
        pytest.fail("Local analysis must not send a transcript to a cloud provider")

    monkeypatch.setattr(engine, "analyze_with_vllm", fail_local)
    monkeypatch.setattr(engine, "analyze_with_gemini", cloud_forbidden)
    monkeypatch.setattr(engine, "analyze_with_nvidia", cloud_forbidden)
    with pytest.raises(RuntimeError) as caught:
        engine.analyze_with_ai("Test transcript", cfg)
    assert caught.value is error
