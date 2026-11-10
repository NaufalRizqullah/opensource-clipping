"""Exercise real Gemini SDK request construction without network or credentials."""

from collections import deque
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from types import ModuleType, SimpleNamespace

from google import genai
from google.genai import types
import httpx
import pytest


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "gemini-3-flash-preview"
FALLBACK = "gemini-2.5-flash"
CLIPS = [{"rank": 1, "start_time": 0.0, "end_time": 15.0}]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def engine(monkeypatch):
    # These imports are unrelated to Gemini and would otherwise load ML tools.
    for name, symbol in (("yt_dlp", "YoutubeDL"), ("faster_whisper", "WhisperModel")):
        stub = ModuleType(name)
        setattr(stub, symbol, object)
        monkeypatch.setitem(sys.modules, name, stub)
    module = _load_module("_gemini_routing_engine", ROOT / "clipping" / "engine.py")
    monkeypatch.setattr(module, "get_analysis_prompt", lambda *args, **kwargs: "Test transcript")
    return module


@pytest.fixture
def cfg():
    return SimpleNamespace(
        api_key_gemini="synthetic-test-key",
        gemini_model=PRIMARY,
        gemini_fallback_model=FALLBACK,
        jumlah_clip=1,
        durasi_hook=5,
    )


@pytest.fixture(params=[False, True], ids=["clean-environment", "platform-overrides"])
def gemini_http(request, monkeypatch):
    # Replace the mapping without reading any developer credentials or URLs.
    environment = {"PYTHON_DOTENV_DISABLED": "1"}
    if request.param:
        environment.update(
            GOOGLE_GENAI_API_VERSION="v1",
            GOOGLE_GENAI_USE_VERTEXAI="true",
            GOOGLE_GENAI_USE_ENTERPRISE="true",
        )
    monkeypatch.setattr(os, "environ", environment)

    def network_forbidden(*args, **kwargs):
        pytest.fail("Gemini regression tests must never open a network connection")

    monkeypatch.setattr(socket.socket, "connect", network_forbidden)
    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    state = SimpleNamespace(responses=deque(), requests=[], options=[], clients=[])

    def respond(request):
        state.requests.append(request)
        assert state.responses, "Unexpected SDK retry or extra request"
        status, payload = state.responses.popleft()
        return httpx.Response(status, json=payload)

    original_client = genai.Client

    def offline_client(*args, **kwargs):
        # Preserve application-selected endpoint, platform and retry settings.
        # Only replace the transport, so SDK defaults remain observable.
        options = kwargs.get("http_options") or types.HttpOptions()
        state.options.append(options)
        options.httpx_client = httpx.Client(
            transport=httpx.MockTransport(respond), trust_env=False
        )
        kwargs["http_options"] = options
        client = original_client(*args, **kwargs)
        state.clients.append(client)
        return client

    monkeypatch.setattr(genai, "Client", offline_client)
    yield state
    for client in state.clients:
        client.close()


def _success(text):
    return 200, {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}


def _failure(status):
    return status, {"error": {"code": status, "status": "NOT_FOUND" if status == 404 else "UNAVAILABLE", "message": "Synthetic SDK failure"}}


def _assert_routes(state, models):
    assert [str(request.url) for request in state.requests] == [
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        for model in models
    ]
    assert all(request.method == "POST" for request in state.requests)
    assert all(client.vertexai is False for client in state.clients)
    assert not state.responses


def test_analysis_uses_developer_beta_and_preserves_json_schema(engine, cfg, gemini_http):
    gemini_http.responses.append(_success(json.dumps(CLIPS)))

    assert engine.analyze_with_gemini("Test transcript", cfg) == CLIPS

    _assert_routes(gemini_http, [PRIMARY])
    body = json.loads(gemini_http.requests[0].content)
    generation = body["generationConfig"]
    assert generation["responseMimeType"] == "application/json"
    schema = generation["responseSchema"]
    assert schema["type"] == "ARRAY"
    assert schema["items"]["type"] == "OBJECT"
    assert {"rank", "start_time", "end_time", "klasifikasi_akun"} <= schema["items"]["properties"].keys()
    assert gemini_http.options[0].timeout == engine.REQUEST_TIMEOUT_MS
    assert gemini_http.options[0].retry_options.attempts == 1


def test_analysis_404_immediately_falls_back_on_same_beta_route(engine, cfg, gemini_http, monkeypatch):
    gemini_http.responses.extend([_failure(404), _success(json.dumps(CLIPS))])
    sleeps = []
    monkeypatch.setattr(engine.time, "sleep", sleeps.append)

    assert engine.analyze_with_gemini("Test transcript", cfg) == CLIPS

    _assert_routes(gemini_http, [PRIMARY, FALLBACK])
    assert sleeps == []
    assert gemini_http.requests[0].content == gemini_http.requests[1].content


def test_analysis_503_keeps_application_retry_policy(engine, cfg, gemini_http, monkeypatch):
    gemini_http.responses.extend([_failure(503), _success(json.dumps(CLIPS))])
    sleeps = []
    monkeypatch.setattr(engine.time, "sleep", sleeps.append)

    assert engine.analyze_with_gemini("Test transcript", cfg) == CLIPS

    _assert_routes(gemini_http, [PRIMARY, PRIMARY])
    assert sleeps == [engine.INITIAL_WAIT_SECONDS]


@pytest.mark.parametrize("primary_status", [200, 404], ids=["primary", "fallback"])
def test_voiceover_uses_developer_beta_for_primary_and_fallback(cfg, gemini_http, primary_status):
    voiceover = _load_module("_gemini_routing_voiceover", ROOT / "clipping" / "voiceover.py")
    if primary_status == 404:
        gemini_http.responses.append(_failure(404))
    gemini_http.responses.append(_success('"Test commentary"'))

    assert voiceover.generate_commentary_script("Test transcript", cfg) == "Test commentary"

    _assert_routes(gemini_http, [PRIMARY, FALLBACK] if primary_status == 404 else [PRIMARY])
    generation = json.loads(gemini_http.requests[-1].content)["generationConfig"]
    assert generation["temperature"] == 0.7
    assert generation["topP"] == 0.9
