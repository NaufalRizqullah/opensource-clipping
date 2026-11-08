import json
import subprocess
from types import SimpleNamespace

import pytest

from web.api import readiness


@pytest.fixture(autouse=True)
def reset_cache(monkeypatch):
    monkeypatch.setattr(readiness, "_cache", None)


def test_probes_real_imports_in_fresh_process_and_caches(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "-c":
            assert "clipping.studio" in command[2]
            assert "numpy.random" in command[2]
            assert kwargs["env"]["PYTHON_DOTENV_DISABLED"] == "1"
            payload = {"checks": {"renderer": {"ok": True}}, "versions": {"numpy": "1.26.4"}}
            return SimpleNamespace(returncode=0, stdout="library output\nOSC_READINESS=" + json.dumps(payload))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(readiness.subprocess, "run", run)
    monkeypatch.setattr(readiness.shutil, "which", lambda name: "ffmpeg")
    first = readiness.get_readiness()
    assert first["ready"] is True
    first["checks"]["renderer"]["ok"] = False
    assert readiness.get_readiness()["checks"]["renderer"]["ok"] is True
    assert len(calls) == 2


@pytest.mark.parametrize("outcome", ["timeout", "crash", "malformed", "abi"])
def test_probe_fails_closed_without_echoing_subprocess_output(monkeypatch, outcome):
    sensitive = "not-a-real-secret-must-not-appear"

    def run(command, **kwargs):
        if command[1] != "-c":
            return SimpleNamespace(returncode=0)
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 120, output=sensitive)
        if outcome == "abi":
            data = {"checks": {"renderer": {"ok": False, "error": "ValueError", "message": "Native dependency ABI mismatch"}}, "versions": {}}
            return SimpleNamespace(returncode=0, stdout=sensitive + "\nOSC_READINESS=" + json.dumps(data))
        return SimpleNamespace(returncode=1 if outcome == "crash" else 0, stdout=sensitive)

    monkeypatch.setattr(readiness.subprocess, "run", run)
    monkeypatch.setattr(readiness.shutil, "which", lambda name: "ffmpeg")
    result = readiness.get_readiness()
    assert result["ready"] is False
    assert sensitive not in json.dumps(result)


def test_missing_ffmpeg_fails_readiness(monkeypatch):
    monkeypatch.setattr(readiness.shutil, "which", lambda name: None)
    monkeypatch.setattr(readiness.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout='OSC_READINESS={"checks":{"renderer":{"ok":true}},"versions":{}}',
    ))
    assert readiness.get_readiness()["checks"]["ffmpeg"]["ok"] is False


def test_child_probe_reports_observed_numpy_abi_failure_safely(monkeypatch, capsys):
    import importlib
    import importlib.metadata

    def import_module(name):
        if name == "mediapipe.tasks.python.vision":
            raise ValueError("numpy.dtype size changed: private-context-must-not-be-echoed")
        return object()

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "test-version")
    exec(readiness._PROBE_CODE, {})
    output = capsys.readouterr().out
    report = json.loads(output.removeprefix(readiness._RESULT_PREFIX))
    assert report["checks"]["renderer"]["ok"] is False
    assert report["checks"]["renderer"]["error"] == "ValueError"
    assert "ABI mismatch" in report["checks"]["renderer"]["message"]
    assert "private-context" not in output


def test_readiness_route_does_not_confuse_liveness(monkeypatch):
    from fastapi.testclient import TestClient
    from web.api.app import app
    from web.api.routes import settings

    monkeypatch.setattr(settings, "get_readiness", lambda: {"ready": False, "checks": {"renderer": {"ok": False}}})
    monkeypatch.setattr(settings, "_check_gpu", lambda: False)
    monkeypatch.setattr(settings, "_check_ffmpeg", lambda: True)
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        response = client.get("/api/ready")
        assert response.status_code == 503
        assert response.json()["ready"] is False
        monkeypatch.setattr(settings, "get_readiness", lambda: {"ready": True, "checks": {}})
        assert client.get("/api/ready").status_code == 200


def test_gpu_probe_handles_broken_native_import(monkeypatch):
    import builtins
    from web.api.routes import settings

    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "torch":
            raise ValueError("numpy.dtype size changed")
        return original_import(name, *args, **kwargs)

    settings._check_gpu.cache_clear()
    monkeypatch.setattr(builtins, "__import__", import_module)
    assert settings._check_gpu() is False
    settings._check_gpu.cache_clear()
