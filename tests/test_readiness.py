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
    assert report["checks"]["renderer"]["code"] == "numpy_abi_mismatch"
    assert report["checks"]["renderer"]["module"] == "mediapipe.tasks.python.vision"
    assert "ABI mismatch" in report["checks"]["renderer"]["message"]
    assert "private-context" not in output


@pytest.mark.parametrize("failed_module", ["cv2", "mediapipe.tasks.python.vision", "clipping.studio"])
def test_child_probe_reports_import_location_without_messages_paths_or_locals(monkeypatch, capsys, failed_module):
    import importlib
    import importlib.metadata

    namespace = {"__name__": "renderer_dependency.loader"}
    # The synthetic filename, source, locals and exception messages all contain
    # a secret marker. None belongs in the API's public readiness response.
    exec(compile(
        "def load_native():\n"
        "    private_key = 'private-value-must-not-be-echoed'\n"
        "    try:\n"
        "        raise ImportError('library URL token=' + private_key)\n"
        "    except ImportError as exc:\n"
        "        raise ValueError('invalid config: ' + private_key) from exc\n",
        "/private-value-must-not-be-echoed/native.py", "exec",
    ), namespace)

    def import_module(name):
        if name == failed_module:
            namespace["load_native"]()
        return object()

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    exec(readiness._PROBE_CODE, {})
    output = capsys.readouterr().out
    report = json.loads(output.removeprefix(readiness._RESULT_PREFIX))
    renderer = report["checks"]["renderer"]
    assert renderer["module"] == failed_module
    assert renderer["code"] == "import_failed"
    assert renderer["error"] == "ValueError"
    assert renderer["traceback"][-1] == {
        "module": "renderer_dependency.loader", "function": "load_native", "line": 6,
    }
    cause = renderer["causes"][0]
    assert cause["error"] == "ImportError"
    assert cause["code"] == "import_error"
    assert cause["traceback"][-1]["line"] == 4
    assert report["checks"]["transcription"]["ok"] is True
    assert {"jax", "jaxlib", "ml-dtypes", "matplotlib", "protobuf", "opencv-contrib-python"} <= report["versions"].keys()
    assert "private-value" not in output
    assert "private_key" not in output
    assert "native.py" not in output
    assert "invalid config" not in output
    assert "library URL" not in output


def test_child_probe_bounds_cause_chain_and_reports_missing_dependency(monkeypatch, capsys):
    import importlib
    import importlib.metadata

    missing = ModuleNotFoundError("unavailable private module")
    wrapped = ValueError("private wrapper")
    missing.__cause__ = wrapped
    wrapped.__cause__ = missing

    def import_module(name):
        if name == "cv2":
            raise missing
        return object()

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    exec(readiness._PROBE_CODE, {})
    output = capsys.readouterr().out
    renderer = json.loads(output.removeprefix(readiness._RESULT_PREFIX))["checks"]["renderer"]
    assert renderer["code"] == "missing_dependency"
    assert len(renderer["causes"]) == 1
    assert "private" not in output


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
