"""Capture startup failures that occur before ASGI lifespan is available."""

import builtins
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import studio_logging, studio_server


@pytest.fixture
def logfile(monkeypatch, tmp_path):
    studio_logging.close_logging()
    monkeypatch.setenv("OSC_LOG_DIR", str(tmp_path / "logs"))
    yield tmp_path / "logs" / "studio-api.jsonl"
    studio_logging.close_logging()


def test_import_failure_is_saved_before_fastapi_starts(monkeypatch, logfile):
    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "web.api.app":
            raise ValueError("synthetic-private-import-detail")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    assert studio_server.main([]) == 1
    content = logfile.read_text(encoding="utf-8")
    events = [json.loads(line) for line in content.splitlines()]
    failed = events[-1]
    assert failed["event"] == "server.failed"
    assert failed["stage"] == "api_import"
    assert failed["error"]["type"] == "ValueError"
    assert any(frame["function"] == "broken_import" for frame in failed["error"]["traceback"])
    assert "synthetic-private" not in content


@pytest.mark.parametrize("code", [0, 3])
def test_uvicorn_lifecycle_exit_is_preserved_and_logged(monkeypatch, logfile, code):
    def serve(app, **options):
        assert options["host"] == "127.0.0.1"
        assert options["port"] == 8123
        raise SystemExit(code)

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=serve))
    assert studio_server.main(["--port", "8123"]) == code
    events = [json.loads(line) for line in logfile.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["stage"] == "api_serve"
    assert events[-1]["exit_code"] == code
