"""Smoke-test the real Uvicorn entry point without loading models or secrets."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest


@pytest.mark.parametrize("entrypoint", [["uvicorn", "web.api.app:app"], ["scripts.studio_server"]])
def test_uvicorn_starts_in_fresh_interpreter(tmp_path, entrypoint):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    environment = {
        **os.environ,
        "PYTHON_DOTENV_DISABLED": "1",
        "PYTHONIOENCODING": "utf-8",
        "OSC_JOB_STORE_PATH": str(tmp_path / "jobs.json"),
        "OSC_LOG_DIR": str(tmp_path / "logs"),
        "OSC_LOG_RUN_ID": "a" * 32,
    }
    process = subprocess.Popen(
        [sys.executable, "-m", *entrypoint, "--host", "127.0.0.1", "--port", str(port)],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8",
    )
    try:
        deadline = time.monotonic() + 20
        while True:
            if process.poll() is not None:
                raise AssertionError(f"Uvicorn failed during startup: {process.stderr.read()}")
            try:
                with urlopen(f"http://127.0.0.1:{port}/openapi.json", timeout=2) as response:
                    schema = json.load(response)
                    assert response.headers.get("X-Request-ID")
                break
            except (URLError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise AssertionError("Uvicorn did not become reachable")
                time.sleep(0.1)
        assert "/api/ready" in schema["paths"]
        assert "/api/jobs" in schema["paths"]
        assert not (tmp_path / "jobs.json").exists()
        log_path = tmp_path / "logs" / "studio-api.jsonl"
        events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert any(event["event"] == "api.startup.completed" for event in events)
        assert all(event["run_id"] == "a" * 32 for event in events)
        if entrypoint == ["scripts.studio_server"]:
            assert any(event["event"] == "server.import.completed" for event in events)
    finally:
        process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
