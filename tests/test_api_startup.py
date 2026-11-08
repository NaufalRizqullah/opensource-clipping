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


def test_uvicorn_starts_in_fresh_interpreter(tmp_path):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    environment = {
        **os.environ,
        "PYTHON_DOTENV_DISABLED": "1",
        "PYTHONIOENCODING": "utf-8",
        "OSC_JOB_STORE_PATH": str(tmp_path / "jobs.json"),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "web.api.app:app", "--host", "127.0.0.1", "--port", str(port)],
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
                break
            except (URLError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise AssertionError("Uvicorn did not become reachable")
                time.sleep(0.1)
        assert "/api/ready" in schema["paths"]
        assert "/api/jobs" in schema["paths"]
        assert not (tmp_path / "jobs.json").exists()
    finally:
        process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
