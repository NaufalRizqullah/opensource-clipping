"""Probe the actual clipping runtime without importing native libraries into ASGI.

Liveness is intentionally separate: an API can serve diagnostics even when a
NumPy/MediaPipe binary dependency is broken. No models or credentials are loaded.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROBE_TIMEOUT = 120
_RESULT_PREFIX = "OSC_READINESS="
_cache: dict | None = None
_lock = threading.Lock()

# Run in a fresh interpreter: a failed native import must not poison the API
# process, and installing packages into a running notebook cannot repair its ABI.
_PROBE_CODE = r'''
import importlib
import importlib.metadata
import json

checks = {}
for name, modules in {
    "numpy": ["numpy", "numpy.random"],
    "renderer": ["cv2", "mediapipe.tasks.python.vision", "clipping.studio"],
    "transcription": ["faster_whisper", "clipping.engine"],
}.items():
    try:
        for module in modules:
            importlib.import_module(module)
        checks[name] = {"ok": True}
    except Exception as exc:
        # Never echo arbitrary exception text (URLs, environment or credentials).
        message = str(exc).lower()
        if any(marker in message for marker in (
            "numpy.dtype size changed", "numpy.core.multiarray", "_array_api",
            "compiled using numpy", "binary incompatibility",
        )):
            reason = "Native dependency ABI mismatch; recreate the isolated Kaggle environment."
        elif isinstance(exc, ImportError):
            reason = "A required module or native library could not be imported; rerun dependency setup."
        else:
            reason = "Runtime import failed; recreate the environment and check package compatibility."
        checks[name] = {"ok": False, "error": type(exc).__name__, "message": reason}

versions = {}
for package in ("numpy", "mediapipe", "opencv-python", "faster-whisper", "torch"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        pass
print("OSC_READINESS=" + json.dumps({"checks": checks, "versions": versions}))
'''


def _probe_runtime() -> dict:
    """Run bounded native-import and FFmpeg checks, returning safe diagnostics."""
    checks: dict = {}
    versions: dict = {}
    env = os.environ.copy()
    env["PYTHON_DOTENV_DISABLED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE_CODE],
            cwd=_PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PROBE_TIMEOUT,
            check=False,
        )
        payload = next(
            (line[len(_RESULT_PREFIX):] for line in reversed(result.stdout.splitlines())
             if line.startswith(_RESULT_PREFIX)),
            None,
        )
        if result.returncode != 0 or payload is None:
            raise ValueError("Probe did not finish")
        data = json.loads(payload)
        checks = data["checks"]
        versions = data["versions"]
        if not checks or not all(isinstance(value.get("ok"), bool) for value in checks.values()):
            raise ValueError("Invalid probe result")
    except subprocess.TimeoutExpired:
        checks = {"runtime": {"ok": False, "message": "Dependency import timed out; restart the backend after setup."}}
    except (OSError, ValueError, KeyError, TypeError):
        checks = {"runtime": {"ok": False, "message": "Dependency probe failed; rerun the isolated environment setup."}}

    try:
        ffmpeg = shutil.which("ffmpeg")
        ffmpeg_ok = bool(ffmpeg) and subprocess.run(
            [ffmpeg, "-version"], capture_output=True, timeout=10, check=False,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ffmpeg_ok = False
    checks["ffmpeg"] = {"ok": ffmpeg_ok}
    if not ffmpeg_ok:
        checks["ffmpeg"]["message"] = "FFmpeg is missing or cannot start."
    return {"ready": all(item["ok"] for item in checks.values()), "checks": checks, "versions": versions}


def get_readiness() -> dict:
    """Cache one probe per server process. Restart after dependency changes."""
    global _cache
    with _lock:
        if _cache is None:
            _cache = _probe_runtime()
        return copy.deepcopy(_cache)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check imports and FFmpeg without starting ASGI")
    parser.parse_args()
    result = get_readiness()
    print(json.dumps(result, indent=2))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
