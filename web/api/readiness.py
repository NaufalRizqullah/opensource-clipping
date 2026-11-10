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
import time

from scripts.studio_logging import configure_logging, log_event

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
import re


def safe_identifier(value, fallback="unknown", dotted=False):
    pattern = r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*" if dotted else r"[A-Za-z_][A-Za-z_0-9]*"
    return value if isinstance(value, str) and len(value) <= 200 and re.fullmatch(pattern, value) else fallback


def exception_details(exc):
    # Inspect text only to select a fixed explanation. Never return the text:
    # native/library errors can contain URLs, environment values or credentials.
    message = str(exc).lower()
    if any(marker in message for marker in (
        "numpy.dtype size changed", "numpy.core.multiarray", "_array_api",
        "compiled using numpy", "binary incompatibility",
    )):
        code = "numpy_abi_mismatch"
        reason = "Native dependency ABI mismatch; recreate the isolated Kaggle environment."
    elif isinstance(exc, ModuleNotFoundError):
        code = "missing_dependency"
        reason = "A required module or native library could not be imported; rerun dependency setup."
    elif isinstance(exc, ImportError):
        code = "import_error"
        reason = "A required module or native library could not be imported; rerun dependency setup."
    else:
        code = "import_failed"
        reason = "Runtime import failed; inspect the module and traceback locations for package compatibility."

    locations = []
    tb = exc.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        function = frame.f_code.co_name
        locations.append({
            "module": safe_identifier(frame.f_globals.get("__name__"), dotted=True),
            "function": function if function in ("<module>", "<lambda>", "<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>") else safe_identifier(function),
            "line": tb.tb_lineno,
        })
        tb = tb.tb_next
    # Module/function identifiers identify the import chain without exposing
    # filenames, source lines, locals, or the user's filesystem paths.
    return {
        "error": safe_identifier(type(exc).__name__, "Exception"),
        "code": code,
        "message": reason,
        "traceback": locations[-8:],
    }

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
        details = exception_details(exc)
        causes = []
        seen = {id(exc)}
        current = exc
        while len(causes) < 3:
            cause = current.__cause__
            if cause is None and not current.__suppress_context__:
                cause = current.__context__
            if cause is None or id(cause) in seen:
                break
            seen.add(id(cause))
            causes.append(exception_details(cause))
            current = cause
        if causes:
            details["causes"] = causes
        checks[name] = {"ok": False, "module": module, **details}

versions = {}
for package in (
    "numpy", "mediapipe", "opencv-python", "opencv-contrib-python",
    "jax", "jaxlib", "ml-dtypes", "matplotlib", "protobuf",
    "faster-whisper", "torch", "google-genai",
):
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
            started = time.monotonic()
            log_event("readiness.started", component="readiness", stage="readiness")
            _cache = _probe_runtime()
            log_event("readiness.completed" if _cache["ready"] else "readiness.failed",
                      component="readiness", stage="readiness",
                      level="INFO" if _cache["ready"] else "ERROR", diagnostics=_cache,
                      elapsed_ms=(time.monotonic() - started) * 1000)
        return copy.deepcopy(_cache)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check imports and FFmpeg without starting ASGI")
    parser.parse_args()
    configure_logging("api")
    result = get_readiness()
    print(json.dumps(result, indent=2))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
