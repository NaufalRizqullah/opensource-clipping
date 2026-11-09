"""Bounded, structured diagnostics for the managed Studio processes.

Only explicitly supplied operational metadata is recorded. This is deliberately
not a redirect of stdout/stderr or a handler for third-party loggers: those can
contain credentials, provider response bodies and private source URLs.
"""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import re
import sys
import subprocess
import threading
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3
current_request_id: ContextVar[str | None] = ContextVar("studio_request_id", default=None)
job_context: ContextVar[str | None] = ContextVar("studio_job_id", default=None)
_lock = threading.RLock()
_handlers: dict[str, tuple[Path, str, RotatingFileHandler | None]] = {}
_warned = False
_FUNCTION_NAMES = {"<module>", "<lambda>", "<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>"}
_CHECK_NAMES = ("numpy", "renderer", "transcription", "ffmpeg", "runtime")
_IMPORT_MODULES = {"numpy", "numpy.random", "cv2", "mediapipe.tasks.python.vision",
                   "clipping.studio", "faster_whisper", "clipping.engine"}
_CODES = {"numpy_abi_mismatch", "missing_dependency", "import_error", "import_failed",
          "matplotlib_backend", "runtime_error"}
_PACKAGES = ("numpy", "mediapipe", "opencv-python", "opencv-contrib-python", "jax", "jaxlib",
             "ml-dtypes", "matplotlib", "protobuf", "faster-whisper", "torch")


def _symbol(value, *, dotted=False, limit=200) -> str | None:
    pattern = r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*" if dotted else r"[A-Za-z_][A-Za-z_0-9.-]*"
    return value if isinstance(value, str) and len(value) <= limit and re.fullmatch(pattern, value) else None


def _identity(value) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) else None


def _warn_once() -> None:
    global _warned
    if not _warned:
        _warned = True
        try:
            sys.stderr.write("Studio diagnostic file is unavailable; application processing continues.\n")
        except Exception:
            pass


class _SafeRotatingHandler(RotatingFileHandler):
    def handleError(self, record):
        # logging's default handler prints the record and exception to stderr.
        # A full disk or permissions problem must not expose data or mask work.
        _warn_once()


def configure_logging(role: str = "api") -> Path:
    """Enable append-only rotating diagnostics and share correlation with children.

    One managed process owns each role's file. Failure to open/write a diagnostic
    file is reported with a fixed warning and never prevents application work.
    """
    if role not in {"api", "bootstrap"}:
        raise ValueError("Unknown Studio logging role")
    with _lock:
        directory = Path(os.environ.get("OSC_LOG_DIR") or PROJECT_ROOT / "logs")
        path = directory / ("kaggle-studio.jsonl" if role == "bootstrap" else "studio-api.jsonl")
        run_id = os.environ.get("OSC_LOG_RUN_ID", "")
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            run_id = uuid.uuid4().hex
        os.environ["OSC_LOG_DIR"] = str(directory)
        os.environ["OSC_LOG_RUN_ID"] = run_id
        existing = _handlers.get(role)
        if existing and existing[:2] == (path, run_id):
            return path
        if existing and existing[2]:
            try:
                existing[2].close()
            except Exception:
                pass
        handler = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            handler = _SafeRotatingHandler(
                path, mode="a", maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT,
                encoding="utf-8", errors="backslashreplace",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
        except Exception:
            _warn_once()
        _handlers[role] = (path, run_id, handler)
        return path


def close_logging() -> None:
    """Release owned file handles; useful for clean test/process shutdown."""
    global _warned
    with _lock:
        for _, _, handler in _handlers.values():
            if handler:
                try:
                    handler.close()
                except Exception:
                    pass
        _handlers.clear()
        _warned = False


def _frames(frames) -> list[dict]:
    result = []
    if not isinstance(frames, list):
        return result
    for frame in frames[-15:]:
        if not isinstance(frame, dict):
            continue
        module = _symbol(frame.get("module"), dotted=True)
        function = frame.get("function")
        function = function if isinstance(function, str) and function in _FUNCTION_NAMES else _symbol(function)
        line = frame.get("line")
        if module and function and type(line) is int and 0 < line < 10000000:
            result.append({"module": module, "function": function, "line": line})
    return result


def _error_details(error: BaseException) -> dict:
    frames = []
    tb = error.__traceback__
    while tb is not None:
        frames.append({"module": tb.tb_frame.f_globals.get("__name__", "unknown"),
                       "function": tb.tb_frame.f_code.co_name, "line": tb.tb_lineno})
        tb = tb.tb_next
    # Select fixed categories only; never store the exception's raw message.
    try:
        text = str(error).lower()
    except Exception:
        text = ""
    if any(marker in text for marker in ("numpy.dtype size changed", "numpy.core.multiarray",
                                         "_array_api", "compiled using numpy", "binary incompatibility")):
        code = "numpy_abi_mismatch"
    elif "backend" in text and ("matplotlib_inline" in text or "not a valid value for backend" in text):
        code = "matplotlib_backend"
    elif isinstance(error, ModuleNotFoundError):
        code = "missing_dependency"
    elif isinstance(error, ImportError):
        code = "import_error"
    else:
        code = "runtime_error"
    result = {"type": _symbol(type(error).__name__) or "Exception", "code": code, "traceback": _frames(frames)}
    if isinstance(error, subprocess.CalledProcessError) and type(error.returncode) is int:
        result["exit_code"] = error.returncode
    if isinstance(error, OSError) and type(error.errno) is int:
        result["errno"] = error.errno
    return result


def _exception(error: BaseException) -> dict:
    result = _error_details(error)
    seen = {id(error)}
    causes = []
    current = error
    while len(causes) < 3:
        cause = current.__cause__
        if cause is None and not current.__suppress_context__:
            cause = current.__context__
        if cause is None or id(cause) in seen:
            break
        seen.add(id(cause))
        causes.append(_error_details(cause))
        current = cause
    if causes:
        result["causes"] = causes
    return result


def _probe_check(check: dict) -> dict:
    result = {}
    if type(check.get("ok")) is bool:
        result["ok"] = check["ok"]
    error = _symbol(check.get("error"))
    if error:
        result["error"] = error
    module, code = check.get("module"), check.get("code")
    if isinstance(module, str) and module in _IMPORT_MODULES:
        result["module"] = module
    if isinstance(code, str) and code in _CODES:
        result["code"] = code
    frames = _frames(check.get("traceback"))
    if frames:
        result["traceback"] = frames
    return result


def _diagnostics(report) -> dict:
    result = {}
    if not isinstance(report, dict):
        return result
    if type(report.get("ready")) is bool:
        result["ready"] = report["ready"]
    checks = report.get("checks")
    if isinstance(checks, dict):
        result["checks"] = {}
        for name in _CHECK_NAMES:
            check = checks.get(name)
            if isinstance(check, dict):
                result["checks"][name] = safe = _probe_check(check)
                causes = check.get("causes")
                if isinstance(causes, list):
                    safe["causes"] = [_probe_check(cause) for cause in causes[:3] if isinstance(cause, dict)]
    versions = report.get("versions")
    if isinstance(versions, dict):
        result["versions"] = {}
        for name in _PACKAGES:
            version = versions.get(name)
            if isinstance(version, str) and re.fullmatch(r"[0-9][0-9A-Za-z.+-]{0,39}", version):
                result["versions"][name] = version
    return result


def log_event(event: str, *, component: str = "api", level: str = "INFO", stage=None,
              job_id=None, request_id=None, elapsed_ms=None, error: BaseException | None = None,
              status=None, count=None, clip_index=None, clip_rank=None, exit_code=None, method=None,
              route=None, status_code=None, diagnostics=None) -> None:
    """Record a safe event; never let diagnostic failures replace work outcomes.

    Identifiers must be application-generated. Routes must be route templates,
    never request URLs. Payloads, settings, headers and exception text have no
    field in this schema. Nothing is written for an unconfigured CLI process.
    """
    try:
        with _lock:
            if not _handlers and not os.environ.get("OSC_LOG_DIR"):
                return
            role = "bootstrap" if component == "bootstrap" else "api"
            configure_logging(role)
            _, run_id, handler = _handlers[role]
            if handler is None:
                return
            record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                      "level": level if level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} else "INFO",
                      "event": _symbol(event, limit=80) or "unknown", "component": _symbol(component, limit=80) or "unknown",
                      "run_id": run_id, "pid": os.getpid(), "thread_id": threading.get_ident()}
            for name, value in (("stage", stage), ("status", status)):
                if safe := _symbol(value, limit=80):
                    record[name] = safe
            for name, value in (("job_id", job_id if job_id is not None else job_context.get()),
                                ("request_id", request_id if request_id is not None else current_request_id.get())):
                if safe := _identity(value):
                    record[name] = safe
            for name, value in (("count", count), ("clip_index", clip_index), ("clip_rank", clip_rank),
                                ("exit_code", exit_code), ("status_code", status_code)):
                if type(value) is int:
                    record[name] = value
            if type(elapsed_ms) in (int, float) and math.isfinite(elapsed_ms) and elapsed_ms >= 0:
                record["elapsed_ms"] = round(elapsed_ms, 3)
            if isinstance(method, str) and method in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"}:
                record["method"] = method
            if isinstance(route, str) and len(route) <= 200 and (route == "unmatched" or re.fullmatch(r"/[A-Za-z0-9_./{}:-]*", route)):
                record["route"] = route
            if isinstance(error, BaseException):
                record["error"] = _exception(error)
            if diagnostics is not None:
                record["diagnostics"] = _diagnostics(diagnostics)
            encoded = json.dumps(record, ensure_ascii=True, allow_nan=False)
            handler.handle(logging.LogRecord("studio.events", logging.INFO, "", 0, encoded, (), None))
    except Exception:
        _warn_once()
