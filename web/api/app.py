"""
web.api.app — FastAPI Application Entry Point

OpenSource Clipping Studio — Web GUI Backend

Run with:
    uvicorn web.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import os
import signal
import sys

# Windows consoles default to cp1252; the pipeline logs emoji and Indonesian
# text, which would raise UnicodeEncodeError and kill startup or a job.
for _stream in (sys.stdout, sys.stderr):
    try:
        if (getattr(_stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

try:  # Load .env before reading OSC_API_TOKEN; real environment variables win.
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
except ImportError:  # python-dotenv is optional for the API itself
    pass

from scripts.studio_logging import configure_logging, log_event

from .routes import jobs, files, settings
from . import worker
from .logging_middleware import RequestLoggingMiddleware
from . import security
from .version import APP_VERSION

DASHBOARD_DIST = Path(__file__).resolve().parents[1] / "dashboard" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    log_path = configure_logging(role="api")
    log_event("api.startup.started", component="api", stage="startup")
    print("🚀 OpenSource Clipping Studio — Backend starting...")
    if log_path is not None:
        print(f"Studio diagnostic log: {log_path}", flush=True)
    if not security.auth_enabled():
        print(
            "⚠️  OSC_API_TOKEN is not set — the Studio API is open to anyone who can reach it. "
            "Set OSC_API_TOKEN before exposing this server publicly.",
            flush=True,
        )
    log_event("api.startup.completed", component="api", stage="startup")
    try:
        yield
    finally:
        log_event("api.shutdown.started", component="api", stage="shutdown")
        try:
            await worker.cancel_all_jobs()
        except BaseException as exc:
            log_event("api.shutdown.failed", component="api", stage="shutdown", level="ERROR", error=exc)
            raise
        else:
            log_event("api.shutdown.completed", component="api", stage="shutdown")
        finally:
            print("👋 Backend shutting down...")


app = FastAPI(
    title="OpenSource Clipping Studio",
    description="AI Auto-Clipper & Teaser Generator — Web GUI API",
    version=APP_VERSION,
    lifespan=lifespan,
)

_DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
    "https://naufalrizqullah.github.io",
]
_EXTRA_ORIGINS = [o.strip().rstrip("/") for o in os.environ.get("OSC_CORS_ORIGINS", "").split(",") if o.strip()]

# Order matters: the last middleware added is the outermost. CORS wraps the
# security layer so even 401 responses are readable by the browser client.
app.add_middleware(security.SecurityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEFAULT_ORIGINS + _EXTRA_ORIGINS,
    # Auth uses a bearer header, never cookies, so credentials are not needed.
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-OSC-Token", "ngrok-skip-browser-warning", "Accept"],
    expose_headers=["X-Request-ID"],
    max_age=600,
)
app.add_middleware(RequestLoggingMiddleware)

# Routes
app.include_router(jobs.router)
app.include_router(files.router)
app.include_router(settings.router)


_API_INFO = {
    "name": "OpenSource Clipping Studio",
    "version": APP_VERSION,
    "docs": "/docs",
    "health": "/api/health",
    "ready": "/api/ready",
}


@app.get("/api")
async def api_info():
    return {**_API_INFO, "auth_required": security.auth_enabled()}


@app.get("/api/auth/check")
async def auth_check():
    """Reached only after the security middleware accepted the request."""
    return {"authenticated": True, "auth_required": security.auth_enabled()}


@app.post("/api/shutdown")
async def shutdown_server():
    """Trigger graceful shutdown of the FastAPI server."""
    log_event("api.shutdown.requested", component="api", stage="shutdown")
    # Send SIGINT to own process to trigger uvicorn graceful shutdown
    async def _shutdown():
        await asyncio.sleep(0.5)
        os.kill(os.getpid(), signal.SIGINT)
    
    asyncio.create_task(_shutdown())
    return {"status": "shutting down", "message": "Server is stopping..."}


# ---------------------------------------------------------------------------
# Bundled dashboard (web/dashboard/dist) — served same-origin when built.
# ---------------------------------------------------------------------------

def _dashboard_file(path: str) -> Path | None:
    if not DASHBOARD_DIST.is_dir():
        return None
    root = DASHBOARD_DIST.resolve()
    candidate = (root / path).resolve() if path else root / "index.html"
    if candidate.is_file() and root in candidate.parents:
        return candidate
    return root / "index.html" if (root / "index.html").is_file() else None


@app.get("/", include_in_schema=False)
async def root():
    index = _dashboard_file("")
    if index is None:
        return _API_INFO
    return FileResponse(index, headers={"Cache-Control": "no-cache"})


@app.exception_handler(StarletteHTTPException)
async def spa_or_http_error(request: Request, exc: StarletteHTTPException):
    """Serve dashboard files/deep links for unknown non-API GETs; keep API errors intact."""
    path = request.url.path.lstrip("/")
    is_api = path == "api" or path.startswith(("api/", "docs", "openapi.json", "redoc"))
    if exc.status_code == 404 and request.method in {"GET", "HEAD"} and not is_api:
        target = _dashboard_file(path)
        if target is not None:
            cache = "public, max-age=31536000, immutable" if path.startswith("assets/") else "no-cache"
            return FileResponse(target, headers={"Cache-Control": cache})
    return await http_exception_handler(request, exc)
