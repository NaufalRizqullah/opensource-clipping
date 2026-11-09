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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from scripts.studio_logging import configure_logging, log_event

from .routes import jobs, files, settings
from . import worker
from .logging_middleware import RequestLoggingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    log_path = configure_logging(role="api")
    log_event("api.startup.started", component="api", stage="startup")
    print("🚀 OpenSource Clipping Studio — Backend starting...")
    if log_path is not None:
        print(f"Studio diagnostic log: {log_path}", flush=True)
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
    version="1.12.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "https://naufalrizqullah.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
app.add_middleware(RequestLoggingMiddleware)

# Routes
app.include_router(jobs.router)
app.include_router(files.router)
app.include_router(settings.router)


@app.get("/")
async def root():
    return {
        "name": "OpenSource Clipping Studio",
        "version": "1.13.0",
        "docs": "/docs",
        "health": "/api/health",
        "ready": "/api/ready",
    }


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
