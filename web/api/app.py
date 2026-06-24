"""
web.api.app - FastAPI Application Entry Point

OpenSource Clipping Studio - Web GUI Backend

Run with:
    uvicorn web.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes import jobs, files, settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    print("🚀 OpenSource Clipping Studio - Backend starting...")
    yield
    print("👋 Backend shutting down...")


app = FastAPI(
    title="OpenSource Clipping Studio",
    description="AI Auto-Clipper & Teaser Generator - Web GUI API",
    version="1.0.7",
    lifespan=lifespan,
)

# CORS - allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(jobs.router)
app.include_router(files.router)
app.include_router(settings.router)


@app.get("/")
async def root():
    return {
        "name": "OpenSource Clipping Studio",
        "version": "1.0.7",
        "docs": "/docs",
        "health": "/api/health",
    }
