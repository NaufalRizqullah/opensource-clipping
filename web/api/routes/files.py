"""
web.api.routes.files — File upload and output serving endpoints.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

router = APIRouter(tags=["files"])

# Resolve absolute path to the project root (2 levels up from web/api/routes)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Max upload size: 2GB
MAX_UPLOAD_SIZE = 2 * 1024 * 1024 * 1024


def _output_path(job_id: str, filename: str | None = None) -> Path:
    """Keep output lookup inside one job, including on Windows and symlinks."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    root = Path(OUTPUTS_DIR).resolve()
    job_dir = (root / job_id).resolve()
    if job_dir.parent != root:
        raise HTTPException(status_code=400, detail="Invalid job output directory")
    if filename is None:
        return job_dir
    if not filename or filename in {".", ".."} or any(ch in filename for ch in '/\\:\x00'):
        raise HTTPException(status_code=400, detail="Invalid filename")
    candidate = (job_dir / filename).resolve()
    if candidate.parent != job_dir:
        raise HTTPException(status_code=400, detail="Invalid output path")
    return candidate


@router.post("/api/upload")
async def upload_video(file: UploadFile = File(...)) -> dict:
    """
    Upload a video file for processing.

    Returns the stored filename that can be used in ``upload_filename``
    when creating a job.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validate extension
    allowed_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".ts"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_exts)}",
        )

    # Sanitize filename
    safe_name = file.filename.replace(" ", "_")
    for ch in '<>:"/\\|?*#':
        safe_name = safe_name.replace(ch, "")

    safe_name = "".join(ch for ch in safe_name if ch.isprintable())[-180:]
    safe_name = f"{uuid.uuid4().hex[:12]}_{safe_name}"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, safe_name)

    # Stream write to disk
    total_written = 0
    try:
        with open(dest, "xb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                total_written += len(chunk)
                if total_written > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024**3)}GB.",
                    )
                f.write(chunk)
        if not total_written:
            raise HTTPException(status_code=400, detail="Uploaded video is empty")
    except BaseException:
        if os.path.exists(dest):
            os.remove(dest)
        raise

    size_mb = total_written / (1024 * 1024)
    return {
        "filename": safe_name,
        "size_mb": round(size_mb, 2),
        "message": f"Upload berhasil: {safe_name} ({size_mb:.1f} MB)",
    }


@router.get("/api/outputs/{job_id}/{filename}")
async def serve_output(job_id: str, filename: str):
    """Serve a rendered clip or other output file."""
    file_path = _output_path(job_id, filename)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type
    ext = os.path.splitext(filename)[1].lower()
    media_types = {
        ".mp4": "video/mp4",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".webm": "video/webm",
        ".json": "application/json",
        ".ass": "text/plain",
        ".srt": "text/plain",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename,
    )


@router.get("/api/outputs/{job_id}")
async def list_outputs(job_id: str) -> dict:
    """List all output files for a job."""
    job_dir = _output_path(job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Job output directory not found")

    files = []
    for fname in sorted(os.listdir(job_dir)):
        try:
            fpath = _output_path(job_id, fname)
        except HTTPException:
            continue
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            files.append({
                "filename": fname,
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 2),
                "download_url": f"/api/outputs/{quote(job_id, safe='')}/{quote(fname, safe='')}",
            })

    return {"job_id": job_id, "files": files, "total": len(files)}
