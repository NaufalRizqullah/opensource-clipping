"""
web.api.store - In-memory job store with JSON file persistence.

Stores all job state in a dict keyed by job ID.
Periodically persists to ``jobs.json`` in the outputs directory.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

from .models import (
    ClipDetail,
    JobProgressEvent,
    JobResponse,
    JobStatus,
    SourcePlatform,
)


_lock = threading.Lock()
_jobs: dict[str, dict] = {}

PERSIST_PATH = os.path.join(os.getcwd(), "outputs", "jobs.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _persist() -> None:
    """Write current job store to disk (best-effort)."""
    try:
        os.makedirs(os.path.dirname(PERSIST_PATH), exist_ok=True)
        serializable = {}
        for jid, job in _jobs.items():
            entry = dict(job)
            for key in ("created_at", "updated_at"):
                if isinstance(entry.get(key), datetime):
                    entry[key] = entry[key].isoformat()
            # Convert progress event
            if entry.get("progress") and isinstance(entry["progress"], JobProgressEvent):
                entry["progress"] = entry["progress"].model_dump()
                if isinstance(entry["progress"].get("timestamp"), datetime):
                    entry["progress"]["timestamp"] = entry["progress"]["timestamp"].isoformat()
            # Convert clip details
            if entry.get("clips"):
                entry["clips"] = [
                    c.model_dump() if isinstance(c, ClipDetail) else c
                    for c in entry["clips"]
                ]
            serializable[jid] = entry
        with open(PERSIST_PATH, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass  # best-effort persistence


def _load() -> None:
    """Load persisted jobs from disk on startup."""
    global _jobs
    if not os.path.exists(PERSIST_PATH):
        return
    try:
        with open(PERSIST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for jid, entry in data.items():
            for key in ("created_at", "updated_at"):
                if isinstance(entry.get(key), str):
                    entry[key] = datetime.fromisoformat(entry[key])
            _jobs[jid] = entry
    except Exception:
        pass


# Load on module import
_load()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def create_job(
    url: Optional[str] = None,
    upload_filename: Optional[str] = None,
    source: str = "youtube",
    config: dict | None = None,
    job_id: str | None = None,
) -> str:
    """Create a new job and return its ID."""
    job_id = job_id or uuid.uuid4().hex[:12]
    now = _now()
    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": JobStatus.QUEUED.value,
            "created_at": now,
            "updated_at": now,
            "url": url,
            "upload_filename": upload_filename,
            "source": source,
            "config": config or {},
            "progress": None,
            "clips": [],
            "error": None,
            "log": [],
        }
        _persist()
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    """Return a job dict or None."""
    with _lock:
        return _jobs.get(job_id)


def list_jobs() -> list[dict]:
    """Return all jobs sorted by creation time (newest first)."""
    with _lock:
        return sorted(
            _jobs.values(),
            key=lambda j: j.get("created_at", _now()),
            reverse=True,
        )


def update_job(job_id: str, **kwargs) -> None:
    """Update arbitrary fields on a job."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.update(kwargs)
        job["updated_at"] = _now()
        _persist()


def update_progress(
    job_id: str,
    step: str,
    step_number: int,
    total_steps: int,
    message: str,
    percent: float = 0.0,
) -> None:
    """Update progress on a running job."""
    event = JobProgressEvent(
        step=step,
        step_number=step_number,
        total_steps=total_steps,
        message=message,
        percent=percent,
    )
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["progress"] = event
        job["updated_at"] = _now()
        # Append to log
        job["log"].append(f"[{step}] {message}")
        if len(job["log"]) > 500:
            job["log"] = job["log"][-500:]
        _persist()


def set_status(job_id: str, status: JobStatus) -> None:
    """Set job status."""
    update_job(job_id, status=status.value)


def set_error(job_id: str, error: str) -> None:
    """Mark job as failed with an error message."""
    update_job(job_id, status=JobStatus.FAILED.value, error=error)


def set_clips(job_id: str, clips: list[ClipDetail]) -> None:
    """Store rendered clip details."""
    update_job(job_id, clips=clips, status=JobStatus.COMPLETED.value)


def delete_job(job_id: str) -> bool:
    """Delete a job. Returns True if found."""
    with _lock:
        if job_id in _jobs:
            del _jobs[job_id]
            _persist()
            return True
        return False


def get_running_count() -> int:
    """Count jobs currently in processing states."""
    processing = {JobStatus.DOWNLOADING.value, JobStatus.TRANSCRIBING.value,
                  JobStatus.ANALYZING.value, JobStatus.RENDERING.value}
    with _lock:
        return sum(1 for j in _jobs.values() if j.get("status") in processing)


def get_queued_count() -> int:
    """Count jobs waiting to start."""
    with _lock:
        return sum(1 for j in _jobs.values() if j.get("status") == JobStatus.QUEUED.value)
