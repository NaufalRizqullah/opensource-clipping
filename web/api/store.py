"""
web.api.store — In-memory job store with JSON file persistence.

Stores all job state in a dict keyed by job ID.
Periodically persists to ``jobs.json`` in the outputs directory.
"""

from __future__ import annotations

import json
import copy
import logging
import os
import tempfile
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

# Resolve absolute path to the project root (2 levels up from web/api)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PERSIST_PATH = os.path.join(PROJECT_ROOT, "outputs", "jobs.json")
PERSIST_PATH = os.environ.get("OSC_JOB_STORE_PATH", PERSIST_PATH)
logger = logging.getLogger(__name__)
TERMINAL_STATES = {
    JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _persist() -> None:
    """Atomically replace the snapshot; callers hold the store lock."""
    temporary_path = None
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
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=os.path.dirname(PERSIST_PATH),
            prefix=".jobs-", suffix=".tmp", delete=False,
        ) as f:
            temporary_path = f.name
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, PERSIST_PATH)
    except Exception as exc:
        logger.error("Could not persist job state (%s)", type(exc).__name__)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                logger.warning("Could not remove a temporary job snapshot")


def _load() -> None:
    """Load persisted jobs from disk on startup."""
    global _jobs
    if not os.path.exists(PERSIST_PATH):
        return
    try:
        with open(PERSIST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        recovered = False
        for jid, entry in data.items():
            for key in ("created_at", "updated_at"):
                if isinstance(entry.get(key), str):
                    entry[key] = datetime.fromisoformat(entry[key])
            if entry.get("status") not in TERMINAL_STATES:
                entry["status"] = JobStatus.FAILED.value
                entry["error"] = "Backend restarted before this job finished. Reuse the job to retry."
                entry["updated_at"] = _now()
                entry["cancel_requested"] = False
                entry["progress"] = JobProgressEvent(
                    step="error", step_number=0, total_steps=7,
                    message=entry["error"], percent=0.0,
                )
                entry.setdefault("log", []).append("[error] " + entry["error"])
                recovered = True
            _jobs[jid] = entry
        if recovered:
            _persist()
    except Exception as exc:
        logger.error("Could not load job state (%s)", type(exc).__name__)


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
        existing = _jobs.get(job_id)
        if existing and existing.get("status") not in TERMINAL_STATES:
            raise ValueError("Job is still active")
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
            "cancel_requested": False,
            "log": [],
        }
        _persist()
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    """Return a job dict or None."""
    with _lock:
        return copy.deepcopy(_jobs.get(job_id))


def list_jobs() -> list[dict]:
    """Return all jobs sorted by creation time (newest first)."""
    with _lock:
        return copy.deepcopy(sorted(
            _jobs.values(),
            key=lambda j: j.get("created_at", _now()),
            reverse=True,
        ))


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


def request_cancel(job_id: str) -> bool:
    """Record cancellation without claiming an active thread has stopped."""
    with _lock:
        job = _jobs.get(job_id)
        if not job or job.get("status") in TERMINAL_STATES:
            return False
        job["cancel_requested"] = True
        job["updated_at"] = _now()
        job.setdefault("log", []).append(
            "[cancel] Cancellation requested; waiting for the current operation to finish."
        )
        _persist()
        return True


def is_cancel_requested(job_id: str) -> bool:
    """Return the cooperative cancellation flag for a worker."""
    with _lock:
        job = _jobs.get(job_id)
        return job is None or bool(job.get("cancel_requested"))


def finish_job(
    job_id: str, clips: list[ClipDetail] | None = None, error: str | None = None,
) -> None:
    """Publish one consistent terminal snapshot, honoring a concurrent cancel."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        if clips is not None:
            job["clips"] = clips
        if job.get("cancel_requested"):
            status, step, message = JobStatus.CANCELLED, "cancelled", "Job dibatalkan."
            error = None
        elif error:
            status, step, message = JobStatus.FAILED, "error", error
        else:
            status, step = JobStatus.COMPLETED, "done"
            message = f"Selesai! {len(job.get('clips', []))} klip berhasil dirender."
        job.update(
            status=status.value, error=error, updated_at=_now(),
            progress=JobProgressEvent(
                step=step, step_number=7 if status == JobStatus.COMPLETED else 0,
                total_steps=7, message=message,
                percent=100.0 if status == JobStatus.COMPLETED else 0.0,
            ),
        )
        job.setdefault("log", []).append(f"[{step}] {message}")
        _persist()


def delete_job(job_id: str) -> bool:
    """Delete a job. Returns True if found."""
    with _lock:
        if job_id in _jobs and _jobs[job_id].get("status") in TERMINAL_STATES:
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
