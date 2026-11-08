"""API contract regressions across job lifecycle, SSE, uploads and media serving."""

import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from web.api.app import app
from web.api import config_adapter, store, worker
from web.api.models import ClipDetail, JobStatus
from web.api.routes import files, jobs


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "PERSIST_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setattr(store, "_jobs", {})
    monkeypatch.setattr(config_adapter, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(files, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(files, "OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr(worker, "_tasks", {})
    monkeypatch.setattr(worker, "_running_jobs", set())
    monkeypatch.setattr(jobs, "get_readiness", lambda: {"ready": True})
    submitted = []
    async def submit(job_id, payload): submitted.append((job_id, payload))
    monkeypatch.setattr(worker, "submit_job", submit)
    with TestClient(app) as client:
        yield client, tmp_path, submitted


def previous_job(root):
    job_id = store.create_job(url="https://example.test/video", source="youtube")
    folder = root / "outputs" / job_id
    folder.mkdir(parents=True)
    (folder / "video_asli.mp4").write_bytes(b"video")
    (folder / "gemini_response.json").write_text("[]")
    store.finish_job(job_id)
    return job_id


def test_unready_runtime_does_not_create_or_submit_job(api, monkeypatch):
    client, _, submitted = api
    monkeypatch.setattr(jobs, "get_readiness", lambda: {"ready": False})
    response = client.post("/api/jobs", json={"url": "https://example.test/video"})
    assert response.status_code == 503
    assert store.list_jobs() == [] and submitted == []


@pytest.mark.parametrize("bypass", [True, False])
def test_reuse_preserves_source_and_honors_bypass_toggle(api, bypass):
    client, root, submitted = api
    job_id = previous_job(root)
    response = client.post("/api/jobs", json={"reuse_job_id": job_id, "load_gemini_json": bypass})
    assert response.status_code == 201, response.text
    assert response.json()["url"] == "https://example.test/video"
    assert submitted[0][1]["load_gemini_json"] is bypass
    assert submitted[0][1]["url"] is None  # Reuse the original local source.


def test_reuse_missing_cache_and_active_job_cannot_overwrite_history(api):
    client, root, _ = api
    job_id = previous_job(root)
    (root / "outputs" / job_id / "gemini_response.json").unlink()
    assert client.post("/api/jobs", json={"reuse_job_id": job_id}).status_code == 400
    assert store.get_job(job_id)["status"] == "completed"
    store.set_status(job_id, JobStatus.RENDERING)
    assert client.post("/api/jobs", json={"reuse_job_id": job_id, "load_gemini_json": False}).status_code == 409
    assert store.get_job(job_id)["status"] == "rendering"


@pytest.mark.parametrize("payload", [
    {"reuse_job_id": "../escape"},
    {"upload_filename": "../video.mp4"},
    {"upload_filename": "C:\\video.mp4"},
])
def test_job_source_paths_cannot_escape_storage(api, payload):
    client, _, submitted = api
    assert client.post("/api/jobs", json=payload).status_code == 422
    assert submitted == []


def test_cancel_retains_running_job_until_safe_completion_then_allows_delete(api):
    client, _, _ = api
    job_id = store.create_job()
    store.set_status(job_id, JobStatus.RENDERING)
    worker._running_jobs.add(job_id)
    response = client.delete(f"/api/jobs/{job_id}")
    assert response.status_code == 200 and response.json()["cancel_requested"] is True
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "rendering"
    store.finish_job(job_id)
    assert client.delete(f"/api/jobs/{job_id}").status_code == 409
    worker._running_jobs.clear()
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "cancelled"
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_terminal_sse_delivers_clips_in_its_first_event(api):
    client, _, _ = api
    job_id = store.create_job()
    store.finish_job(job_id, clips=[ClipDetail(rank=1, filename="clip.mp4", download_url=f"/api/outputs/{job_id}/clip.mp4")])
    response = client.get(f"/api/jobs/{job_id}/status")
    frames = [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data:")]
    assert len(frames) == 1
    assert frames[0]["type"] == "completed"
    assert frames[0]["clips"][0]["filename"] == "clip.mp4"


def test_emitted_unicode_download_url_is_served_and_supports_video_ranges(api):
    client, root, _ = api
    job_id = previous_job(root)
    (root / "outputs" / job_id / "clip #你好.mp4").write_bytes(b"0123456789")
    listing = client.get(f"/api/outputs/{job_id}").json()
    link = next(item["download_url"] for item in listing["files"] if item["filename"].startswith("clip"))
    assert "%23" in link
    assert client.get(link).content == b"0123456789"
    partial = client.get(link, headers={"Range": "bytes=2-5"})
    assert partial.status_code == 206
    assert partial.content == b"2345"


def test_uploads_do_not_overwrite_and_failed_uploads_are_removed(api, monkeypatch):
    client, root, _ = api
    first = client.post("/api/upload", files={"file": ("podcast.webm", b"one", "video/webm")})
    second = client.post("/api/upload", files={"file": ("podcast.webm", b"two", "video/webm")})
    assert first.status_code == second.status_code == 200
    assert first.json()["filename"] != second.json()["filename"]
    monkeypatch.setattr(files, "MAX_UPLOAD_SIZE", 1)
    assert client.post("/api/upload", files={"file": ("large.webm", b"too large", "video/webm")}).status_code == 413
    assert sorted(path.read_bytes() for path in (root / "uploads").iterdir()) == [b"one", b"two"]
