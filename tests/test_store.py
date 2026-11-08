"""Regression tests for durable job state and cancellation publication."""

import json

import pytest

from web.api import store
from web.api.models import ClipDetail, JobStatus


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "PERSIST_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setattr(store, "_jobs", {})


def test_restart_marks_active_jobs_interrupted_and_preserves_terminal_jobs():
    active = store.create_job()
    complete = store.create_job()
    store.set_status(complete, JobStatus.COMPLETED)
    store._jobs.clear()

    store._load()

    assert store.get_job(active)["status"] == "failed"
    assert "restarted" in store.get_job(active)["error"]
    assert store.get_job(complete)["status"] == "completed"
    with open(store.PERSIST_PATH, encoding="utf-8") as handle:
        assert json.load(handle)[active]["status"] == "failed"


def test_failed_atomic_replace_preserves_previous_snapshot(monkeypatch, caplog):
    job_id = store.create_job()
    with open(store.PERSIST_PATH, encoding="utf-8") as handle:
        original = handle.read()

    def fail_replace(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(store.os, "replace", fail_replace)
    store.set_status(job_id, JobStatus.RENDERING)

    with open(store.PERSIST_PATH, encoding="utf-8") as handle:
        assert handle.read() == original
    assert "Could not persist" in caplog.text


def test_cancellation_wins_over_success_and_record_cannot_disappear_while_running():
    job_id = store.create_job()
    store.set_status(job_id, JobStatus.RENDERING)
    assert store.request_cancel(job_id)
    assert store.get_job(job_id)["status"] == "rendering"
    assert not store.delete_job(job_id)
    clips = [ClipDetail(rank=1, filename="one.mp4", download_url="/one.mp4")]
    store.finish_job(job_id, clips=clips)
    job = store.get_job(job_id)
    assert job["status"] == "cancelled"
    assert job["clips"] == clips


def test_job_snapshots_cannot_mutate_store_and_active_reuse_is_rejected():
    job_id = store.create_job()
    copy = store.get_job(job_id)
    copy["log"].append("external change")
    assert store.get_job(job_id)["log"] == []
    with pytest.raises(ValueError, match="still active"):
        store.create_job(job_id=job_id)
