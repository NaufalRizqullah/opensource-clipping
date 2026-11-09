"""Exercise orchestration with deterministic adapters instead of GPU/network calls."""

import asyncio
import ast
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import clipping
from web.api import config_adapter, store, worker


@pytest.fixture
def worker_events(monkeypatch):
    events = []
    monkeypatch.setattr(worker, "log_event", lambda event, **fields: events.append({"event": event, **fields}))
    return events


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "PERSIST_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setattr(store, "_jobs", {})
    monkeypatch.setattr(config_adapter, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(worker, "_settings_env", {"GOOGLE_API_KEY": "test-key", "NVIDIA_API_KEY": ""})
    calls = SimpleNamespace(download=[], transcribe=[], analyze=[], render=[], extract=[])
    state = SimpleNamespace(clips=[{"rank": 1}], results=None, on_download=None, on_render=None)

    def download(url, destination, *args, **kwargs):
        calls.download.append(kwargs)
        Path(destination).write_bytes(b"video")
        if state.on_download:
            state.on_download()

    def transcribe(*args, **kwargs):
        calls.transcribe.append(args)
        return "transcript", [{"text": "hello"}]

    def analyze(*args):
        calls.analyze.append(args)
        return state.clips

    def render(rank, clip, ratio, glitch, segments, cfg, encoder, **kwargs):
        calls.render.append(rank)
        path = Path(cfg.outputs_dir) / f"clip #{rank}.mp4"
        path.write_bytes(b"clip")
        result = {"rank": rank, "status": "success", "video_exists": True, "video_path": str(path)}
        if state.results is not None:
            result = state.results[len(calls.render) - 1]
        if state.on_render:
            state.on_render()
        return result

    def extract(video, audio):
        calls.extract.append((video, audio))
        Path(audio).write_bytes(b"audio")

    substitutes = {
        "engine": SimpleNamespace(download_video=download, transcribe_video=transcribe, analyze_with_ai=analyze),
        "metadata": SimpleNamespace(normalize_and_validate=lambda data: data, save_metadata_preview=lambda *a, **k: None),
        "studio": SimpleNamespace(
            _is_vertical_ratio=lambda ratio: ratio == "9:16", _get_render_dims=lambda *a, **k: (1080, 1920),
            detect_video_encoder=lambda *a, **k: "libx264", proses_klip=render,
        ),
        "diarization": SimpleNamespace(extract_audio=extract, run_diarization=lambda *a, **k: []),
        "hook_manager": SimpleNamespace(),
    }
    for name, module in substitutes.items():
        monkeypatch.setattr(clipping, name, module, raising=False)
        monkeypatch.setitem(sys.modules, f"clipping.{name}", module)
    cap = SimpleNamespace(get=lambda *a: 1080, release=lambda: None)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=lambda *a: cap, CAP_PROP_FRAME_HEIGHT=4))
    payload = {"url": "https://example.test/video", "use_hook_glitch": False}

    def run(**overrides):
        data = dict(payload, **overrides)
        job_id = store.create_job(config=data)
        worker._run_pipeline_sync(job_id, data)
        return store.get_job(job_id)

    return SimpleNamespace(root=tmp_path, calls=calls, state=state, run=run, payload=payload)


def test_passes_detected_cookie_file_to_downloader(pipeline):
    (pipeline.root / "youtube_cookies.txt").touch()  # No cookie content is read.
    job = pipeline.run()
    assert job["status"] == "completed"
    assert pipeline.calls.download[0]["yt_cookies"] == str(pipeline.root / "youtube_cookies.txt")
    assert job["clips"][0].download_url.endswith("clip%20%231.mp4")


def test_nvidia_only_credentials_can_run_without_google_key(pipeline, monkeypatch):
    monkeypatch.setattr(worker, "_settings_env", {"GOOGLE_API_KEY": "", "NVIDIA_API_KEY": "test-key"})
    assert pipeline.run(ai_provider="nvidia")["status"] == "completed"
    assert len(pipeline.calls.analyze) == 1


def test_cached_analysis_bypasses_provider_credentials_and_api_call(pipeline, monkeypatch):
    monkeypatch.setattr(worker, "_settings_env", {"GOOGLE_API_KEY": "", "NVIDIA_API_KEY": ""})
    job_id = store.create_job()
    output = pipeline.root / "outputs" / job_id
    output.mkdir(parents=True)
    (output / "video_asli.mp4").write_bytes(b"video")
    (output / "gemini_response.json").write_text('[{"rank": 1}]', encoding="utf-8")
    worker._run_pipeline_sync(job_id, {"load_gemini_json": True, "use_hook_glitch": False})
    assert store.get_job(job_id)["status"] == "completed"
    assert pipeline.calls.analyze == []
    assert pipeline.calls.download == []


def test_missing_cached_json_does_not_silently_spend_an_api_call(pipeline):
    job = pipeline.run(load_gemini_json=True)
    assert job["status"] == "failed"
    assert "Cached AI JSON" in job["error"]
    assert pipeline.calls.download == pipeline.calls.analyze == []


def test_failed_render_is_not_reported_as_a_completed_clip(pipeline):
    pipeline.state.clips = [{"rank": 1}, {"rank": 2}]
    pipeline.state.results = [
        {"rank": 1, "status": "success", "video_exists": True, "video_path": "one.mp4"},
        {"rank": 2, "status": "failed", "video_exists": True, "video_path": "broken.mp4", "error": "FFmpeg failed"},
    ]
    job = pipeline.run()
    assert job["status"] == "failed"
    assert "1/2" in job["error"]
    assert [clip.filename for clip in job["clips"]] == ["one.mp4"]
    manifest = json.loads((pipeline.root / "outputs" / job["id"] / "render_manifest.json").read_text())
    assert len(manifest) == 2


def test_non_mp4_upload_diarization_never_uses_video_as_audio_destination(pipeline):
    source = pipeline.root / "uploads" / "podcast.webm"
    source.parent.mkdir()
    source.write_bytes(b"original video")
    job = pipeline.run(url=None, upload_filename=source.name, use_camera_switch=True, diarization_speakers=2)
    assert job["status"] == "completed"
    video, audio = pipeline.calls.extract[0]
    assert Path(video) == source
    assert Path(audio).parent == pipeline.root / "outputs" / job["id"]
    assert audio != video
    assert source.read_bytes() == b"original video"


def test_invalid_renderer_metadata_still_publishes_failed_state(pipeline):
    pipeline.state.results = [{
        "rank": 1, "status": "success", "video_exists": True,
        "video_path": "one.mp4", "duration": "not-a-duration",
    }]
    job = pipeline.run()
    assert job["status"] == "failed"
    assert job["clips"] == []


def test_cancel_after_download_prevents_transcription(pipeline):
    job_id = store.create_job()
    pipeline.state.on_download = lambda: store.request_cancel(job_id)
    worker._run_pipeline_sync(job_id, pipeline.payload)
    assert store.get_job(job_id)["status"] == "cancelled"
    assert pipeline.calls.transcribe == []


def test_cancel_after_first_render_retains_success_and_stops_next_clip(pipeline):
    job_id = store.create_job()
    pipeline.state.clips = [{"rank": 1}, {"rank": 2}]
    pipeline.state.on_render = lambda: store.request_cancel(job_id)
    worker._run_pipeline_sync(job_id, pipeline.payload)
    job = store.get_job(job_id)
    assert job["status"] == "cancelled"
    assert len(job["clips"]) == 1
    assert pipeline.calls.render == [1]


def test_queued_cancel_never_starts_and_running_cancel_waits_for_safe_boundary(pipeline, monkeypatch, worker_events):
    started = threading.Event()
    release = threading.Event()
    calls = []

    def run(job_id, payload):
        calls.append(job_id)
        started.set()
        assert release.wait(5)
        store.finish_job(job_id)

    monkeypatch.setattr(worker, "_run_pipeline_sync", run)
    monkeypatch.setattr(worker, "_tasks", {})
    monkeypatch.setattr(worker, "_running_jobs", set())
    monkeypatch.setattr(worker, "_semaphore", asyncio.Semaphore(1))

    async def scenario():
        first, second = store.create_job(), store.create_job()
        await worker.submit_job(first, {})
        await worker.submit_job(second, {})
        try:
            assert await asyncio.to_thread(started.wait, 3)
            assert worker.cancel_job(second)
            assert store.get_job(second)["status"] == "cancelled"
            assert worker.cancel_job(first)
            assert store.get_job(first)["status"] != "cancelled"
            assert worker.is_job_active(first)
        finally:
            release.set()
            await asyncio.gather(*list(worker._tasks.values()), return_exceptions=True)
        assert calls == [first]
        assert store.get_job(first)["status"] == "cancelled"
        assert [event["job_id"] for event in worker_events if event["event"] == "job_queued"] == [first, second]
        assert [event["job_id"] for event in worker_events if event["event"] == "job_cancel_requested"] == [second, first]
        assert any(event["event"] == "job_cancelled" and event["job_id"] == second and event["stage"] == "queue" for event in worker_events)

    asyncio.run(scenario())


def test_pipeline_trace_records_ordered_stages_and_clip_result(pipeline, worker_events):
    job = pipeline.run()
    assert job["status"] == "completed"
    started = [event["stage"] for event in worker_events if event["event"] == "stage_started"]
    completed = [event["stage"] for event in worker_events if event["event"] == "stage_completed"]
    assert started == completed == [
        "configure", "download", "transcribe", "analyze", "metadata",
        "renderer_import", "render_prepare", "render", "finalize",
    ]
    assert all(event["job_id"] == job["id"] for event in worker_events)
    assert any(event["event"] == "stage_skipped" and event["stage"] == "diarization" for event in worker_events)
    assert [event["event"] for event in worker_events if event["event"].startswith("clip_")] == ["clip_started", "clip_completed"]
    assert worker_events[-1]["event"] == "job_completed"
    assert worker_events[-1]["elapsed_ms"] >= 0
    assert worker.job_context.get() is None
    encoded = json.dumps(worker_events)
    assert "https://example.test/video" not in encoded
    assert "test-key" not in encoded
    assert "transcript" not in encoded


def test_pipeline_trace_marks_failed_stage_with_original_exception(pipeline, monkeypatch, worker_events):
    failure = ValueError("private provider response must not become a log message")

    def transcribe(*args, **kwargs):
        raise failure

    monkeypatch.setattr(clipping.engine, "transcribe_video", transcribe)
    job = pipeline.run()
    assert job["status"] == "failed"
    failed = [event for event in worker_events if event["event"] == "stage_failed"]
    assert len(failed) == 1
    assert failed[0]["stage"] == "transcribe"
    assert failed[0]["error"] is failure  # The logger extracts safe traceback fields.
    assert worker_events[-1]["event"] == "job_failed"
    assert not any(event.get("stage") == "analyze" for event in worker_events)
    assert worker.job_context.get() is None
    assert str(failure) not in json.dumps([{key: value for key, value in event.items() if key != "error"} for event in worker_events])


def test_pipeline_trace_records_diarization_fallback_then_success(pipeline, monkeypatch, worker_events):
    failure = RuntimeError("private HF response")

    def diarize(*args, **kwargs):
        raise failure

    monkeypatch.setattr(clipping.diarization, "run_diarization", diarize)
    job = pipeline.run(use_camera_switch=True, diarization_speakers=2)
    assert job["status"] == "completed"
    fallback = [event for event in worker_events if event["event"] == "stage_fallback"]
    assert len(fallback) == 1
    assert fallback[0]["stage"] == "diarization"
    assert fallback[0]["error"] is failure
    assert worker_events[-1]["event"] == "job_completed"


def test_pipeline_trace_records_cancellation_boundary(pipeline, worker_events):
    job_id = store.create_job()
    pipeline.state.on_download = lambda: store.request_cancel(job_id)
    worker._run_pipeline_sync(job_id, pipeline.payload)
    assert store.get_job(job_id)["status"] == "cancelled"
    assert worker_events[-2]["event"] == "stage_cancelled"
    assert worker_events[-2]["stage"] == "download"
    assert worker_events[-1]["event"] == "job_cancelled"
    assert worker.job_context.get() is None


def test_failed_manifest_is_logged_without_copying_renderer_message(pipeline, worker_events):
    pipeline.state.results = [{"rank": 1, "status": "failed", "error": "private FFmpeg command", "video_exists": False}]
    assert pipeline.run()["status"] == "failed"
    assert any(event["event"] == "clip_failed" and event["clip_index"] == 1 for event in worker_events)
    assert worker_events[-1]["event"] == "job_failed"
    assert "private FFmpeg command" not in json.dumps(worker_events)


def test_renderer_trace_hook_is_inactive_in_cli_and_retains_api_exception(monkeypatch):
    # Load this dependency-free hook alone; no native renderer imports/models.
    source = (Path(__file__).resolve().parents[1] / "clipping/studio/core.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    hook = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_trace_render_failure")
    namespace = {"sys": sys}
    exec(compile(ast.Module(body=[hook], type_ignores=[]), "renderer_trace_hook", "exec"), namespace)
    events = []
    job = SimpleNamespace(get=lambda: None)
    logger = SimpleNamespace(job_context=job, log_event=lambda event, **fields: events.append({"event": event, **fields}))
    monkeypatch.setitem(sys.modules, "scripts.studio_logging", logger)
    failure = RuntimeError("private renderer details")
    namespace["_trace_render_failure"](failure, "concat", 2)
    assert events == []
    job.get = lambda: "job-123"
    namespace["_trace_render_failure"](failure, "concat", 2)
    namespace["_trace_render_failure"](failure, "edge_glow", 2, fallback=True)
    assert [event["event"] for event in events] == ["renderer_failed", "renderer_fallback"]
    assert events[0]["error"] is failure
    assert events[1]["level"] == "WARNING"
    assert events[0]["stage"] == "concat"


def test_failed_worker_writes_traceable_jsonl_without_private_values(pipeline, monkeypatch, tmp_path):
    from scripts import studio_logging

    monkeypatch.setenv("OSC_LOG_DIR", str(tmp_path / "diagnostics"))
    monkeypatch.setenv("OSC_LOG_RUN_ID", "a" * 32)
    path = studio_logging.configure_logging("api")
    private = "sensitive-provider-token-and-response"

    def provider_failure(*args, **kwargs):
        raise RuntimeError(private)

    monkeypatch.setattr(clipping.engine, "analyze_with_ai", provider_failure)
    try:
        job = pipeline.run()
        text = path.read_text(encoding="utf-8")
    finally:
        studio_logging.close_logging()
    assert job["status"] == "failed"
    events = [json.loads(line) for line in text.splitlines()]
    failed = next(event for event in events if event["event"] == "stage_failed")
    assert failed["stage"] == "analyze"
    assert failed["job_id"] == job["id"]
    assert failed["error"]["type"] == "RuntimeError"
    assert any(frame["function"] == "provider_failure" and frame["line"] > 0 for frame in failed["error"]["traceback"])
    assert all(event["run_id"] == "a" * 32 for event in events)
    assert private not in text and "test-key" not in text and "https://example.test/video" not in text
