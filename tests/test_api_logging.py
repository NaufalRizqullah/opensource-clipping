"""Exercise real ASGI responses while keeping credentials out of file logs."""

import asyncio
import json
import re
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
import pytest

from scripts import studio_logging
from web.api import logging_middleware


@pytest.fixture
def diagnostic_log(monkeypatch, tmp_path):
    monkeypatch.setenv("OSC_LOG_DIR", str(tmp_path / "logs"))
    path = studio_logging.configure_logging(role="api")
    return path


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_requests_log_templates_and_ids_without_sensitive_input(diagnostic_log):
    app = FastAPI()
    app.add_middleware(logging_middleware.RequestLoggingMiddleware)
    observed = []

    @app.post("/items/{item_id}")
    async def endpoint(item_id: str, request: Request):
        await request.body()
        observed.append(studio_logging.current_request_id.get())
        studio_logging.log_event("job.queued", component="worker", job_id="test-job-123")
        return {"ok": True}

    @app.get("/denied/{item_id}")
    async def denied(item_id: str):
        raise HTTPException(status_code=403, detail="private-response-marker")

    with TestClient(app) as client:
        response = client.post(
            "/items/private-path-marker?token=private-query-marker",
            headers={"Authorization": "Bearer private-header-marker", "X-Request-ID": "private-client-id"},
            json={"api_key": "private-body-marker"},
        )
        denied_response = client.get("/denied/private-denied-path")
        missing_response = client.get("/private-unmatched-path")

    assert response.status_code == 200
    request_id = response.headers["X-Request-ID"]
    assert re.fullmatch(r"[0-9a-f]{32}", request_id)
    assert observed == [request_id]
    assert denied_response.status_code == 403
    assert missing_response.status_code == 404
    events = read_events(diagnostic_log)
    completed = [item for item in events if item["event"] == "request.completed"]
    assert [item["status_code"] for item in completed] == [200, 403, 404]
    assert [item["route"] for item in completed] == ["/items/{item_id}", "/denied/{item_id}", "unmatched"]
    assert all(item["elapsed_ms"] >= 0 for item in completed)
    queued = next(item for item in events if item["event"] == "job.queued")
    assert queued["request_id"] == request_id
    assert queued["job_id"] == "test-job-123"
    assert "private-" not in diagnostic_log.read_text(encoding="utf-8")
    assert studio_logging.current_request_id.get() is None


def test_unhandled_exception_is_logged_without_changing_500(diagnostic_log):
    app = FastAPI()
    app.add_middleware(logging_middleware.RequestLoggingMiddleware)

    @app.get("/broken")
    async def broken():
        private_local = "private-local-marker"
        raise ValueError("private-exception-marker " + private_local)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/broken")
    assert response.status_code == 500
    failed = next(item for item in read_events(diagnostic_log) if item["event"] == "request.failed")
    assert failed["route"] == "/broken"
    assert failed["status_code"] == 500
    assert "ValueError" in json.dumps(failed)
    assert "broken" in json.dumps(failed)
    assert "private-" not in diagnostic_log.read_text(encoding="utf-8")


def test_streaming_failure_keeps_chunks_and_original_exception(diagnostic_log):
    failure = RuntimeError("private-stream-marker")
    messages = []

    async def streaming_app(scope, receive, send):
        scope["route"] = SimpleNamespace(path="/stream/{job_id}")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"first chunk", "more_body": True})
        raise failure

    async def scenario():
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        with pytest.raises(RuntimeError) as caught:
            await logging_middleware.RequestLoggingMiddleware(streaming_app)(
                {"type": "http", "method": "GET", "path": "/stream/private-path-marker"}, receive, send,
            )
        assert caught.value is failure
        assert studio_logging.current_request_id.get() is None

    asyncio.run(scenario())
    assert messages[1] == {"type": "http.response.body", "body": b"first chunk", "more_body": True}
    events = read_events(diagnostic_log)
    assert [item["event"] for item in events] == ["request.started", "request.failed"]
    assert events[-1]["route"] == "/stream/{job_id}"
    assert events[-1]["status_code"] == 200  # Headers were already sent.
    assert "private-" not in diagnostic_log.read_text(encoding="utf-8")


def test_logging_failure_does_not_break_request(monkeypatch):
    def broken_logger(*args, **kwargs):
        raise OSError("simulated unavailable disk")

    monkeypatch.setattr(logging_middleware, "log_event", broken_logger)
    app = FastAPI()
    app.add_middleware(logging_middleware.RequestLoggingMiddleware)

    @app.get("/")
    async def endpoint():
        return {"ok": True}

    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_request_cancellation_restores_context_and_propagates(diagnostic_log):
    async def cancelled_app(scope, receive, send):
        raise asyncio.CancelledError()

    async def scenario():
        token = studio_logging.current_request_id.set("outer-request")
        try:
            with pytest.raises(asyncio.CancelledError):
                await logging_middleware.RequestLoggingMiddleware(cancelled_app)(
                    {"type": "http", "method": "GET"}, None, None,
                )
            assert studio_logging.current_request_id.get() == "outer-request"
        finally:
            studio_logging.current_request_id.reset(token)

    asyncio.run(scenario())
    events = read_events(diagnostic_log)
    assert [item["event"] for item in events] == ["request.started", "request.cancelled"]
    assert events[-1]["level"] == "WARNING"


def test_application_lifecycle_logs_start_and_shutdown(diagnostic_log, monkeypatch):
    from web.api.app import app
    from web.api import worker

    async def cancelled():
        pass

    monkeypatch.setattr(worker, "cancel_all_jobs", cancelled)
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
    events = [item["event"] for item in read_events(diagnostic_log)]
    assert "api.startup.started" in events
    assert "api.startup.completed" in events
    assert "api.shutdown.started" in events
    assert "api.shutdown.completed" in events


def test_shutdown_failure_preserves_exception_and_logs_location(diagnostic_log, monkeypatch):
    from web.api.app import app
    from web.api import worker

    failure = RuntimeError("private-shutdown-marker")

    async def cancel_failed():
        raise failure

    monkeypatch.setattr(worker, "cancel_all_jobs", cancel_failed)
    with pytest.raises(RuntimeError) as caught:
        with TestClient(app):
            pass
    assert caught.value is failure
    events = read_events(diagnostic_log)
    assert events[-1]["event"] == "api.shutdown.failed"
    assert "cancel_failed" in json.dumps(events[-1])
    assert "private-shutdown-marker" not in diagnostic_log.read_text(encoding="utf-8")
