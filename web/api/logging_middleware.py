"""Trace HTTP requests without reading bodies, credentials, or concrete URLs."""

from __future__ import annotations

import asyncio
import time
import uuid

from scripts.studio_logging import current_request_id, log_event


def _log(event: str, **fields) -> None:
    """A diagnostics failure must never change an HTTP response or exception."""
    try:
        log_event(event, component="api", **fields)
    except Exception:
        pass


class RequestLoggingMiddleware:
    """Pure ASGI middleware: preserve streaming and cancellation semantics."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex
        token = current_request_id.set(request_id)
        scope.setdefault("state", {})["request_id"] = request_id
        method = scope.get("method", "OTHER")
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD", "CONNECT", "TRACE"}:
            method = "OTHER"
        started = time.monotonic()
        status_code = 500
        completed = False
        _log("request.started", request_id=request_id, method=method)

        def route_template():
            # Routing resolves this object after the middleware starts. Never
            # fall back to scope['path'], query_string, headers or client data.
            return getattr(scope.get("route"), "path", None) or "unmatched"

        async def traced_send(message):
            nonlocal status_code, completed
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message = {
                    **message,
                    "headers": [
                        (key, value) for key, value in message.get("headers", [])
                        if key.lower() != b"x-request-id"
                    ] + [(b"x-request-id", request_id.encode("ascii"))],
                }
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                completed = True

        try:
            await self.app(scope, receive, traced_send)
        except BaseException as exc:
            cancelled = isinstance(exc, asyncio.CancelledError)
            _log(
                "request.cancelled" if cancelled else "request.failed",
                level="WARNING" if cancelled else "ERROR",
                request_id=request_id, method=method, route=route_template(),
                status_code=status_code, error=exc,
                elapsed_ms=round((time.monotonic() - started) * 1000, 3),
            )
            raise
        else:
            _log(
                "request.completed" if completed else "request.disconnected",
                level="ERROR" if status_code >= 500 else "WARNING" if status_code >= 400 or not completed else "INFO",
                request_id=request_id, method=method, route=route_template(),
                status_code=status_code,
                elapsed_ms=round((time.monotonic() - started) * 1000, 3),
            )
        finally:
            current_request_id.reset(token)
