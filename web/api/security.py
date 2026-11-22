"""
web.api.security — Access control and hardening for the Studio API.

The Studio is commonly exposed to the internet through a tunnel (ngrok) from a
Kaggle/Colab notebook. Without protection, anyone holding the public URL could
spend the operator's AI credits, read rendered clips, or stop the server.

Behaviour
---------
* ``OSC_API_TOKEN`` set  → every ``/api/*`` request must present it via
  ``Authorization: Bearer <token>`` (or ``X-OSC-Token``). Liveness
  (``/api/health``) and CORS preflights stay public.
* ``OSC_API_TOKEN`` unset → open access (local development, backwards
  compatible with the GitHub Pages Studio); a warning is printed at startup.
* Media files cannot send headers from ``<video src>``, so output URLs carry a
  short-lived HMAC signature (``?exp=…&sig=…``) bound to the job and filename.
* Security headers are attached to every HTTP response.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from urllib.parse import parse_qs, quote

TOKEN_ENV = "OSC_API_TOKEN"
MEDIA_LINK_TTL = 12 * 60 * 60  # seconds
# Liveness, readiness (already sanitized) and API info are safe to expose.
_PUBLIC_PATHS = {"/api", "/api/health", "/api/ready"}
_MEDIA_PREFIX = "/api/outputs/"


def configured_token() -> str:
    return os.environ.get(TOKEN_ENV, "").strip()


def auth_enabled() -> bool:
    return bool(configured_token())


def _sign(token: str, job_id: str, filename: str, exp: int) -> str:
    message = f"{job_id}/{filename}:{exp}".encode("utf-8")
    return hmac.new(token.encode("utf-8"), message, hashlib.sha256).hexdigest()


def media_url(job_id: str, filename: str) -> str:
    """Public URL for an output file; signed when access control is enabled."""
    base = f"/api/outputs/{quote(job_id, safe='')}/{quote(filename, safe='')}"
    token = configured_token()
    if not token:
        return base
    exp = int(time.time()) + MEDIA_LINK_TTL
    return f"{base}?exp={exp}&sig={_sign(token, job_id, filename, exp)}"


def _valid_media_signature(token: str, path: str, query: bytes) -> bool:
    from urllib.parse import unquote

    parts = path[len(_MEDIA_PREFIX):].split("/")
    if len(parts) != 2 or not all(parts):
        return False
    job_id, filename = unquote(parts[0]), unquote(parts[1])
    params = parse_qs(query.decode("latin-1"))
    try:
        exp = int(params.get("exp", [""])[0])
        sig = params.get("sig", [""])[0]
    except ValueError:
        return False
    if exp < time.time() or exp > time.time() + MEDIA_LINK_TTL + 60:
        return False
    return hmac.compare_digest(sig, _sign(token, job_id, filename, exp))


def _presented_token(headers: dict[bytes, bytes]) -> str:
    auth = headers.get(b"authorization", b"").decode("latin-1")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return headers.get(b"x-osc-token", b"").decode("latin-1").strip()


_SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
    (b"x-frame-options", b"DENY"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=(), interest-cohort=()"),
    (b"cross-origin-opener-policy", b"same-origin"),
]
# Applied to the bundled dashboard pages only; the API returns JSON/media.
SPA_CSP = (
    "default-src 'self'; img-src 'self' data: blob: https://i.ytimg.com https://img.youtube.com; "
    "media-src 'self' blob: https:; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com data:; script-src 'self'; "
    "connect-src 'self' https: http://localhost:* http://127.0.0.1:*; "
    "frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'"
)


class SecurityMiddleware:
    """Pure ASGI middleware so streaming responses (SSE) are untouched."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        method: str = scope.get("method", "GET")
        token = configured_token()

        if token and path.startswith("/api") and method != "OPTIONS" and path not in _PUBLIC_PATHS:
            headers = dict(scope.get("headers") or [])
            presented = _presented_token(headers)
            allowed = bool(presented) and hmac.compare_digest(presented, token)
            if not allowed and method in {"GET", "HEAD"} and path.startswith(_MEDIA_PREFIX):
                allowed = _valid_media_signature(token, path, scope.get("query_string", b""))
            if not allowed:
                await self._reject(send)
                return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                existing = {k.lower() for k, _ in message.get("headers", [])}
                extra = [(k, v) for k, v in _SECURITY_HEADERS if k not in existing]
                if not path.startswith("/api") and not path.startswith("/docs") and path != "/openapi.json":
                    extra.append((b"content-security-policy", SPA_CSP.encode()))
                message["headers"] = list(message.get("headers", [])) + extra
            await send(message)

        await self.app(scope, receive, send_with_headers)

    @staticmethod
    async def _reject(send):
        body = b'{"detail":"Unauthorized: a valid Studio access token is required."}'
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b'Bearer realm="osc-studio"'),
                *_SECURITY_HEADERS,
            ],
        })
        await send({"type": "http.response.body", "body": body})
