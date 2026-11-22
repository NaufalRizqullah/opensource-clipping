"""Access-token, signed media links, URL validation and security headers."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from web.api import security
from web.api.app import app
from web.api.models import JobCreateRequest

TOKEN = "test-token-0123456789"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv(security.TOKEN_ENV, TOKEN)
    with TestClient(app) as c:
        yield c


def test_protected_routes_reject_missing_or_wrong_token(client):
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/api/jobs", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.post("/api/shutdown").status_code == 401
    assert client.put("/api/settings", json={"google_api_key": "x"}).status_code == 401


def test_valid_token_is_accepted_via_either_header(client):
    assert client.get("/api/jobs", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert client.get("/api/auth/check", headers={"X-OSC-Token": TOKEN}).json()["authenticated"] is True


def test_public_endpoints_and_preflight_stay_open(client):
    assert client.get("/api/health").status_code == 200
    assert client.get("/api").json()["auth_required"] is True
    preflight = client.options(
        "/api/jobs",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET",
                 "Access-Control-Request-Headers": "authorization"},
    )
    assert preflight.status_code == 200


def test_unauthorized_response_is_readable_cross_origin(client):
    response = client.get("/api/jobs", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_signed_media_url_round_trip_and_tamper_rejection(client, tmp_path, monkeypatch):
    from web.api.routes import files

    monkeypatch.setattr(files, "OUTPUTS_DIR", str(tmp_path))
    (tmp_path / "job1").mkdir()
    (tmp_path / "job1" / "clip.mp4").write_bytes(b"\x00" * 16)

    url = security.media_url("job1", "clip.mp4")
    assert "sig=" in url and "exp=" in url
    assert client.get(url).status_code == 200
    assert client.get(url.replace("job1/clip.mp4", "job1/other.mp4")).status_code == 401
    assert client.get(url[:-1] + ("0" if url[-1] != "0" else "1")).status_code == 401

    expired = int(time.time()) - 5
    sig = security._sign(TOKEN, "job1", "clip.mp4", expired)
    assert client.get(f"/api/outputs/job1/clip.mp4?exp={expired}&sig={sig}").status_code == 401


def test_security_headers_present(client):
    response = client.get("/api/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_open_mode_when_token_unset(monkeypatch):
    monkeypatch.delenv(security.TOKEN_ENV, raising=False)
    with TestClient(app) as c:
        assert c.get("/api/jobs").status_code == 200
    assert security.media_url("a", "b c.mp4") == "/api/outputs/a/b%20c.mp4"


@pytest.mark.parametrize("bad", ["file:///etc/passwd", "C:\\video.mp4", "ytsearch:cats", "https://", "javascript:alert(1)"])
def test_job_url_must_be_remote_http(bad):
    with pytest.raises(ValueError):
        JobCreateRequest(url=bad)


def test_job_url_accepts_https():
    assert JobCreateRequest(url=" https://youtu.be/abc ").url == "https://youtu.be/abc"
