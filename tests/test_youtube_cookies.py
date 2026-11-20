"""Cookie upload and credential boundaries, using only synthetic cookies."""

import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from web.api import config_adapter, youtube_cookies
from web.api.app import app
from web.api.routes import settings


ENDPOINT = "/api/settings/youtube-cookies"
HEADERS = {"Content-Type": "application/octet-stream"}
YOUTUBE_URL = "https://www.youtube.com/watch?v=test"
FILE = b"# Netscape HTTP Cookie File\n#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tsynthetic-value\n"
OTHER = b".example.net\tTRUE\t/\tFALSE\t0\tOTHER\tprivate-other-site\n"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config_adapter, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("OSC_YOUTUBE_COOKIES_FILE", raising=False)
    monkeypatch.setattr(settings, "_check_gpu", lambda: False)
    monkeypatch.setattr(youtube_cookies, "_uploaded", None)
    with TestClient(app) as client:
        yield client


def test_upload_replace_remove_and_fallback(client, tmp_path, monkeypatch):
    fallback = tmp_path / "attached.txt"
    fallback.write_bytes(FILE.replace(b"synthetic-value", b"operator-value"))
    monkeypatch.setenv("OSC_YOUTUBE_COOKIES_FILE", str(fallback))
    uploaded = client.put(ENDPOINT, content=b"\xef\xbb\xbf" + (FILE + OTHER).replace(b"\n", b"\r\n"), headers=HEADERS)
    assert uploaded.status_code == 200
    assert uploaded.json() == {"youtube_cookies_set": True, "youtube_cookies_uploaded": True}
    assert youtube_cookies.for_download("youtube", YOUTUBE_URL, str(fallback)) == FILE
    assert "synthetic-value" not in client.get("/api/settings").text
    for invalid, status in [
        (b"private-invalid-value", 422),
        (FILE.splitlines(keepends=True)[0] + OTHER, 422),
        (FILE.replace(b"\t0\t", b"\t1\t"), 422),
        (b"x" * (youtube_cookies.MAX_BYTES + 1), 413),
    ]:
        result = client.put(ENDPOINT, content=invalid, headers=HEADERS)
        assert result.status_code == status and "private-" not in result.text
        assert youtube_cookies.for_download("youtube", YOUTUBE_URL, None) == FILE
    assert client.delete(ENDPOINT).json() == {"youtube_cookies_set": True, "youtube_cookies_uploaded": False}
    assert b"operator-value" in youtube_cookies.for_download("youtube", YOUTUBE_URL, str(fallback))
    assert client.get(ENDPOINT).status_code == 405


@pytest.mark.parametrize("platform,url", [
    ("tiktok", YOUTUBE_URL),
    ("youtube", "https://youtube.com.example.net/watch?v=a"),
    ("youtube", "https://youtube.com@example.net"),
    ("youtube", "file://youtube.com/path"),
])
def test_only_youtube_downloads_receive_upload(client, platform, url):
    assert client.put(ENDPOINT, content=FILE, headers=HEADERS).is_success
    assert youtube_cookies.for_download(platform, url, None) is None


def test_export_loads_as_a_yt_dlp_session_cookie(client):
    from yt_dlp.cookies import YoutubeDLCookieJar

    assert client.put(ENDPOINT, content=FILE + OTHER, headers=HEADERS).is_success
    data = youtube_cookies.for_download("youtube", YOUTUBE_URL, None)
    jar = YoutubeDLCookieJar(io.StringIO(data.decode("utf-8")))
    jar.load()
    assert jar.get_cookie_header(YOUTUBE_URL) == "SID=synthetic-value"
    assert jar.get_cookie_header("https://example.net/") is None


def test_subtitle_and_media_downloads_get_independent_cookie_streams(monkeypatch, tmp_path):
    destination = tmp_path / "source.mp4"
    received = []

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def extract_info(self, *args, **kwargs):
            return {}
        def download(self, urls):
            jar = self.opts["cookiefile"]
            received.append(jar.read())
            jar.write("updated by fake yt-dlp")
            if self.opts.get("skip_download"):
                destination.with_suffix(".en.json3").touch()
            else:
                destination.touch()

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYDL))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=object))
    source = Path(__file__).resolve().parents[1] / "clipping/engine.py"
    spec = importlib.util.spec_from_file_location("_cookie_download_engine", source)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.download_video(YOUTUBE_URL, str(destination), use_dlp_subs=True, yt_cookies=FILE)
    assert received == [FILE.decode(), FILE.decode()]
    assert not list(tmp_path.glob("*.txt"))
