"""Cookie upload and credential boundaries, using only synthetic cookies."""

import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
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


@pytest.mark.parametrize("configured", [False, True])
def test_kaggle_video_cell_cookie_file_on_download_failure(tmp_path, monkeypatch, configured):
    notebook = Path(__file__).resolve().parents[1] / "notebooks/kaggle-vllm-gemma4.ipynb"
    sources = ("".join(cell["source"]) for cell in json.loads(notebook.read_text("utf-8"))["cells"])
    cell = next(source for source in sources if "def uji_satu_video(" in source)
    workspace = tmp_path / "output"
    workspace.mkdir()
    attached = tmp_path / "attached.txt"
    attached.write_bytes(FILE + OTHER)
    monkeypatch.setenv("OSC_YOUTUBE_COOKIES_FILE", str(attached) if configured else "")
    monkeypatch.setattr(youtube_cookies, "_uploaded", None)
    copies = []

    def failed_download(argv, **kwargs):
        if configured:
            jar = Path(argv[argv.index("--yt-cookies") + 1])
            assert jar != attached and not jar.is_relative_to(workspace)
            assert jar.read_bytes() == FILE
            jar.write_bytes(b"synthetic downloader update")
            copies.append(jar)
        else:
            assert "--yt-cookies" not in argv
        raise RuntimeError("simulated CLI failure")

    scope = dict(
        os=os, Path=Path, workspace=workspace, PROJECT_DIR=tmp_path,
        STUDIO_PYTHON="python", STUDIO_ENV=tmp_path, cache_env={},
        check_storage=lambda *_: None, storage=None, MIN_OUTPUT_GIB=1,
        runtime_environment=lambda _: {}, subprocess=SimpleNamespace(run=failed_download),
    )
    exec(compile(cell, str(notebook), "exec"), scope)

    def kaggle_private_dir(*, prefix, dir):
        assert dir == "/tmp"
        return tempfile.TemporaryDirectory(prefix=prefix, dir=tmp_path)

    scope["tempfile"] = SimpleNamespace(mkdtemp=tempfile.mkdtemp, TemporaryDirectory=kaggle_private_dir)
    with pytest.raises(RuntimeError, match="simulated CLI failure"):
        scope["uji_satu_video"](YOUTUBE_URL)
    assert attached.read_bytes() == FILE + OTHER
    assert all(not path.exists() for path in copies)


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
