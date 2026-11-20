"""A new download must never reuse the previous run's source video or subtitles."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

URL = "https://www.youtube.com/watch?v=new"


@pytest.fixture
def engine_and_calls(monkeypatch, tmp_path):
    destination = tmp_path / "video_asli.mp4"
    calls = []

    class FakeYDL:
        """Mimics yt-dlp: skips the media download when the target already exists."""

        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, *args, **kwargs):
            return {}

        def download(self, urls):
            if self.opts.get("skip_download"):
                calls.append("subs")
                destination.with_suffix(".en.json3").write_text("new subs")
            elif destination.exists():
                calls.append("skipped: already downloaded")
            else:
                calls.append("video")
                destination.write_text("new video")

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYDL))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=object))
    source = Path(__file__).resolve().parents[1] / "clipping/engine.py"
    spec = importlib.util.spec_from_file_location("_stale_source_engine", source)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine, destination, calls


def test_previous_source_video_and_subtitles_are_replaced(engine_and_calls, tmp_path):
    engine, destination, calls = engine_and_calls
    destination.write_text("OLD video")
    (tmp_path / "video_asli.id.json3").write_text("OLD subs")
    (tmp_path / "video_asli.f616.mp4.part").write_text("OLD fragment")
    (tmp_path / "video_asli_audio.wav").write_text("OLD audio")
    keep = tmp_path / "highlight_rank_1_ready.mp4"
    keep.write_text("rendered clip")

    engine.download_video(URL, str(destination), use_dlp_subs=True)

    assert calls == ["subs", "video"]
    assert destination.read_text() == "new video"
    assert sorted(p.name for p in tmp_path.glob("video_asli*")) == [
        "video_asli.en.json3",
        "video_asli.mp4",
    ]
    assert keep.read_text() == "rendered clip"


def test_first_download_without_leftovers_is_unchanged(engine_and_calls, tmp_path):
    engine, destination, calls = engine_and_calls

    engine.download_video(URL, str(destination))

    assert calls == ["video"]
    assert destination.read_text() == "new video"
