"""The shared CLI/API audio extractor must never unlink its source."""

import pytest

from clipping import diarization


def test_same_path_is_rejected_before_touching_video(tmp_path, monkeypatch):
    source = tmp_path / "podcast.webm"
    source.write_bytes(b"original video")
    calls = []
    monkeypatch.setattr(diarization.subprocess, "run", lambda *a, **k: calls.append(a))

    with pytest.raises(ValueError, match="different"):
        diarization.extract_audio(str(source), str(source))

    assert source.read_bytes() == b"original video"
    assert calls == []


def test_cli_signature_and_ffmpeg_arguments_remain_compatible(tmp_path, monkeypatch):
    source = tmp_path / "podcast.mp4"
    output = tmp_path / "podcast_audio.wav"
    source.write_bytes(b"original video")
    calls = []
    monkeypatch.setattr(diarization.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    assert diarization.extract_audio(str(source), str(output)) == str(output)
    args, options = calls[0]
    assert args[args.index("-i") + 1] == str(source)
    assert args[-1] == str(output)
    assert options["check"] is True
    assert source.read_bytes() == b"original video"
