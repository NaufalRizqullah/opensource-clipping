"""Kaggle retry contracts using synthetic source files and no model or downloader."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import clipping
from clipping.config import build_config


@pytest.fixture
def source(monkeypatch, tmp_path):
    previous = tmp_path / "previous [run]"
    previous.mkdir()
    video = previous / "video_asli.MP4"
    video.write_bytes(b"synthetic MP4")
    current = tmp_path / "new-run"
    current.mkdir()
    monkeypatch.chdir(current)
    return video


@pytest.mark.parametrize("with_subtitles", [True, False])
def test_resume_uses_supplied_files_and_skips_download(source, with_subtitles, monkeypatch):
    url = "https://example.test/source"
    args = ["--url", url, "--reuse-source-video", str(source), "--use-dlp-subs", "--vllm-chunk-transcript"]
    cfg = build_config(args)
    assert cfg.file_video_asli == str(source) and cfg.url_youtube == url
    assert Path(cfg.outputs_dir) == Path.cwd() / "outputs"
    assert cfg.vllm_chunk_transcript

    subtitle = source.with_suffix(".en.json3")
    if with_subtitles:
        subtitle.write_bytes(b"synthetic subtitles")
    before = {p.name: p.read_bytes() for p in source.parent.iterdir()}
    cfg.api_key_gemini = "test"
    engine = Mock()
    engine.download_video.side_effect = AssertionError("Unexpected download")
    engine.parse_youtube_json3_subs.return_value = ("caption", ["segment"])
    engine.transcribe_video.return_value = ("speech", ["segment"])
    engine.analyze_with_ai.side_effect = RuntimeError("stop at inference")
    for name in ("diarization", "metadata", "studio", "hook_manager", "voiceover"):
        monkeypatch.setattr(clipping, name, SimpleNamespace(), raising=False)
    monkeypatch.setattr(clipping, "engine", engine, raising=False)
    path = Path(__file__).resolve().parents[1] / "clipping" / "runner.py"
    spec = importlib.util.spec_from_file_location("clipping._test_reuse_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    with pytest.raises(RuntimeError, match="stop at inference"):
        runner.run_pipeline(cfg)
    if with_subtitles:
        engine.parse_youtube_json3_subs.assert_called_once_with(str(subtitle), max_words_per_subtitle=cfg.max_kata_per_subtitle)
        engine.transcribe_video.assert_not_called()
    else:
        assert engine.transcribe_video.call_args.args == (str(source),)
    engine.analyze_with_ai.assert_called_once_with("caption" if with_subtitles else "speech", cfg)
    assert {p.name: p.read_bytes() for p in source.parent.iterdir()} == before


def test_reuse_rejects_invalid_source_or_unsafe_options(source):
    url = "https://example.test/source"
    for extra in (["--cleanup-source"], ["--url", url, url]):
        with pytest.raises(SystemExit):
            build_config(["--url", url, "--reuse-source-video", str(source), *extra])
    source.write_bytes(b"")
    for invalid in (source, source.parent, source.with_suffix(".missing")):
        with pytest.raises(SystemExit):
            build_config(["--url", url, "--reuse-source-video", str(invalid)])
    default = build_config(["--url", url])
    assert default.reuse_source_video is None and not default.vllm_chunk_transcript
