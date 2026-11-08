"""CLI contracts that must remain independent of the Studio backend."""

import os
from pathlib import Path
import subprocess
import sys

from clipping import config


def test_cli_help_does_not_require_fastapi_or_gpu(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-S", str(root / "main.py"), "--help"],
        cwd=tmp_path, env={**os.environ, "PYTHON_DOTENV_DISABLED": "1", "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, encoding="utf-8", timeout=15,
    )
    assert result.returncode == 0, result.stderr
    for flag in ("--url", "--yt-cookies", "--load-gemini-json", "--ai-provider", "--story-mode"):
        assert flag in result.stdout
    assert not (tmp_path / "outputs").exists()


def test_cli_single_batch_and_cached_ai_flags_are_preserved(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("GOOGLE_API_KEY", "NVIDIA_API_KEY", "PEXELS_API_KEY", "HF_TOKEN"):
        monkeypatch.setenv(name, "")
    urls = ["https://example.test/first", "https://example.test/second"]
    single = config.build_config(["--url", urls[0], "--yt-cookies", "browser-cookies.txt", "--load-gemini-json"])
    assert single.url_youtube == urls[0]
    assert single.url_list == urls[:1]
    assert single.yt_cookies == "browser-cookies.txt"
    assert single.load_gemini_json is True
    batch = config.build_config(["--url", *urls, "--ai-provider", "nvidia", "--cleanup-source"])
    assert batch.url_youtube == urls
    assert batch.url_list == urls
    assert batch.ai_provider == "nvidia"
    assert batch.cleanup_source is True
    assert batch.file_video_asli == str(tmp_path / "video_asli.mp4")


def test_cli_story_mode_still_accepts_no_url(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("GOOGLE_API_KEY", "NVIDIA_API_KEY", "PEXELS_API_KEY", "HF_TOKEN"):
        monkeypatch.setenv(name, "")
    cfg = config.build_config(["--story-mode"])
    assert cfg.story_mode is True
    assert cfg.url_list == []
