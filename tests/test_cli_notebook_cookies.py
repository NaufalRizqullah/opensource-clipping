"""Execute only cookie setup cells with synthetic files; never run the pipeline."""

import json
import os
from pathlib import Path
import shlex
import stat
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
COOKIE = b"# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tsynthetic-value\n"
LEGACY_KAGGLE = "/kaggle/input/datasets/muhammadnaufal/tiktok-secret-cookies/youtube_cookies.txt"
COLAB = "/content/youtube_cookies.txt"


def code_cell(notebook, marker):
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    matches = ["".join(cell["source"]) for cell in cells
               if cell["cell_type"] == "code" and marker in "".join(cell["source"])]
    assert len(matches) == 1
    return matches[0]


@pytest.fixture(params=["Quick_Start.ipynb", "Lib_OpenSource_Clipping.ipynb"])
def notebook(request, monkeypatch, tmp_path):
    working = tmp_path / "working directory"
    working.mkdir()
    monkeypatch.chdir(working)
    monkeypatch.delenv("OSC_YOUTUBE_COOKIES_FILE", raising=False)
    # Do not read real Colab/Kaggle credentials if these tests run in a notebook.
    isfile = os.path.isfile
    monkeypatch.setattr(os.path, "isfile", lambda path: False if path in (LEGACY_KAGGLE, COLAB) else isfile(path))
    return ROOT / "notebooks" / request.param


@pytest.mark.parametrize("configuration", ["environment", "manual_relative"])
def test_explicit_file_takes_priority_and_copy_is_writable(notebook, monkeypatch, tmp_path, capsys, configuration):
    attached = tmp_path / "private input" / "cookies.txt"
    attached.parent.mkdir()
    attached.write_bytes(COOKIE)
    attached.chmod(stat.S_IREAD)
    fallback = Path("youtube_cookies.txt")
    fallback.write_bytes(b"old fallback")
    source = code_cell(notebook, "cookie_candidates = [")
    scope = {}
    if configuration == "environment":
        monkeypatch.setenv("OSC_YOUTUBE_COOKIES_FILE", str(attached))
    else:
        monkeypatch.setenv("OSC_YOUTUBE_COOKIES_FILE", "unused-environment-path.txt")
        relative = os.path.relpath(attached, Path.cwd())
        source = source.replace(
            'YOUTUBE_COOKIES_FILE = os.environ.get("OSC_YOUTUBE_COOKIES_FILE", "")',
            f"YOUTUBE_COOKIES_FILE = {relative!r}",
        )
    try:
        exec(compile(source, str(notebook), "exec"), scope)
        args = shlex.split(scope["YT_COOKIES_ARG"])
        assert args == ["--yt-cookies", str(fallback.resolve())]
        jar = Path(args[1])
        assert jar.read_bytes() == COOKIE
        # yt-dlp can update the copy without writing to a read-only Input.
        jar.write_bytes(b"synthetic downloader update")
        assert attached.read_bytes() == COOKIE
        assert "synthetic-value" not in capsys.readouterr().out
    finally:
        attached.chmod(stat.S_IREAD | stat.S_IWRITE)


@pytest.mark.parametrize("location", ["working", "colab", "kaggle"])
def test_empty_setting_preserves_auto_detection(notebook, monkeypatch, tmp_path, location):
    source = Path("youtube_cookies.txt") if location == "working" else tmp_path / "attached.txt"
    source.write_bytes(COOKIE)
    if location != "working":
        # Map just the historical candidate to a synthetic file.
        candidate = COLAB if location == "colab" else LEGACY_KAGGLE
        isfile = os.path.isfile
        monkeypatch.setattr(os.path, "isfile", lambda path: path == candidate or isfile(path))
        import shutil
        copyfile = shutil.copyfile
        monkeypatch.setattr(shutil, "copyfile", lambda src, dst: copyfile(source if src == candidate else src, dst))
    scope = {}
    exec(compile(code_cell(notebook, "cookie_candidates = ["), str(notebook), "exec"), scope)
    args = shlex.split(scope["YT_COOKIES_ARG"])
    assert args == ["--yt-cookies", str(Path("youtube_cookies.txt").resolve())]
    assert Path(args[1]).read_bytes() == COOKIE


def test_same_file_is_not_copied_onto_itself(notebook, monkeypatch):
    source = Path("youtube_cookies.txt")
    source.write_bytes(COOKIE)
    monkeypatch.setenv("OSC_YOUTUBE_COOKIES_FILE", str(source.resolve()))
    scope = {}
    exec(compile(code_cell(notebook, "cookie_candidates = ["), str(notebook), "exec"), scope)
    assert shlex.split(scope["YT_COOKIES_ARG"])[1] == str(source.resolve())
    assert source.read_bytes() == COOKIE


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_invalid_explicit_path_does_not_use_fallback(notebook, monkeypatch, tmp_path, kind):
    candidate = tmp_path / "missing.txt"
    if kind == "directory":
        candidate.mkdir()
    Path("youtube_cookies.txt").write_bytes(COOKIE)
    monkeypatch.setenv("OSC_YOUTUBE_COOKIES_FILE", str(candidate))
    scope = {"YT_COOKIES_ARG": "--yt-cookies previous-run.txt"}
    with pytest.raises(FileNotFoundError, match="YOUTUBE_COOKIES_FILE"):
        exec(compile(code_cell(notebook, "cookie_candidates = ["), str(notebook), "exec"), scope)
    assert scope["YT_COOKIES_ARG"] == ""
    assert Path("youtube_cookies.txt").read_bytes() == COOKIE


@pytest.mark.parametrize("configured", [False, True])
def test_pipeline_command_receives_cookie_argument(notebook, configured, monkeypatch):
    if configured:
        Path("youtube_cookies.txt").write_bytes(COOKIE)
    scope = {
        "YT_COOKIES_ARG": "--yt-cookies previous-run.txt",
        "YOUTUBE_URL": "https://www.youtube.com/watch?v=test",
        "TOTAL_CLIPS": 3, "ASPECT_RATIO": "9:16", "FONT_STYLE": "DEFAULT",
        "HOOK_DURATION": 3, "WORDS_PER_SUBTITLE": 5, "WHISPER_MODEL": "small",
        "WHISPER_DEVICE": "cpu", "WHISPER_COMPUTE_TYPE": "int8",
        "AI_PROVIDER": "gemini", "START_LOCAL_VLLM": False,
        "APP_PYTHON": "python", "APP_ENV": None, "PROJECT_DIR": Path.cwd(),
        "os": os, "subprocess": subprocess,
    }
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    exec(compile(code_cell(notebook, "cookie_candidates = ["), str(notebook), "exec"), scope)
    source = code_cell(notebook, "# Run the clipping pipeline")
    exec(compile(source, str(notebook), "exec"), scope)
    assert len(calls) == 1
    argv = calls[0]
    assert argv[:2] == ["python", "main.py"]
    assert argv[argv.index("--words-per-sub") + 1] == "5"
    if configured:
        assert argv.count("--yt-cookies") == 1
        assert argv[argv.index("--yt-cookies") + 1] == str(Path("youtube_cookies.txt").resolve())
    else:
        assert "--yt-cookies" not in argv
        assert scope["YT_COOKIES_ARG"] == ""
