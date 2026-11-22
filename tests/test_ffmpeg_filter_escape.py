import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_helpers():
    spec = importlib.util.spec_from_file_location(
        "_test_studio_helpers", ROOT / "clipping" / "studio" / "helpers.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _load_helpers()


def test_posix_escaping_unchanged(monkeypatch):
    monkeypatch.setattr(helpers.os, "name", "posix")
    value = "/content/out/a:b's.ass"
    assert helpers.escape_ffmpeg_filter_value(value) == r"/content/out/a\:b\'s.ass"


def test_windows_drive_path_uses_forward_slashes_and_double_escaped_colon(monkeypatch):
    monkeypatch.setattr(helpers.os, "name", "nt")
    value = r"E:\short-clipper\custom_fonts"
    assert helpers.escape_ffmpeg_filter_value(value) == r"E\\:/short-clipper/custom_fonts"


@pytest.mark.skipif(os.name != "nt", reason="Windows drive-letter paths only")
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_windows_subtitles_filter_parses_with_drive_letter_paths(tmp_path):
    ass = tmp_path / "t.ass"
    ass.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 320\nPlayResY: 240\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,"
        "100,100,0,0,1,1,0,2,10,10,10,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,Hello\n",
        encoding="utf-8",
    )
    esc = helpers.escape_ffmpeg_filter_value
    vf = f"subtitles={esc(str(ass))}:fontsdir={esc(str(tmp_path))}"
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1",
            "-vf", vf, "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
