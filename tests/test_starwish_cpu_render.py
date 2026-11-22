"""Run in a Linux CPU environment with FFmpeg after the package review."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


AGENT = Path(__file__).resolve().parents[1] / "starwish" / "cpu-video-clip"
FIXTURE = AGENT / "fixtures" / "color-bars.mp4"
spec = importlib.util.spec_from_file_location("cpu_clip_harness", AGENT / "harness.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def test_only_supported_video_sources_are_accepted():
    assert harness.requested_video({}) is None
    assert harness.requested_video({"youtube_url": "https://youtu.be/kR8bhkVAgBY"}) == "kR8bhkVAgBY"
    for url in ("file:///tmp/video.mp4", "https://example.org/a", "https://youtube.com/watch?v=INVALID"):
        with pytest.raises(ValueError):
            harness.requested_video({"youtube_url": url})


def test_job_with_nested_attachment_calls_renderer(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(AGENT))
    import workspace_job

    source = tmp_path / "input" / "fixtures" / "video.mp4"
    source.parent.mkdir(parents=True)
    source.touch()
    output = tmp_path / "output"
    calls = []
    monkeypatch.setattr(workspace_job, "render", lambda *args: calls.append(args))
    workspace_job.process(tmp_path / "input", output, None)
    assert calls == [(source, output, 0.2, 59.6)]
    with pytest.raises(workspace_job.RenderError, match="exactly one source"):
        workspace_job.process(tmp_path / "input", output, "kR8bhkVAgBY")


def test_harness_calls_workspace_and_propagates_failed_render(monkeypatch):
    monkeypatch.setenv("STARWISH_MCP_URL", "https://example.invalid")
    monkeypatch.setenv("STARWISH_MCP_TOKEN", "test")
    workspace = harness.Workspace()
    calls = []
    result = {"content": [{"type": "text", "text": "STARWISH_CPU_CLIP_COMPLETE"}]}
    responses = {"initialize": {"protocolVersion": "2025-06-18"}, "notifications/initialized": None,
                 "tools/list": {"tools": [{"name": "run_command"}]}, "tools/call": result}

    def fake_rpc(method, params, **kwargs):
        calls.append((method, params))
        return responses[method]

    monkeypatch.setattr(workspace, "rpc", fake_rpc)
    workspace.render(None)
    assert [method for method, _ in calls] == list(responses)
    call = calls[-1][1]
    assert call["name"] == "run_command"
    assert call["arguments"]["command"] == harness.WORKSPACE_COMMAND
    assert json.loads(call["arguments"]["stdin"]) == {"video_id": None}
    result["isError"] = True
    with pytest.raises(RuntimeError, match="render did not finish"):
        workspace.render(None)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg is required")
@pytest.mark.parametrize("with_audio", [False, True])
def test_render_outputs_and_preserves_audio(tmp_path, with_audio):
    source = FIXTURE
    if with_audio:
        source = tmp_path / "audio.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(FIXTURE),
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=60",
                        "-c:v", "copy", "-c:a", "aac", "-shortest", str(source)], check=True)
    output = tmp_path / "output"
    cmd = [sys.executable, str(AGENT / "render.py"), "--input", str(source),
           "--output-dir", str(output), "--start", "0.2", "--end", "1.6"]
    subprocess.run(cmd, check=True)
    assert {p.name for p in output.iterdir()} == {"clip.mp4", "metadata.json", "thumbnail.jpg"}
    record = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert (record["width"], record["height"], record["has_audio"]) == (720, 1280, with_audio)
    assert record["duration_seconds"] == pytest.approx(1.4, abs=0.1)
    assert (output / "thumbnail.jpg").read_bytes().startswith(b"\xff\xd8")
    assert subprocess.run(cmd).returncode != 0


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffprobe is required")
def test_refuses_missing_footage(tmp_path):
    output = tmp_path / "output"
    result = subprocess.run([sys.executable, str(AGENT / "render.py"), "--input", str(FIXTURE),
                             "--output-dir", str(output), "--end", "61"])
    assert result.returncode != 0
    assert not output.exists() or not any(output.iterdir())
