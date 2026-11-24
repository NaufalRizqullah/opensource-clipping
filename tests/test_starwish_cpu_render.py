"""Run in a Linux CPU environment with FFmpeg after the package review."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


AGENT = Path(__file__).resolve().parents[1] / "starwish" / "opensource-clipping"
FIXTURE = AGENT / "fixtures" / "color-bars.mp4"


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.syspath_prepend(str(AGENT))
    spec = importlib.util.spec_from_file_location("cpu_clip_harness", AGENT / "harness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_supported_video_sources_are_accepted(harness):
    assert harness.requested_video({}) is None
    assert harness.requested_video({"youtube_url": "https://youtu.be/kR8bhkVAgBY"}) == "kR8bhkVAgBY"
    for url in ("file:///tmp/video.mp4", "https://example.org/a", "https://youtube.com/watch?v=INVALID"):
        with pytest.raises(ValueError):
            harness.requested_video({"youtube_url": url})


def test_normalizes_mixed_range_formats_in_request_order(monkeypatch):
    monkeypatch.syspath_prepend(str(AGENT))
    from clip_ranges import normalize_ranges

    assert normalize_ranges([
        {"start": "06:00", "end": "06:45"},
        {"start": 1.5, "end": 12},
        {"start": "01:02:03.5", "end": "01:02:04.25"},
    ]) == [{"start": 360, "end": 405}, {"start": 1.5, "end": 12}, {"start": 3723.5, "end": 3724.25}]


@pytest.mark.parametrize("ranges", [
    None, [], {"start": 0, "end": 1}, [{"start": 0, "end": 1}] * 4,
    [{"start": 1}], [{"start": 0, "end": 1, "extra": True}], [{"start": 1, "end": 1}],
    [{"start": 3, "end": 2}], [{"start": -1, "end": 1}], [{"start": False, "end": 1}],
    [{"start": 0, "end": float("inf")}], [{"start": float("nan"), "end": 2}],
    [{"start": "00:60", "end": 120}], [{"start": "01:60:00", "end": 7210}],
    [{"start": "not a time", "end": 1}], [{"start": 0, "end": 1}, None],
])
def test_rejects_invalid_ranges(monkeypatch, ranges):
    monkeypatch.syspath_prepend(str(AGENT))
    from clip_ranges import normalize_ranges

    with pytest.raises(ValueError):
        normalize_ranges(ranges)


def test_media_failure_reports_bounded_stderr_tail(monkeypatch):
    monkeypatch.syspath_prepend(str(AGENT))
    import render

    command = [sys.executable, "-c",
               "import sys; sys.stderr.write('early diagnostic' + 'x' * 8192 + "
               "'\\nInvalid data found when processing input\\n'); sys.exit(183)"]
    with pytest.raises(render.RenderError) as failure:
        render.run_media(command, "CPU clip encoding")
    message = str(failure.value)
    assert "CPU clip encoding" in message and "183" in message
    assert "Invalid data found when processing input" in message
    assert "early diagnostic" not in message
    assert len(message) < 2200


def test_rejects_attachment_and_youtube_together(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(AGENT))
    import workspace_job

    source = tmp_path / "input" / "fixtures" / "video.mp4"
    source.parent.mkdir(parents=True)
    source.touch()
    with pytest.raises(workspace_job.RenderError, match="exactly one source"):
        workspace_job.process(tmp_path / "input", tmp_path / "output", "kR8bhkVAgBY")


def test_omitting_ranges_preserves_first_clip(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(AGENT))
    import workspace_job

    source = tmp_path / "input" / "fixtures" / "source.mp4"
    source.parent.mkdir(parents=True)
    source.touch()
    calls = []

    def fake_render(local_source, stage, start, end, *, source_offset=0.0, index=1):
        calls.append((local_source, start, end, source_offset, index))
        return {}

    monkeypatch.setattr(workspace_job, "render_interval", fake_render)
    monkeypatch.setattr(workspace_job, "finish_render", lambda *args: None)
    workspace_job.process(tmp_path / "input", tmp_path / "output", None)
    assert calls == [(source, 0.2, 59.6, 0, 1)]


def test_harness_calls_workspace_and_propagates_failed_render(tmp_path, monkeypatch, harness):
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

    request = tmp_path / "request.json"
    request.write_text(json.dumps({"params": {"youtube_url": "https://youtu.be/kR8bhkVAgBY",
                                              "ranges": [{"start": "06:00", "end": "06:45"}]}}),
                       encoding="utf-8")
    monkeypatch.setenv("STARWISH_REQUEST_FILE", str(request))
    monkeypatch.setattr(workspace, "rpc", fake_rpc)
    monkeypatch.setattr(harness, "Workspace", lambda: workspace)
    harness.main()
    assert [method for method, _ in calls] == list(responses)
    assert calls[0][1]["clientInfo"]["name"] == "opensource-clipping"
    call = calls[-1][1]
    assert call["name"] == "run_command"
    assert call["arguments"]["command"] == harness.WORKSPACE_COMMAND
    assert json.loads(call["arguments"]["stdin"]) == {
        "video_id": "kR8bhkVAgBY", "ranges": [{"start": 360, "end": 405}]}
    result["isError"] = True
    with pytest.raises(RuntimeError, match="render did not finish"):
        workspace.render(None)


def test_render_seeks_in_section_using_absolute_timestamps(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(AGENT))
    import render

    source = tmp_path / "section.mp4"
    source.touch()
    monkeypatch.setattr(render, "probe", lambda path: {"streams": [
        {"index": 0, "codec_type": "video", "duration": "55"},
        {"index": 1, "codec_type": "audio", "duration": "55"},
    ]})
    commands = []

    def intercept(command, stage):
        commands.append(command)
        raise render.RenderError("Intercepted before encoding")

    monkeypatch.setattr(render, "run_media", intercept)
    with pytest.raises(render.RenderError, match="Intercepted before encoding"):
        render.render_interval(source, tmp_path, 360, 405, source_offset=355)
    command, = commands
    assert float(command[command.index("-ss") + 1]) == 5
    assert float(command[command.index("-t") + 1]) == 45


def test_youtube_job_fetches_requested_sections_and_passes_absolute_offsets(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(AGENT))
    import workspace_job

    downloads = []
    rendered = []

    def fake_download(command, **kwargs):
        downloads.append(command)
        template = command[command.index("-o") + 1]
        Path(template.replace("%(ext)s", "mp4")).touch()
        return subprocess.CompletedProcess(command, 0)

    def fake_render(source, stage, start, end, *, source_offset, index):
        assert source.is_file()
        rendered.append((start, end, source_offset, index))
        return {}

    monkeypatch.setattr(workspace_job.subprocess, "run", fake_download)
    monkeypatch.setattr(workspace_job, "render_interval", fake_render)
    monkeypatch.setattr(workspace_job, "finish_render", lambda *args: None)
    workspace_job.process(tmp_path / "input", tmp_path / "output", "kR8bhkVAgBY",
                          [{"start": 360, "end": 405}, {"start": 0, "end": 2}])
    assert rendered == [(360, 405, 355, 1), (0, 2, 0, 2)]
    assert [cmd[cmd.index("--download-sections") + 1] for cmd in downloads] == ["*355-410", "*0-7"]
    assert "--force-keyframes-at-cuts" in downloads[0]
    assert all(cmd[-1] == "https://www.youtube.com/watch?v=kR8bhkVAgBY" for cmd in downloads)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg is required")
def test_three_requested_clips_and_metadata_preserve_order(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(AGENT))
    import workspace_job

    input_dir = tmp_path / "input"
    nested = input_dir / "fixtures"
    nested.mkdir(parents=True)
    shutil.copyfile(FIXTURE, nested / "source.mp4")
    output = tmp_path / "output"
    ranges = [{"start": 57, "end": 58.4}, {"start": 0.2, "end": 1.2}, {"start": 30, "end": 31.6}]
    workspace_job.process(input_dir, output, None, ranges)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    records = metadata["clips"]
    assert len(records) == len(ranges)
    assert metadata["start_seconds"] == records[0]["start_seconds"]
    assert metadata["end_seconds"] == records[0]["end_seconds"]
    assert metadata["video"] == records[0]["video"]
    filenames = {"metadata.json"}
    for record, interval, suffix in zip(records, ranges, ("", "_2", "_3")):
        assert (record["start_seconds"], record["end_seconds"]) == (interval["start"], interval["end"])
        assert record["duration_seconds"] == pytest.approx(interval["end"] - interval["start"], abs=0.1)
        assert (record["video"], record["thumbnail"]) == (f"clip{suffix}.mp4", f"thumbnail{suffix}.jpg")
        filenames.update((record["video"], record["thumbnail"]))
        assert (output / record["video"]).stat().st_size > 0
        assert (output / record["thumbnail"]).read_bytes().startswith(b"\xff\xd8")
    assert {p.name for p in output.iterdir()} == filenames


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg is required")
def test_error_in_later_range_cannot_deliver_partial_results(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(AGENT))
    import workspace_job

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copyfile(FIXTURE, input_dir / "source.mp4")
    output = tmp_path / "output"
    ranges = [{"start": 0, "end": 1}, {"start": 59, "end": 62}]
    with pytest.raises(workspace_job.RenderError):
        workspace_job.process(input_dir, output, None, ranges)
    assert not output.exists() or not any(output.iterdir())


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
