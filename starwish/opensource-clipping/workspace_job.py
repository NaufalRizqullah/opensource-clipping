"""Entry point installed next to render.py on the Starwish workspace VM."""

import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from clip_ranges import normalize_ranges
from render import RenderError, finish_render, render_interval


def download_section(video_id, source_path, start, end):
    # Decode from the exact padded boundary; stream-copy can start at an earlier keyframe.
    section_start = max(0, math.floor(start) - 5)
    section_end = math.ceil(end) + 5
    args = [
        sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist", "--no-cache-dir", "--no-progress",
        "--js-runtimes", "node", "--match-filters", "!is_live", "--download-sections",
        f"*{section_start}-{section_end}",
    ]
    if section_start:
        args.append("--force-keyframes-at-cuts")
    args.extend([
        "-f", "bv[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/bv[height<=720][ext=mp4]+ba/bv[height<=720][ext=mp4]",
        "-S", "vcodec:h264,acodec:aac",
        "--merge-output-format", "mp4", "-o", str(source_path.with_suffix(".%(ext)s")),
        "--", "https://www.youtube.com/watch?v=" + video_id,
    ])
    result = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode or not source_path.is_file():
        raise RenderError("Public YouTube section unavailable. Submit an authorized MP4 attachment instead.")
    return section_start


def process(input_dir, output_dir, video_id, ranges=None):
    intervals = normalize_ranges([{"start": 0.2, "end": 59.6}] if ranges is None else ranges)
    files = [p for p in input_dir.rglob("*") if p.is_file() or p.is_symlink()]
    if video_id is None:
        if (len(files) != 1 or files[0].is_symlink() or files[0].suffix.lower() != ".mp4"
                or not files[0].resolve().is_relative_to(input_dir.resolve())):
            raise RenderError("Attach one MP4 or provide the youtube_url parameter.")
    elif files or not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise RenderError("Choose exactly one source: an MP4 or a supported public YouTube video.")
    with tempfile.TemporaryDirectory(prefix="cpu-video-job-") as temp:
        stage = Path(temp)
        source = files[0] if files else stage / "source.mp4"
        records = []
        for index, interval in enumerate(intervals, start=1):
            start, end = interval["start"], interval["end"]
            source_offset = 0
            if video_id is not None:
                source_offset = download_section(video_id, source, start, end)
            records.append(render_interval(source, stage, start, end, source_offset=source_offset, index=index))
            if video_id is not None:
                source.unlink()
        finish_render(stage, output_dir, records)


if __name__ == "__main__":
    try:
        job = json.load(sys.stdin)
        process(Path("/job/input"), Path("/job/output"), job["video_id"], job.get("ranges"))
    except (RenderError, ValueError, OSError) as exc:
        sys.exit(f"Render failed: {exc}")
    print("STARWISH_CPU_CLIP_COMPLETE")
