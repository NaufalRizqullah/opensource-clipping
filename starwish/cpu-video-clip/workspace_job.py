"""Entry point installed next to render.py on the Starwish workspace VM."""

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from render import RenderError, render


def process(input_dir, output_dir, video_id):
    files = [p for p in input_dir.rglob("*") if p.is_file() or p.is_symlink()]
    if video_id is None:
        if (len(files) != 1 or files[0].is_symlink() or files[0].suffix.lower() != ".mp4"
                or not files[0].resolve().is_relative_to(input_dir.resolve())):
            raise RenderError("Attach one MP4 or provide the youtube_url parameter.")
        render(files[0], output_dir, 0.2, 59.6)
        return
    if files or not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise RenderError("Choose exactly one source: an MP4 or a supported public YouTube video.")
    with tempfile.TemporaryDirectory(prefix="cpu-clip-source-") as temp:
        directory = Path(temp)
        # Fetch only the opening section; allow a margin for source frame boundaries.
        result = subprocess.run([
            sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist", "--no-cache-dir", "--no-progress",
            "--js-runtimes", "node", "--match-filters", "!is_live", "--download-sections", "*0-65",
            "-f", "bv[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/bv[height<=720][ext=mp4]+ba/bv[height<=720][ext=mp4]",
            "--merge-output-format", "mp4", "-o", str(directory / "source.%(ext)s"),
            "--", "https://www.youtube.com/watch?v=" + video_id,
        ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        source = directory / "source.mp4"
        if result.returncode or not source.is_file():
            raise RenderError("Public YouTube download unavailable. Submit an authorized MP4 attachment instead.")
        render(source, output_dir, 0.2, 59.6)


if __name__ == "__main__":
    try:
        process(Path("/job/input"), Path("/job/output"), json.load(sys.stdin)["video_id"])
    except (RenderError, OSError) as exc:
        sys.exit(f"Render failed: {exc}")
    print("STARWISH_CPU_CLIP_COMPLETE")
