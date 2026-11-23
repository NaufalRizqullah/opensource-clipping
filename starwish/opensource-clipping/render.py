"""Render one local MP4 on the CPU for a Starwish job."""

import argparse
import errno
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


WIDTH, HEIGHT = 720, 1280
FILES = ("clip.mp4", "thumbnail.jpg", "metadata.json")
FFMPEG = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror"]
# Accept only local MP4/MOV demuxer input. Never follow references to other files.
MP4_INPUT = ["-protocol_whitelist", "file", "-f", "mov", "-enable_drefs", "0"]
SAR = "if(gt(sar,0),sar,1)"
PORTRAIT_FILTER = (
    f"crop=w='max(2,trunc(min(iw,ih*9/16/{SAR})/2)*2)':"
    f"h='max(2,trunc(min(ih,iw*{SAR}*16/9)/2)*2)',"
    f"scale={WIDTH}:{HEIGHT},setsar=1,fps=30"
)


class RenderError(Exception):
    pass


def run_media(command, stage):
    with tempfile.TemporaryFile() as errors:
        completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors)
        if completed.returncode:
            errors.seek(max(0, errors.tell() - 2048))
            detail = errors.read().decode("utf-8", errors="replace").strip()
            message = f"{stage} failed (exit {completed.returncode})."
            raise RenderError(f"{message}\n{detail}" if detail else message)
    return completed.stdout


def probe(path):
    entries = "format=duration:stream=index,codec_type,codec_name,width,height,duration:stream_disposition=attached_pic"
    options = MP4_INPUT if path.suffix.lower() == ".mp4" else []
    return json.loads(run_media([
        "ffprobe", "-v", "error", *options, "-show_entries", entries, "-of", "json", str(path)
    ], "Media inspection"))


def video_stream(media):
    for stream in media.get("streams", ()):
        if stream.get("codec_type") == "video" and not stream.get("disposition", {}).get("attached_pic"):
            return stream
    raise RenderError("No video track found.")


def duration(media, video):
    for value in (video.get("duration"), media.get("format", {}).get("duration")):
        if value not in (None, "N/A"):
            seconds = float(value)
            if math.isfinite(seconds) and seconds > 0:
                return seconds
    raise RenderError("The video duration is unavailable.")


def publish(stage, output_dir):
    # Jobs may mount /tmp and /job/output on different filesystems. Deliver metadata last.
    created = []
    try:
        for name in FILES:
            dest = output_dir / name
            try:
                os.link(stage / name, dest)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                with (stage / name).open("rb") as src, dest.open("xb") as target:
                    created.append(dest)
                    shutil.copyfileobj(src, target)
            else:
                created.append(dest)
    except OSError:
        for dest in created:
            dest.unlink()
        raise


def render(source, output_dir, start, end):
    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
        raise RenderError("Use a finite interval with 0 <= start < end (seconds).")
    source = source.resolve()
    if source.suffix.lower() != ".mp4" or not source.is_file():
        raise RenderError("Provide a local MP4 file.")
    media = probe(source)
    original = video_stream(media)
    if end > duration(media, original):
        raise RenderError("The requested end exceeds the source video duration.")
    has_audio = any(s["codec_type"] == "audio" for s in media["streams"])
    length = end - start

    output_dir.mkdir(parents=True, exist_ok=True)
    if any(os.path.lexists(output_dir / name) for name in FILES):
        raise RenderError("Clip outputs already exist in the job output directory.")
    with tempfile.TemporaryDirectory(prefix="cpu-video-render-") as temp:
        stage = Path(temp)
        clip, thumb, metadata = (stage / name for name in FILES)
        run_media([
            *FFMPEG, "-filter_threads", "2", "-threads", "2", "-ss", f"{start:.6f}", *MP4_INPUT,
            "-i", str(source), "-t", f"{length:.6f}", "-map", f"0:{original['index']}", "-map", "0:a:0?",
            "-map_metadata", "-1", "-map_chapters", "-1", "-vf", PORTRAIT_FILTER,
            "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(clip),
        ], "CPU clip encoding")
        result = probe(clip)
        video = video_stream(result)
        actual = duration(result, video)
        audio = next((s for s in result["streams"] if s["codec_type"] == "audio"), None)
        if (video["width"], video["height"], video["codec_name"]) != (WIDTH, HEIGHT, "h264"):
            raise RenderError("The rendered video has unexpected dimensions or codec.")
        if has_audio and (audio is None or audio["codec_name"] != "aac"):
            raise RenderError("The source audio track was not preserved.")
        if abs(actual - length) > 0.1:
            raise RenderError("The rendered duration does not match the requested interval.")
        run_media([*FFMPEG, "-i", str(clip), "-map", "0:v:0", "-map", "0:a:0?", "-f", "null", "-"],
                  "Clip decode verification")
        run_media([*FFMPEG, "-i", str(clip), "-map", "0:v:0", "-frames:v", "1", "-q:v", "2", str(thumb)],
                  "Thumbnail encoding")
        image = video_stream(probe(thumb))
        if (image["codec_name"], image["width"], image["height"]) != ("mjpeg", WIDTH, HEIGHT):
            raise RenderError("The thumbnail failed verification.")
        metadata.write_text(json.dumps({
            "start_seconds": start, "end_seconds": end, "duration_seconds": actual,
            "width": WIDTH, "height": HEIGHT, "has_audio": has_audio,
            "video": clip.name, "thumbnail": thumb.name,
        }, indent=2) + "\n", encoding="utf-8")
        publish(stage, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.2, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=59.6, help="End time in seconds")
    args = parser.parse_args()
    try:
        render(args.input, args.output_dir, args.start, args.end)
    except RenderError as exc:
        parser.exit(1, f"{exc}\n")
    except OSError as exc:
        parser.exit(1, f"Workspace error: {exc.strerror or type(exc).__name__}.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
