"""
v2_helpers - Hook V2 transition generators.

Creates white-flash and glitch transition clips using FFmpeg lavfi filters.
These are short (0.1-0.3s) videos designed to be concatenated between micro-hook
clips in the Hook V2 multi-intro pipeline.
"""

import subprocess


def create_white_flash_transition(
    output_path: str,
    duration: float = 0.12,
    fps: int = 30,
    width: int = 1080,
    height: int = 1920,
) -> str:
    """Generate a solid white flash clip."""
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c=white:s={width}x{height}:d={duration}:r={fps}",
        "-f", "lavfi",
        "-i", f"anullsrc=r=48000:cl=stereo",
        "-t", str(duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return output_path


def create_glitch_transition(
    output_path: str,
    duration: float = 0.12,
    fps: int = 30,
    width: int = 1080,
    height: int = 1920,
) -> str:
    """Generate an RGB-shift glitch noise clip."""
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c=black:s={width}x{height}:d={duration}:r={fps}",
        "-f", "lavfi",
        "-i", f"anullsrc=r=48000:cl=stereo",
        "-t", str(duration),
        "-vf", "noise=alls=100:allf=t+u,rgbashift=rh=20:bv=20",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return output_path
