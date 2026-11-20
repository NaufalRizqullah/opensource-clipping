"""
General helper utilities for Studio rendering workflow.
"""

import os


def format_seconds(seconds):
    """
    Format a duration in seconds into HH:MM:SS.

    Args:
        seconds: Numeric duration in seconds.

    Returns:
        Duration string in `HH:MM:SS` format, clamped to non-negative.
    """
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def escape_ffmpeg_filter_value(value: str) -> str:
    """
    Escape a value so it is safe in FFmpeg filter expressions.

    Args:
        value: Raw value to place inside an FFmpeg filter string.

    Returns:
        Escaped value string for FFmpeg filter usage.
    """
    value = str(value)
    if os.name == "nt":
        # Windows: "E:\dir\file" breaks -vf parsing. Use forward slashes and
        # escape ":" twice (filtergraph level + filter-option level).
        return value.replace("\\", "/").replace(":", r"\\:").replace("'", r"\'")
    return value.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")

