"""Normalize requested source timestamps before sending them to the media tools."""

import math
import re


_CLOCK = re.compile(r"(?:(?P<hours>[0-9]+):)?(?P<minutes>[0-9]+):(?P<seconds>[0-5][0-9](?:\.[0-9]+)?)")


def _seconds(value):
    if isinstance(value, str):
        clock = _CLOCK.fullmatch(value)
        if not clock:
            raise ValueError("Use seconds as a number, or a timestamp such as 06:00 or 01:06:00.5.")
        hours, minutes, seconds = clock.groups()
        if hours is not None and (len(minutes) != 2 or int(minutes) >= 60):
            raise ValueError("Minutes in an HH:MM:SS timestamp must be between 00 and 59.")
        seconds = float(hours or 0) * 3600 + float(minutes) * 60 + float(seconds)
    elif type(value) in (int, float):
        try:
            seconds = float(value)
        except OverflowError as exc:
            raise ValueError("Timestamp exceeds the supported numeric range.") from exc
    else:
        raise ValueError("Each timestamp must be a number or a clock string.")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Timestamps must be finite and nonnegative.")
    return seconds


def normalize_ranges(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 3:
        raise ValueError("Provide between one and three ranges.")
    result = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict) or item.keys() != {"start", "end"}:
            raise ValueError(f"Range {index} requires exactly start and end.")
        start, end = _seconds(item["start"]), _seconds(item["end"])
        if end <= start:
            raise ValueError(f"Range {index} must end after its start.")
        result.append({"start": start, "end": end})
    return result
