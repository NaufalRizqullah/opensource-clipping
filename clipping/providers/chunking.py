"""Plan whole-source vLLM requests on timestamp boundaries."""

from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache

from .base import ProviderConfigError
from .validation import _STAMP, transcript_end


@dataclass(frozen=True)
class TranscriptWindow:
    text: str
    start: float
    end: float


def plan_windows(transcript, fits, *, min_duration, overlap_seconds):
    """Cover each cue, testing complete requests with the served model's tokenizer."""
    transcript_end(transcript)
    stamps = list(_STAMP.finditer(transcript))
    starts = [float(match[1]) for match in stamps]
    ends = [float(match[2]) for match in stamps]
    if transcript[:stamps[0].start()].strip() or any(a > b for a, b in zip(starts, starts[1:])):
        raise ProviderConfigError("Chunking memerlukan transkrip bertimestamp yang terurut.")

    # Slice the original string so continuations, whitespace and absolute timestamps survive.
    offsets = [match.start() for match in stamps] + [len(transcript)]
    offsets[0] = 0
    total = len(starts)

    def window(first, stop):
        return TranscriptWindow(transcript[offsets[first]:offsets[stop]], starts[first], max(ends[first:stop]))

    @lru_cache(maxsize=256)
    def accepts(first, stop):
        return fits(window(first, stop))

    def seed(first, cursor):
        """Smallest window from first containing new material and enough seconds for a clip."""
        stop = cursor + 1
        end = max(ends[first:stop])
        while end - starts[first] < min_duration and stop < total:
            end = max(end, ends[stop])
            stop += 1
        if end - starts[first] >= min_duration and accepts(first, stop):
            return stop
        return None

    def extend(first, stop):
        # Exponential search finds a nearby upper bound even for very long transcripts.
        step = stop - first
        while stop < total:
            high = min(total, stop + step)
            if not accepts(first, high):
                while high - stop > 1:
                    middle = (stop + high) // 2
                    if accepts(first, middle):
                        stop = middle
                    else:
                        high = middle
                break
            stop = high
            step *= 2
        return stop

    planned = []
    cursor = 0
    last_first = -1
    while cursor < total:
        # Carry context over the cut but reserve space for new material on small servers.
        floor = last_first + 1
        half_back = cursor - (cursor - floor + 1) // 2
        time_cut = bisect_left(starts, starts[cursor] - overlap_seconds, lo=floor, hi=cursor)
        first = max(half_back, time_cut)
        stop = seed(first, cursor)
        if stop is None and accepts(cursor, cursor + 1):
            # Dense speech can exhaust the budget even with the proposed overlap. A
            # short final tail can instead need *more* prior cues to reach min_duration.
            for alternate in range(cursor, floor - 1, -1):
                if alternate != first:
                    stop = seed(alternate, cursor)
                    if stop is not None:
                        first = alternate
                        break
        if stop is None:
            raise ProviderConfigError(
                f"Bagian transkrip pada {starts[cursor]:g} detik tidak muat dalam konteks "
                f"bersama instruksi, anggaran output, dan durasi minimum klip {min_duration:g} detik. "
                "Periksa kepadatan timestamp, anggaran output, atau batas konteks server."
            )
        stop = extend(first, stop)
        planned.append(window(first, stop))
        cursor, last_first = stop, first
    return planned


def select_candidates(candidates, requested_clips):
    """Compare candidates on the same source timeline and renumber selected clips."""
    ordered = sorted(candidates, key=lambda clip: (-clip["viral_score"], clip["start_time"], clip["end_time"]))
    selected = []
    for clip in ordered:
        if any(clip["start_time"] < other["end_time"] and other["start_time"] < clip["end_time"]
               for other in selected):
            continue
        selected.append(dict(clip, rank=len(selected) + 1))
        if len(selected) == requested_clips:
            break
    return selected
