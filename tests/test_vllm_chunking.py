"""Kaggle regression tests for full-source transcript coverage; no model required."""

import re

import pytest

from clipping.providers.base import ProviderConfigError
from clipping.providers.chunking import plan_windows, select_candidates


def cues(count, seconds=10):
    return [f"[{i * seconds} - {(i + 1) * seconds}] cue-{i}\n" for i in range(count)]


def test_50k_transcript_fits_16k_requests_and_keeps_every_cue():
    lines = cues(500)
    # 50,000 simulated transcript tokens plus fixed prompt and completion tokens.
    def fits(window):
        return 2000 + 100 * len(window.text.splitlines()) + 4096 <= 16384

    windows = plan_windows("".join(lines), fits, min_duration=20, overlap_seconds=179)
    assert 1 < len(windows) < 10
    covered = set()
    for part in windows:
        rows = part.text.splitlines(keepends=True)
        first = lines.index(rows[0])
        stop = first + len(rows)
        assert rows == lines[first:stop]
        assert (part.start, part.end) == (10 * first, 10 * stop)
        assert fits(part)
        covered.update(range(first, stop))
    assert covered == set(range(500))
    assert windows[-1].end == 5000
    assert all(a.start < b.start < a.end < b.end for a, b in zip(windows, windows[1:]))


def test_short_tail_borrows_earlier_speech_when_overlap_is_zero():
    lines = cues(5)
    windows = plan_windows("".join(lines), lambda w: len(w.text.splitlines()) <= 4,
                           min_duration=20, overlap_seconds=0)
    assert [w.text for w in windows] == ["".join(lines[:4]), "".join(lines[3:])]


def test_reduces_overlap_to_include_a_large_new_cue():
    lines = cues(6, seconds=30)
    def fits(window):
        cost = sum(5 if "cue-4" in line else 1 for line in window.text.splitlines())
        return cost <= 5

    windows = plan_windows("".join(lines), fits, min_duration=20, overlap_seconds=179)
    assert [w.text for w in windows] == ["".join(lines[:4]), lines[4], lines[5]]


def test_multiline_cues_preserve_original_text_and_timestamp_bounds():
    source = "\r\n[10 - 75] alpha\r\ncontinued\r\n[20 - 60] beta\r\n"
    windows = plan_windows(source, lambda _: True, min_duration=20, overlap_seconds=30)
    assert [(w.text, w.start, w.end) for w in windows] == [(source, 10, 75)]


@pytest.mark.parametrize("lines,limit,error", [
    (["[0 - 30] private " + "x" * 300, "[30 - 60] ending"], 60, "0 detik"),
    (cues(6) + ["[60 - 90] private " + "x" * 300], 60, "60 detik"),
    (["[0 - 10] private"], 100, "durasi minimum"),
])
def test_cannot_fit_one_required_window_fails_without_transcript_text(lines, limit, error):
    transcript = "\n".join(line.rstrip("\n") for line in lines)
    with pytest.raises(ProviderConfigError, match=re.escape(error)) as raised:
        plan_windows(transcript, lambda w: len(w.text) <= limit, min_duration=20, overlap_seconds=30)
    assert "private" not in str(raised.value)


@pytest.mark.parametrize("transcript", [
    "[40 - 60] later\n[10 - 30] earlier",
    "[0 - 30] first\n[bad - 40] malformed",
    "unattributed preface\n[0 - 30] first",
])
def test_invalid_timestamp_order_or_unassigned_text_fails(transcript):
    with pytest.raises(ProviderConfigError):
        plan_windows(transcript, lambda _: True, min_duration=20, overlap_seconds=0)


def test_global_selection_can_choose_last_window_and_preserves_input_ranks():
    def candidate(start, end, score):
        return dict(start_time=start, end_time=end, viral_score=score, rank=1)

    candidates = [candidate(25, 45, 85), candidate(0, 25, 85), candidate(120, 160, 99),
                  candidate(115, 140, 88), candidate(120, 160, 99)]
    assert select_candidates(candidates, 1) == [candidates[2]]
    selected = select_candidates(candidates, 5)
    assert [c["start_time"] for c in selected] == [120, 0, 25]
    assert [c["rank"] for c in selected] == [1, 2, 3]
    assert all(c["rank"] == 1 for c in candidates)
