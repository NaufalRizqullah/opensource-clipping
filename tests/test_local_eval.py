"""Evaluation reports must measure warnings rather than count discarded fields."""

from copy import deepcopy
import json
import os
import sys
from types import ModuleType

from clipping.metadata import normalize_and_validate
from scripts import eval_local_llm


def test_optional_warning_report_preserves_existing_normalized_output():
    clips = [{"rank": 1, "start_time": 0, "end_time": 30, "viral_score": 50}]
    report = []
    expected = normalize_and_validate(deepcopy(clips))
    actual = normalize_and_validate(deepcopy(clips), report=report)
    assert actual == expected
    assert len(report) == 1
    assert "title_inggris kosong" in report[0]["warnings"]
    assert "_warnings_temp" not in actual[0]


def test_eval_counts_metadata_warnings_and_keeps_raw_result(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(os, "environ", {})
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("[0 - 30] Synthetic source", encoding="utf-8")
    output = tmp_path / "report.json"
    raw = [{"rank": 1, "start_time": 0, "end_time": 30, "viral_score": 50}]
    engine = ModuleType("clipping.engine")

    def analyze(source, cfg):
        assert source == "[0 - 30] Synthetic source"
        cfg.ai_run_meta = {"actual_provider": "vllm", "status": "completed"}
        return raw

    engine.analyze_with_vllm = analyze
    monkeypatch.setitem(sys.modules, "clipping.engine", engine)
    assert eval_local_llm.main(["--transcript", str(transcript), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["clips_with_metadata_warnings"] == 1
    assert report["clips"] == raw
    assert "youtube_title_final" not in raw[0]
    assert "Synthetic source" not in capsys.readouterr().out
