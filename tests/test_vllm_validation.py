"""Offline boundary checks for local analysis; execute in the Kaggle app environment."""

from copy import deepcopy
import os
from types import SimpleNamespace

import pytest

from clipping.providers.base import InvalidOutputError, ProviderConfigError
from clipping.providers.config import normalize_base_url, vllm_config_values
from clipping.providers.parsing import clean_text, parse_clips
from clipping.providers.validation import transcript_end, validate_analysis_config, validate_clips


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", {"PYTHON_DOTENV_DISABLED": "1"})


@pytest.mark.parametrize("url", [
    "file:///tmp/model", "ftp://localhost", "http://", "http://localhost:0",
    "http://user:password@localhost:8001", "http://localhost?api_key=secret",
    "http://localhost#fragment", "http://169.254.169.254", "http://metadata.google.internal",
    "http://metadata.goog", "http://[fe80::1]", "http://[::ffff:169.254.169.254]",
    "http://0.0.0.0", "http://[::]", "http://224.0.0.1", "http://localhost:99999",
    "http://localhost/v1/v1", "http://localhost/proxy/../metadata",
    "http://localhost/%2e%2e/metadata", "http://localhost\\metadata", "http://local host",
])
def test_invalid_operator_endpoints_rejected_without_echoing_url(url):
    with pytest.raises(ProviderConfigError) as captured:
        normalize_base_url(url)
    assert "password" not in str(captured.value)
    assert "secret" not in str(captured.value)


@pytest.mark.parametrize("url,root", [
    ("http://localhost:8001/", "http://localhost:8001"),
    ("http://127.0.0.1:8001/v1/", "http://127.0.0.1:8001"),
    ("http://[::1]:8001", "http://[::1]:8001"),
    ("https://192.168.1.10/inference/v1", "https://192.168.1.10/inference"),
])
def test_loopback_lan_and_reverse_proxy_paths_supported(url, root):
    assert normalize_base_url(url) == (root, root + "/v1")


@pytest.mark.parametrize("field,value", [
    ("local_max_tokens", 0), ("local_max_tokens", -1), ("local_max_tokens", True),
    ("local_max_tokens", 1.5), ("local_max_tokens", "bad"),
    ("vllm_context_limit", 0), ("vllm_context_limit", float("inf")),
    ("vllm_timeout", float("nan")), ("vllm_timeout", -0.1),
    ("vllm_model", ""), ("vllm_model", "model with spaces"),
    ("vllm_model", "model\nname"), ("vllm_model", "x" * 257),
])
def test_invalid_local_configuration_rejected(field, value):
    with pytest.raises(ProviderConfigError):
        vllm_config_values({field: value}, environ={})


@pytest.mark.parametrize("provider,google,nvidia,allowed", [
    ("gemini", "key", "", True), ("gemini", "", "key", False),
    ("nvidia", "", "key", True), ("nvidia", "key", "", True),
    ("nvidia", "", "", False), ("vllm", "", "", True),
    ("unrecognized", "key", "key", False),
])
def test_credentials_are_checked_for_selected_provider(provider, google, nvidia, allowed):
    cfg = SimpleNamespace(ai_provider=provider, api_key_gemini=google, api_key_nvidia=nvidia)
    if allowed:
        validate_analysis_config(cfg)
    else:
        with pytest.raises(ProviderConfigError):
            validate_analysis_config(cfg)


def test_cached_analysis_needs_no_provider_or_credentials(tmp_path):
    (tmp_path / "gemini_response.json").write_text("[]", encoding="utf-8")
    cfg = SimpleNamespace(
        ai_provider="vllm", load_gemini_json=True, outputs_dir=tmp_path,
        vllm_base_url="invalid", vllm_model="",
    )
    validate_analysis_config(cfg, probe=True)


def test_missing_cache_does_not_trigger_fresh_inference(tmp_path):
    cfg = SimpleNamespace(load_gemini_json=True, outputs_dir=tmp_path, api_key_gemini="key")
    with pytest.raises(ProviderConfigError, match="Cached AI JSON"):
        validate_analysis_config(cfg)
    # Batch CLI resolves each output path later, but still validates credentials
    # if a new voiceover will be generated.
    validate_analysis_config(cfg, check_cache=False)


def test_cached_local_voiceover_still_validates_local_endpoint(tmp_path):
    (tmp_path / "gemini_response.json").write_text("[]", encoding="utf-8")
    cfg = SimpleNamespace(
        ai_provider="vllm", load_gemini_json=True, outputs_dir=tmp_path,
        voiceover=True, vllm_base_url="file:///invalid",
    )
    with pytest.raises(ProviderConfigError):
        validate_analysis_config(cfg)


def test_nvidia_voiceover_still_requires_gemini_key():
    cfg = SimpleNamespace(ai_provider="nvidia", api_key_nvidia="key", voiceover=True)
    with pytest.raises(ProviderConfigError, match="GOOGLE_API_KEY.*voiceover"):
        validate_analysis_config(cfg)


@pytest.mark.parametrize("source", [
    "No timestamps", "[0 - 0] Empty", "[-1 - 20] Negative", "[20 - 10] Reversed",
    "[0 - 20] Valid\n[nan - 100] Invalid trailing row",
    "[0 - 20] Valid\n[40 - inf] Invalid trailing row",
    "[0 - 20] Valid\n[broken timestamp] Invalid row",
])
def test_invalid_transcript_boundaries_rejected(source):
    with pytest.raises(ProviderConfigError):
        transcript_end(source)


def test_transcript_end_uses_maximum_source_boundary():
    assert transcript_end("[30.25 - 120.75] Last.\n[0 - 30.25] First.") == 120.75


@pytest.fixture
def clip():
    return {
        "start_time": 10, "end_time": 40, "viral_score": 80,
        "hook_start_time": 10, "hook_end_time": 13,
        "keep_segments": [{"start_time": 10, "end_time": 40}],
        "broll_list": [], "hook_v2": {"enabled": False, "items": []},
    }


def validate(items, **kwargs):
    return validate_clips(items, source_end=300, requested_clips=3, **kwargs)


@pytest.mark.parametrize("field,value", [
    ("start_time", -1), ("start_time", 40), ("start_time", 35),
    ("end_time", 301), ("end_time", 190), ("end_time", float("inf")),
    ("hook_start_time", 0), ("hook_end_time", 41),
    ("viral_score", 0), ("viral_score", 101), ("viral_score", True),
    ("viral_score", float("nan")), ("start_time", "10"), ("end_time", 10 ** 1000),
])
def test_invalid_timestamps_and_scores_rejected_without_clamping(clip, field, value):
    clip[field] = value
    with pytest.raises(InvalidOutputError):
        validate([clip])
    assert clip[field] is value


@pytest.mark.parametrize("field", ["keep_segments", "broll_list"])
@pytest.mark.parametrize("segments", [
    [{"start_time": 5, "end_time": 20}],
    [{"start_time": 20, "end_time": 41}],
    [{"start_time": 20, "end_time": 20}],
    [{"start_time": 10, "end_time": 25}, {"start_time": 20, "end_time": 30}],
    [{"start_time": 25, "end_time": 30}, {"start_time": 10, "end_time": 20}],
])
def test_nested_segments_stay_within_clip_and_do_not_overlap(clip, field, segments):
    clip[field] = segments
    with pytest.raises(InvalidOutputError):
        validate([clip])


def test_hook_montage_can_reorder_valid_source_segments(clip):
    clip["hook_v2"] = {"enabled": True, "items": [
        {"start_time": 30, "end_time": 33}, {"start_time": 10, "end_time": 13},
    ]}
    assert validate([clip]) == [clip]
    clip["hook_v2"]["items"][1]["end_time"] = 41
    with pytest.raises(InvalidOutputError):
        validate([clip])


def test_enabled_hook_must_have_segments(clip):
    clip["hook_v2"]["enabled"] = True
    with pytest.raises(InvalidOutputError):
        validate([clip])


def test_adjacent_clips_allowed_but_overlap_rejected(clip):
    second = deepcopy(clip)
    second.update(start_time=40, end_time=60, hook_start_time=40, hook_end_time=43)
    second["keep_segments"] = [{"start_time": 40, "end_time": 60}]
    assert validate([second, clip]) == [second, clip]
    second["start_time"] = 39
    with pytest.raises(InvalidOutputError, match="overlap"):
        validate([second, clip])


@pytest.mark.parametrize("content", ["", " ", "<think>unfinished", "```json\n{}", None])
def test_empty_or_incomplete_output_rejected(content):
    with pytest.raises(InvalidOutputError):
        clean_text(content)


@pytest.mark.parametrize("content", [
    '{"clips": [NaN]}', '{"clips": [Infinity]}', '{"clips": [-Infinity]}',
    '{"clips": [}', '[]', '{"data": []}', '{"clips": [], "secret": "private-output"}',
])
def test_json_parser_requires_schema_and_finite_json_numbers(content):
    schema = {
        "type": "object", "properties": {"clips": {"type": "array", "minItems": 1}},
        "required": ["clips"], "additionalProperties": False,
    }
    with pytest.raises(InvalidOutputError) as captured:
        parse_clips(content, schema)
    assert "private-output" not in str(captured.value)
