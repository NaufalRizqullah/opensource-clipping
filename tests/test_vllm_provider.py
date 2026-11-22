"""Real OpenAI SDK requests over an in-memory transport; run these in Kaggle."""

from collections import deque
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from clipping.providers.base import (
    ContextOverflowError, InvalidOutputError, ProviderConfigError, ProviderUnavailableError,
)
from clipping.providers.config import DEFAULT_VLLM_MODEL
from clipping.providers.schema import build_clips_schema
from clipping.providers.vllm import VllmProvider


TRANSCRIPT = "[0.00 - 30.00] Pembuka.\n[30.00 - 120.00] Pembahasan utama."


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", {
        "PYTHON_DOTENV_DISABLED": "1", "OPENAI_API_KEY": "unrelated-cloud-key",
    })

    def forbidden(*args, **kwargs):
        pytest.fail("Provider contract tests must never open a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def schema():
    return build_clips_schema({"education": {}})


@pytest.fixture
def clip():
    return {
        "rank": 1, "viral_score": 85, "start_time": 10, "end_time": 40,
        "hook_start_time": 10, "hook_end_time": 13, "bgm_mood": "chill",
        "typography_plan": [], "broll_list": [], "recommended_visual_broll_hook": [],
        "title_indonesia": "Belajar bersama", "title_inggris": "Learning together",
        "hastag": "#belajar", "description_hook": "Satu pelajaran penting",
        "description_context": "Pembahasan cara belajar", "keyword_tags": ["learning"],
        "tiktok_title_id": "Belajar", "tiktok_caption_id": "Ayo belajar",
        "tiktok_caption": "Keep learning", "alasan": "Pesan lengkap",
        "klasifikasi_akun": {
            "tipe_akun": "education", "akun_tujuan": "Belajar", "confidence": 90,
            "angle_utama": "Pendidikan", "alasan": "Relevan", "kata_kunci_pendukung": [],
            "bio_akun": "Belajar bersama", "alternatif_akun": {
                "tipe_akun": "education", "akun_tujuan": "Belajar", "alasan": "Relevan",
            },
        },
        "hook_v2": {"enabled": False, "items": [], "transition": {"type": "glitch"}},
        "keep_segments": [{"start_time": 10, "end_time": 40}],
    }


def completion(content, *, finish_reason="stop"):
    return {
        "id": "local-test", "object": "chat.completion", "created": 1, "model": "served-alias",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 64, "completion_tokens": 128, "total_tokens": 192},
    }


@pytest.fixture
def provider_factory():
    clients = []

    def create(outcomes, *, base_url="http://inference.test:8001", key="synthetic-local-key",
               model_limit=8192, token_limit=8192, count=64, context=None,
               max_tokens=512, timeout=900, models_status=200, tokenize_status=200,
               model_outcomes=None, token_outcomes=None, on_completion=None):
        state = SimpleNamespace(requests=[], completions=[], tokens=[], remaining=deque(outcomes),
                                time=0.0, delays=[], model_outcomes=deque(model_outcomes or []),
                                token_outcomes=deque(token_outcomes or []))

        def response_for(outcome):
            if isinstance(outcome, Exception):
                raise outcome
            status, body = outcome if isinstance(outcome, tuple) else (200, outcome)
            return httpx.Response(status, json=body)

        def respond(request):
            state.requests.append(request)
            path = request.url.path
            if path.endswith("/v1/models"):
                if state.model_outcomes:
                    return response_for(state.model_outcomes.popleft())
                return httpx.Response(models_status, json={
                    "data": [{"id": "served-alias", "max_model_len": model_limit}],
                })
            if path.endswith("/tokenize"):
                state.tokens.append(json.loads(request.content))
                if state.token_outcomes:
                    return response_for(state.token_outcomes.popleft())
                tokens = count(state.tokens[-1]["messages"]) if callable(count) else count
                return httpx.Response(tokenize_status, json={"count": tokens, "max_model_len": token_limit})
            assert path.endswith("/v1/chat/completions"), str(request.url)
            state.completions.append(json.loads(request.content))
            if on_completion is not None:
                return response_for(on_completion(state.completions[-1], len(state.completions)))
            assert state.remaining, "Unexpected SDK retry or extra completion request"
            return response_for(state.remaining.popleft())

        def sleep(seconds):
            state.delays.append(seconds)
            state.time += seconds

        client = httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=False, trust_env=False)
        clients.append(client)
        cfg = SimpleNamespace(
            ai_provider="vllm", vllm_model="served-alias", vllm_base_url=base_url,
            api_key_vllm=key, local_max_tokens=max_tokens, vllm_context_limit=context,
            vllm_timeout=timeout,
        )
        provider = VllmProvider(cfg, http_client=client, sleep=sleep, clock=lambda: state.time)
        state.provider, state.cfg = provider, cfg
        return state

    yield create
    for client in clients:
        client.close()


def generate(state, schema, *, requested=1, prompt="Analyze transcript"):
    return state.provider.generate_clips(
        prompt, schema, transcript=TRANSCRIPT, requested_clips=requested,
    )


@pytest.mark.parametrize("base,prefix", [
    ("http://inference.test:8001", ""),
    ("http://inference.test:8001/v1/", ""),
    ("http://inference.test:8001/proxy/v1", "/proxy"),
])
@pytest.mark.parametrize("key,expected_key", [("synthetic-local-key", "synthetic-local-key"), ("", "local-vllm")])
def test_sdk_paths_auth_schema_and_token_template(provider_factory, schema, clip, base, prefix, key, expected_key):
    state = provider_factory([completion(json.dumps({"clips": [clip]}))], base_url=base, key=key)
    original_schema = deepcopy(schema)
    assert generate(state, schema) == [clip]
    assert [request.url.path for request in state.requests] == [
        prefix + "/v1/models", prefix + "/tokenize", prefix + "/v1/chat/completions",
    ]
    assert all(r.headers["Authorization"] == "Bearer " + expected_key for r in state.requests)
    body = state.completions[0]
    assert body["model"] == "served-alias"
    assert body["max_tokens"] == 512
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert state.tokens[0]["messages"] == body["messages"]
    assert state.tokens[0]["chat_template_kwargs"] == body["chat_template_kwargs"]
    assert state.tokens[0]["add_generation_prompt"] is True
    response_schema = body["response_format"]["json_schema"]
    assert body["response_format"]["type"] == "json_schema"
    assert response_schema["strict"] is True
    assert response_schema["schema"]["properties"]["clips"]["maxItems"] == 1
    assert "guided_json" not in body and "nvext" not in body
    assert schema == original_schema
    assert state.provider._client.max_retries == 0
    assert state.cfg.ai_run_meta["actual_provider"] == "vllm"
    assert state.cfg.ai_run_meta["fallback_used"] is False
    assert state.cfg.ai_run_meta["usage"][0]["total_tokens"] == 192


def test_from_cfg_uses_local_defaults_without_probing():
    with VllmProvider.from_cfg(SimpleNamespace()) as provider:
        assert provider.model == DEFAULT_VLLM_MODEL
        assert provider.api_base == "http://127.0.0.1:8001/v1"
        assert provider._client.max_retries == 0


@pytest.mark.parametrize("model_limit,token_limit,context", [(None, None, 1000), (1000, None, None), (None, 1000, None)])
def test_context_limit_fallback_when_server_fields_are_null(provider_factory, schema, clip, model_limit, token_limit, context):
    state = provider_factory([completion(json.dumps({"clips": [clip]}))],
                             model_limit=model_limit, token_limit=token_limit, context=context)
    assert generate(state, schema) == [clip]
    assert state.cfg.ai_run_meta["context_limit"] == 1000


def test_unknown_context_requires_operator_limit(provider_factory, schema):
    state = provider_factory([], model_limit=None, token_limit=None)
    with pytest.raises(ProviderConfigError, match="VLLM_CONTEXT_LIMIT"):
        generate(state, schema)
    assert not state.completions


def test_context_checks_prompt_plus_output_against_smallest_limit(provider_factory, schema):
    state = provider_factory([], count=600, model_limit=4096, token_limit=1024, context=900)
    with pytest.raises(ContextOverflowError, match=r"600 \+ output 512 > 900.*menyisakan 300 token"):
        generate(state, schema)
    assert not state.completions
    assert state.cfg.ai_run_meta["attempts"] == 0


@pytest.mark.parametrize("count", [8192, 53459])
def test_prompt_exhausting_context_reports_required_capacity_before_inference(provider_factory, schema, count):
    state = provider_factory([], count=count, token_limit=8192, max_tokens=2048)
    with pytest.raises(ContextOverflowError, match=rf"input yang lebih pendek.*minimal {count + 2048} token"):
        generate(state, schema)
    assert not state.completions
    meta = state.cfg.ai_run_meta
    assert (meta["prompt_tokens"], meta["context_limit"], meta["max_tokens"]) == (count, 8192, 2048)
    assert meta["error_stage"] == "tokenize" and meta["attempts"] == 0


def test_exact_context_budget_is_accepted(provider_factory, schema, clip):
    state = provider_factory([completion(json.dumps({"clips": [clip]}))], count=512, token_limit=1024)
    assert generate(state, schema) == [clip]


@pytest.mark.parametrize("fail_second_window", [False, True], ids=["whole-video", "partial-failure"])
def test_long_source_uses_model_tokenizer_and_commits_only_complete_analysis(
    provider_factory, schema, clip, fail_second_window,
):
    # Simulate the served model counting 1,000 tokens per caption and 512 for
    # the instructions/template. Only the server knows the size of each caption.
    def source_ids(messages):
        return [
            int(i) for message in messages
            for i in re.findall(r"caption-(\d+)", message["content"])
        ]

    def token_count(messages):
        return 512 + 1000 * len(source_ids(messages))

    def answer(body, call_number):
        if fail_second_window and call_number == 2:
            return 401, {"error": {"message": "sensitive server response"}}
        final_id = max(source_ids(body["messages"]))
        start = 120 * final_id
        candidate = dict(
            clip, start_time=start, end_time=start + 30, hook_start_time=start,
            hook_end_time=start + 3, keep_segments=[], viral_score=99 if final_id == 49 else 75,
        )
        return completion(json.dumps({"clips": [candidate]}))

    state = provider_factory(
        [], model_limit=16384, token_limit=16384, count=token_count, max_tokens=4096,
        on_completion=answer,
    )
    state.cfg.vllm_chunk_transcript = True
    transcript = "".join(f"[{120*i} - {120*(i+1)}] caption-{i}\n" for i in range(50))
    args = (f"Select one highlight.\nTranskrip:\n{transcript}\n", schema)
    kwargs = dict(transcript=transcript, requested_clips=1)
    if fail_second_window:
        with pytest.raises(ProviderConfigError, match="Autentikasi"):
            state.provider.generate_clips(*args, **kwargs)
    else:
        winner, = state.provider.generate_clips(*args, **kwargs)
        assert (winner["start_time"], winner["viral_score"], winner["rank"]) == (120 * 49, 99, 1)

    covered = set()
    for previous, current in zip(state.requests, state.requests[1:]):
        if not current.url.path.endswith("/chat/completions"):
            continue
        sent, measured = json.loads(current.content), json.loads(previous.content)
        assert previous.url.path.endswith("/tokenize")
        assert measured["messages"] == sent["messages"]
        assert measured["chat_template_kwargs"] == sent["chat_template_kwargs"]
        assert token_count(measured["messages"]) + sent["max_tokens"] <= 16384
        covered.update(source_ids(sent["messages"]))

    meta = state.cfg.ai_run_meta
    assert meta["mode"] == "chunked" and meta["full_prompt_tokens"] == 50512
    assert meta["chunks_total"] > 1 and meta["attempts"] == len(state.completions)
    assert len(meta["windows"]) == meta["chunks_completed"]
    for record, request in zip(meta["windows"], state.completions):
        ids = source_ids(request["messages"])
        assert (record["start"], record["end"]) == (120 * min(ids), 120 * (max(ids) + 1))
        assert record["prompt_tokens"] == token_count(request["messages"])
    assert "caption-" not in json.dumps(meta)
    if fail_second_window:
        assert meta["status"] == "failed" and meta["failed_chunk"] == 2
        assert meta["chunks_completed"] == 1 and meta["candidate_count"] == 1
        assert "returned_clips" not in meta and len(state.completions) == 2
    else:
        assert covered == set(range(50))
        assert meta["status"] == "completed" and meta["returned_clips"] == 1
        assert meta["chunks_completed"] == meta["chunks_total"] == len(state.completions)
        assert meta["candidate_count"] == len(meta["usage"]) == len(state.completions)


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_opt_in_fitting_prompt_stays_one_request(provider_factory, schema, clip, finish_reason):
    state = provider_factory([completion(json.dumps({"clips": [clip]}), finish_reason=finish_reason)])
    state.cfg.vllm_chunk_transcript = True
    if finish_reason == "length":
        with pytest.raises(ContextOverflowError, match="terpotong"):
            generate(state, schema)
    else:
        assert generate(state, schema) == [clip]
    assert state.cfg.ai_run_meta["mode"] == "single_pass"
    assert len(state.tokens) == len(state.completions) == 1


def _window_captions(messages):
    return [int(index) for message in messages
            for index in re.findall(r"private-caption-(\d+)", message["content"])]


def _clip_fields(body):
    clips = body["response_format"]["json_schema"]["schema"]["properties"]["clips"]
    return clips["items"]["properties"]


@pytest.fixture
def shortlist_run(provider_factory, schema):
    # Budget a seven-cue search window. The full video contains 18 consecutive
    # 30-second cues, one of which identifies the best highlight at the very end.
    transcript = "".join(f"[{i * 30} - {(i + 1) * 30}] private-caption-{i}\n" for i in range(18))

    def configure(answer):
        def count(messages):
            return 512 + 1000 * len(_window_captions(messages))

        state = provider_factory([], model_limit=8192, token_limit=8192, count=count,
                                 max_tokens=2048, on_completion=answer)
        state.cfg.vllm_chunk_transcript = True
        state.cfg.vllm_shortlist = True

        def run():
            return state.provider.generate_clips(
                f"Select a highlight.\nTranskrip:\n{transcript}\n", schema,
                transcript=transcript, requested_clips=1,
            )

        return state, run

    return configure


@pytest.mark.parametrize("failure", [None, "hook-outside", "publishing-length"])
def test_shortlist_reviews_entire_video_then_details_only_the_winner(shortlist_run, clip, failure):
    reviewed_cues = set()
    shortlist_requests = []
    detail_requests = []

    def answer(body, _call):
        fields = _clip_fields(body)
        ids = _window_captions(body["messages"])
        assert ids
        assert 512 + 1000 * len(ids) + body["max_tokens"] <= 8192
        if "choice" in fields:
            assert set(fields) == {"choice", "viral_score"}
            assert body["max_tokens"] <= 128
            assert not detail_requests
            reviewed_cues.update(ids)
            shortlist_requests.append(ids)
            item = {"choice": max(fields["choice"]["enum"]), "viral_score": 98 if 17 in ids else 70}
            return completion(json.dumps({"clips": [item]}))

        # Full caption coverage precedes even the first metadata request.
        assert reviewed_cues == set(range(18))
        assert 17 in ids and 0 not in ids
        assert not {"rank", "viral_score", "start_time", "end_time"}.intersection(fields)
        start = min(ids) * 30
        end = (max(ids) + 1) * 30
        candidate = dict(clip, hook_start_time=start, hook_end_time=start + 2, keep_segments=[])
        detail_requests.append((set(fields), set(ids)))
        if failure == "hook-outside" and "hook_end_time" in fields:
            candidate["hook_end_time"] = end + 1
        item = {key: candidate[key] for key in fields}
        truncated = failure == "publishing-length" and "title_inggris" in fields
        return completion(json.dumps({"clips": [item]}), finish_reason="length" if truncated else "stop")

    state, run = shortlist_run(answer)
    if failure:
        error = InvalidOutputError if failure == "hook-outside" else ContextOverflowError
        with pytest.raises(error):
            run()
        assert state.cfg.ai_run_meta["status"] == "failed"
        assert "returned_clips" not in state.cfg.ai_run_meta
    else:
        winner, = run()
        assert len(detail_requests) == 2
        (editing, first_cues), (publishing, second_cues) = detail_requests
        assert not editing.intersection(publishing)
        assert set(winner) == set(clip) == editing | publishing | {"rank", "viral_score", "start_time", "end_time"}
        assert first_cues == second_cues
        assert (winner["rank"], winner["viral_score"], winner["end_time"]) == (1, 98, 540)
        assert first_cues == set(range(int(winner["start_time"] // 30), 18))
        assert winner["title_inggris"] == clip["title_inggris"]
        assert state.cfg.ai_run_meta["status"] == "completed"
    meta = state.cfg.ai_run_meta
    assert meta["strategy"] == "shortlist" and meta["mode"] == "chunked"
    assert meta["chunks_total"] == meta["chunks_completed"] == len(shortlist_requests) > 1
    assert meta["candidate_count"] == len(shortlist_requests)
    assert "private-caption-" not in json.dumps(meta)


@pytest.mark.parametrize("bad", ["truncated", "invalid-json", "unoffered"])
def test_invalid_shortlist_choice_never_finishes_a_window(shortlist_run, bad):
    def answer(body, call):
        fields = _clip_fields(body)
        choices = fields["choice"]["enum"]
        item = {"choice": choices[0], "viral_score": 88}
        if call == 1:
            return completion(json.dumps({"clips": [item]}))
        if bad == "invalid-json":
            return completion('{"clips": [private-model-output')
        if bad == "unoffered":
            item["choice"] = max(choices) + 1
        return completion(json.dumps({"clips": [item]}), finish_reason="length" if bad == "truncated" else "stop")

    state, run = shortlist_run(answer)
    with pytest.raises(ContextOverflowError if bad == "truncated" else InvalidOutputError):
        run()
    meta = state.cfg.ai_run_meta
    assert meta["status"] == "failed" and meta["failed_chunk"] == 2
    assert meta["chunks_completed"] == meta["candidate_count"] == 1
    assert "returned_clips" not in meta
    assert "private-model-output" not in json.dumps(meta)


def test_shortlist_uses_compact_request_even_when_full_prompt_fits(provider_factory, schema):
    state = provider_factory([completion('{"clips":[', finish_reason="length")])
    state.cfg.vllm_chunk_transcript = state.cfg.vllm_shortlist = True
    with pytest.raises(ContextOverflowError):
        generate(state, schema, prompt=f"Select one clip.\nTranskrip:\n{TRANSCRIPT}")
    first, = state.completions
    assert set(_clip_fields(first)) == {"choice", "viral_score"}
    assert state.cfg.ai_run_meta["chunks_completed"] == 0


def _window_reply(clip, start, end, *, score=75, **changes):
    candidate = dict(clip, start_time=start, end_time=end, hook_start_time=start,
                     hook_end_time=start + 2, keep_segments=[], viral_score=score)
    candidate.update(changes)
    return completion(json.dumps({"clips": [candidate]}))


@pytest.fixture
def chunk_duration_run(provider_factory, schema):
    def prepare(answer, *, cue_seconds=60, requested=1):
        # Six cues, four fit at once. Every initial window is long enough that
        # a duration error can occur even when both endpoints are in bounds.
        def count(messages):
            return 100 + 1000 * len(_window_captions(messages))

        state = provider_factory([], model_limit=5500, token_limit=5500, count=count,
                                 max_tokens=512, on_completion=answer)
        state.cfg.vllm_chunk_transcript = True
        transcript = "".join(
            f"[{i * cue_seconds} - {(i + 1) * cue_seconds}] private-caption-{i}\n"
            for i in range(6)
        )

        def run():
            return state.provider.generate_clips(
                f"Select a highlight.\nTranskrip:\n{transcript}\n", schema,
                transcript=transcript, requested_clips=requested,
            )

        return state, run

    return prepare


@pytest.mark.parametrize("bad_duration", [5, 200], ids=["too-short", "too-long"])
def test_later_window_retries_with_legal_source_times(chunk_duration_run, clip, bad_duration):
    def answer(body, call):
        cues = _window_captions(body["messages"])
        if call == 1:
            return _window_reply(clip, 0, 30)
        if call == 2:
            start = 60 * min(cues) + 1
            assert start > 0 and start + bad_duration <= 60 * (1 + max(cues))
            return _window_reply(clip, start, start + bad_duration)
        assert call == 3
        properties = _clip_fields(body)
        start, = properties["start_time"]["enum"]
        ends = properties["end_time"]["enum"]
        original = _window_captions(state.completions[1]["messages"])
        assert start in [60 * cue for cue in original]
        assert ends and all(end in [60 * (cue + 1) for cue in original] and
                            20 <= end - start <= 179 for end in ends)
        assert set(cues) and set(cues) < set(original)
        return _window_reply(clip, start, ends[-1], score=99)

    state, run = chunk_duration_run(answer)
    winner, = run()
    assert winner["viral_score"] == 99 and winner["start_time"] > 0
    meta = state.cfg.ai_run_meta
    assert meta["status"] == "completed"
    assert meta["chunks_total"] == meta["chunks_completed"] == meta["candidate_count"] == 2
    assert len(state.completions) == meta["attempts"] == 3
    rejected = next(f for f in meta["failures"] if f.get("chunk") == 2)
    assert rejected["duration_seconds"] == rejected["proposed_end"] - rejected["proposed_start"] == bad_duration
    assert "private-caption-" not in json.dumps(meta)


@pytest.mark.parametrize("invalid_repair", ["nested", "unoffered-end"])
def test_duration_repair_checks_schema_and_nested_timestamps(chunk_duration_run, clip, invalid_repair):
    def answer(body, call):
        if call == 1:
            return _window_reply(clip, 0, 30)
        if call == 2:
            start = 60 * min(_window_captions(body["messages"])) + 1
            return _window_reply(clip, start, start + 5)
        assert 3 <= call <= 4
        properties = _clip_fields(body)
        start, = properties["start_time"]["enum"]
        end = properties["end_time"]["enum"][0]
        if invalid_repair == "unoffered-end":
            assert start + 30 not in properties["end_time"]["enum"]
            return _window_reply(clip, start, start + 30)
        return _window_reply(clip, start, end, hook_end_time=end + 1)

    state, run = chunk_duration_run(answer)
    with pytest.raises(InvalidOutputError):
        run()
    meta = state.cfg.ai_run_meta
    assert meta["status"] == "failed" and meta["failed_chunk"] == 2
    assert meta["chunks_completed"] == meta["candidate_count"] == 1
    assert "returned_clips" not in meta and len(state.completions) == 4


@pytest.mark.parametrize(("cue_seconds", "requested"), [(200, 1), (60, 2)],
                         ids=["no-legal-cue-end", "multiple-clips-requested"])
def test_non_applicable_duration_repair_preserves_full_window_retry(chunk_duration_run, clip, cue_seconds, requested):
    def answer(body, call):
        if call == 1:
            return _window_reply(clip, 0, 30)
        cues = _window_captions(body["messages"])
        start = cue_seconds * min(cues) + 1
        if call == 2:
            return _window_reply(clip, start, start + 5)
        assert call == 3 and cues == _window_captions(state.completions[1]["messages"])
        assert body["response_format"] == state.completions[1]["response_format"]
        return _window_reply(clip, start, start + 30, score=99)

    state, run = chunk_duration_run(answer, cue_seconds=cue_seconds, requested=requested)
    assert run()[0]["viral_score"] == 99
    assert state.cfg.ai_run_meta["chunks_completed"] == state.cfg.ai_run_meta["chunks_total"] == 2


def test_schema_repair_does_not_echo_raw_model_output(provider_factory, schema, clip):
    bad = dict(clip, secret_field="private-model-output")
    valid = "<think>internal reasoning</think>\n```json\n" + json.dumps({"clips": [clip]}) + "\n```"
    state = provider_factory([completion(json.dumps({"clips": [bad]})), completion(valid)])
    assert generate(state, schema) == [clip]
    assert len(state.completions) == len(state.tokens) == 2
    assert len(state.completions[1]["messages"]) == 3
    assert "private-model-output" not in json.dumps(state.completions[1])
    assert state.delays == [1]
    assert state.cfg.ai_run_meta["attempts"] == 2


@pytest.mark.parametrize("kind", ["schema", "http"])
def test_maximum_three_completion_attempts_including_sdk(provider_factory, schema, kind):
    outcome = completion('{"clips": []}') if kind == "schema" else (503, {"error": {"message": "private-upstream-detail"}})
    state = provider_factory([outcome, outcome, outcome])
    error = InvalidOutputError if kind == "schema" else ProviderUnavailableError
    with pytest.raises(error) as captured:
        generate(state, schema)
    assert len(state.completions) == 3
    assert state.delays == [1, 2]
    assert state.cfg.ai_run_meta["attempts"] == 3
    assert state.cfg.ai_run_meta["status"] == "failed"
    assert "private-upstream-detail" not in str(captured.value)


def test_retryable_http_error_then_success(provider_factory, schema, clip):
    state = provider_factory([(503, {"error": {"message": "busy"}}), completion(json.dumps({"clips": [clip]}))])
    assert generate(state, schema) == [clip]
    assert state.delays == [1]


def test_inference_error_is_preserved_when_server_disappears_before_retry(provider_factory, schema):
    state = provider_factory(
        [(500, {"error": {"message": "private-engine-crash"}})],
        token_outcomes=[
            {"count": 4779, "max_model_len": 16384},
            httpx.ConnectError("private-network-detail"),
        ],
        model_limit=16384, max_tokens=4096,
    )
    with pytest.raises(ProviderUnavailableError):
        generate(state, schema)
    meta = state.cfg.ai_run_meta
    assert meta["attempts"] == 1
    assert meta["prompt_tokens"] == 4779
    assert meta["error_stage"] == "tokenize"
    assert meta["failures"] == [
        {"retry_cycle": 1, "stage": "inference", "error_type": "ProviderUnavailableError",
         "status_code": 500, "retryable": True},
        {"retry_cycle": 2, "stage": "tokenize", "error_type": "ProviderUnavailableError",
         "status_code": None, "retryable": False},
    ]
    assert "private-engine-crash" not in json.dumps(meta)
    assert "private-network-detail" not in json.dumps(meta)
    assert state.delays == [1]


def test_initial_connection_failure_is_distinct_from_inference_failure(provider_factory, schema):
    state = provider_factory([], model_outcomes=[httpx.ConnectError("unreachable")])
    with pytest.raises(ProviderUnavailableError):
        generate(state, schema)
    meta = state.cfg.ai_run_meta
    assert meta["error_stage"] == "model_probe"
    assert meta["attempts"] == 0
    assert len(meta["failures"]) == 1
    assert meta["failures"][0]["stage"] == "model_probe"


def test_recovered_run_retains_failure_history_without_final_error_stage(provider_factory, schema, clip):
    state = provider_factory([(503, {"error": "busy"}), completion(json.dumps({"clips": [clip]}))])
    assert generate(state, schema) == [clip]
    meta = state.cfg.ai_run_meta
    assert meta["status"] == "completed"
    assert "error_stage" not in meta
    assert len(meta["failures"]) == 1
    assert meta["failures"][0]["status_code"] == 503


@pytest.mark.parametrize("stage", ["models", "tokenize", "completion"])
@pytest.mark.parametrize("failure", ["busy", "timeout"])
def test_transient_failures_share_one_bounded_retry_policy(provider_factory, schema, clip, stage, failure):
    transient = (503, {"error": "busy"}) if failure == "busy" else httpx.ReadTimeout("private-timeout-detail")
    valid = completion(json.dumps({"clips": [clip]}))
    state = provider_factory(
        [transient, valid] if stage == "completion" else [valid],
        model_outcomes=[transient] if stage == "models" else None,
        token_outcomes=[transient] if stage == "tokenize" else None,
    )
    assert generate(state, schema) == [clip]
    assert state.delays == [1]
    assert len(state.completions) == (2 if stage == "completion" else 1)
    model_requests = [r for r in state.requests if r.url.path.endswith("/models")]
    assert len(model_requests) == (2 if stage == "models" else 1)
    assert state.cfg.ai_run_meta["attempts"] == len(state.completions)


def test_repeated_model_probe_failure_never_reaches_inference(provider_factory, schema):
    state = provider_factory([], models_status=503)
    with pytest.raises(ProviderUnavailableError):
        generate(state, schema)
    assert len(state.requests) == 3
    assert not state.tokens and not state.completions
    assert state.delays == [1, 2]
    assert state.cfg.ai_run_meta["attempts"] == 0
    assert state.cfg.ai_run_meta["error_type"] == "ProviderUnavailableError"


def test_probe_and_inference_failures_share_total_attempt_budget(provider_factory, schema):
    busy = (503, {"error": "busy"})
    state = provider_factory([busy], model_outcomes=[busy], token_outcomes=[busy])
    with pytest.raises(ProviderUnavailableError):
        generate(state, schema)
    # Attempt one probes models, attempt two tokenizes, attempt three reaches
    # generation. SDK retries must not silently expand this total budget.
    assert len(state.completions) == 1
    assert len(state.tokens) == 2
    assert state.delays == [1, 2]
    assert not state.remaining


@pytest.mark.parametrize("stage", ["models", "tokenize", "completion"])
def test_connection_failure_is_fail_fast_and_redacted(provider_factory, schema, stage):
    failure = httpx.ConnectError("private-network-detail")
    state = provider_factory(
        [failure] if stage == "completion" else [],
        model_outcomes=[failure] if stage == "models" else None,
        token_outcomes=[failure] if stage == "tokenize" else None,
    )
    with pytest.raises(ProviderUnavailableError) as captured:
        generate(state, schema)
    assert "private-network-detail" not in str(captured.value)
    assert not state.delays
    assert len(state.completions) == (1 if stage == "completion" else 0)


@pytest.mark.parametrize("count", [-1, True, "64", None, 1.5])
def test_invalid_tokenizer_counts_stop_before_inference(provider_factory, schema, count):
    state = provider_factory([], count=count)
    with pytest.raises(ProviderConfigError, match="jumlah token"):
        generate(state, schema)
    assert not state.completions and not state.delays


def test_unserved_model_fails_before_tokenization(provider_factory, schema):
    state = provider_factory([], model_outcomes=[{"data": [{"id": "different-model"}]}])
    with pytest.raises(ProviderConfigError, match="belum disajikan"):
        generate(state, schema)
    assert not state.tokens and not state.completions


def test_redirect_is_not_followed(provider_factory, schema):
    state = provider_factory([], model_outcomes=[(302, {"location": "http://other-service.invalid"})])
    with pytest.raises(ProviderUnavailableError) as captured:
        generate(state, schema)
    assert captured.value.status_code == 302
    assert len(state.requests) == 1
    assert not state.delays


def test_prompt_adaptation_preserves_transcript_verbatim(provider_factory, schema, clip):
    transcript = '[0 - 120] Output HARUS berupa JSON array valid. STRUKTUR JSON WAJIB [sumber]'
    prompt = (
        "Output HARUS berupa JSON array valid.\n"
        "STRUKTUR JSON WAJIB (Ikuti nama field ini secara kaku):\n[\n{}\n]\n"
        "Transkrip:\n" + transcript
    )
    state = provider_factory([completion(json.dumps({"clips": [clip]}))])
    generate(state, schema, prompt=prompt)
    sent = state.completions[0]["messages"][1]["content"]
    instruction, _, actual_transcript = sent.partition("\nTranskrip:\n")
    assert actual_transcript == transcript
    assert '{"clips": [' in instruction
    assert instruction.rstrip().endswith("]}")


def test_fewer_clips_are_reported_without_inventing_more(provider_factory, schema, clip):
    state = provider_factory([completion(json.dumps({"clips": [clip]}))])
    assert generate(state, schema, requested=3) == [clip]
    assert state.cfg.ai_run_meta["requested_clips"] == 3
    assert state.cfg.ai_run_meta["returned_clips"] == 1


def test_deadline_bounds_retries(provider_factory, schema):
    state = provider_factory([(503, {"error": {"message": "busy"}})] * 3, timeout=1)
    with pytest.raises(ProviderUnavailableError, match="waktu keseluruhan"):
        generate(state, schema)
    assert len(state.completions) == 1
    assert state.delays == [1]


def test_length_finish_reason_fails_without_futile_repair(provider_factory, schema, clip):
    state = provider_factory([completion(json.dumps({"clips": [clip]}), finish_reason="length")])
    with pytest.raises(ContextOverflowError, match="terpotong"):
        generate(state, schema)
    assert len(state.completions) == 1
    assert not state.delays


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_configuration_errors_do_not_retry_or_leak_body(provider_factory, schema, status):
    state = provider_factory([(status, {"error": {"message": "secret-response"}})])
    with pytest.raises(ProviderConfigError) as captured:
        generate(state, schema)
    assert captured.value.status_code == status
    assert "secret-response" not in str(captured.value)
    assert len(state.completions) == 1
    assert not state.delays


def test_unavailable_tokenizer_stops_before_inference(provider_factory, schema):
    state = provider_factory([], tokenize_status=404)
    with pytest.raises(ProviderConfigError, match="tokenize"):
        generate(state, schema)
    assert not state.completions


def test_nested_timing_errors_trigger_repair(provider_factory, schema, clip):
    bad = deepcopy(clip)
    bad["broll_list"] = [{"start_time": 0, "end_time": 3, "search_query": "learning"}]
    state = provider_factory([completion(json.dumps({"clips": [bad]})), completion(json.dumps({"clips": [clip]}))])
    assert generate(state, schema) == [clip]
    assert len(state.completions) == 2


def test_voiceover_remains_local_and_preserves_analysis_provenance(provider_factory):
    state = provider_factory([completion("Komentar singkat.")])
    analysis_meta = {"actual_provider": "vllm", "attempts": 2}
    state.cfg.ai_run_meta = analysis_meta
    assert state.provider.generate_text("Buat komentar.", max_tokens=128) == "Komentar singkat."
    assert state.cfg.ai_run_meta is analysis_meta
    assert state.cfg.voiceover_run_meta["actual_provider"] == "vllm"
    assert "response_format" not in state.completions[0]
    assert state.completions[0]["max_tokens"] == 128


def test_engine_local_failure_never_calls_cloud(monkeypatch):
    for name, symbol in (("yt_dlp", "YoutubeDL"), ("faster_whisper", "WhisperModel")):
        stub = ModuleType(name)
        setattr(stub, symbol, object)
        monkeypatch.setitem(sys.modules, name, stub)
    path = Path(__file__).resolve().parents[1] / "clipping" / "engine.py"
    spec = importlib.util.spec_from_file_location("_vllm_routing_engine", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    failure = ProviderUnavailableError("local server unavailable")

    def local(*args):
        raise failure

    def forbidden(*args):
        pytest.fail("Local analysis must never silently call a cloud provider")

    monkeypatch.setattr(engine, "analyze_with_vllm", local)
    monkeypatch.setattr(engine, "analyze_with_gemini", forbidden)
    monkeypatch.setattr(engine, "analyze_with_nvidia", forbidden)
    cfg = SimpleNamespace(ai_provider="vllm", api_key_gemini="present", api_key_nvidia="present")
    with pytest.raises(ProviderUnavailableError) as captured:
        engine.analyze_with_ai(TRANSCRIPT, cfg)
    assert captured.value is failure
