"""OpenAI-compatible self-hosted inference with bounded, local-only retries.

No vLLM/torch import belongs here: the model server has its own environment.
"""

from copy import deepcopy
import json
import time

from .base import (
    ContextOverflowError, InvalidOutputError, ProviderConfigError,
    ProviderError, ProviderUnavailableError,
)
from .config import normalize_base_url, vllm_config_values
from .parsing import clean_text, parse_clips
from .validation import ClipDurationError, transcript_end, validate_clips


MAX_ATTEMPTS = 3
_RETRYABLE = {408, 429, 500, 502, 503, 504}


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _local_prompt(prompt):
    # Only change the template, never replace strings inside user transcripts.
    instruction, separator, transcript = prompt.partition("\nTranskrip:\n")
    instruction = instruction.replace(
        "Output HARUS berupa JSON array valid.",
        'Output HARUS berupa objek JSON valid dengan satu field "clips" berupa array.',
    )
    instruction = instruction.replace(
        "STRUKTUR JSON WAJIB (Ikuti nama field ini secara kaku):\n[",
        'STRUKTUR JSON WAJIB (Ikuti nama field ini secara kaku):\n{"clips": [',
        1,
    )
    if 'kaku):\n{"clips": [' in instruction:
        # The fixed template finishes with its example array before Transkrip.
        position = instruction.rfind("]")
        instruction = instruction[:position + 1] + "}" + instruction[position + 1:]
    return instruction + separator + transcript


class VllmProvider:
    name = "vllm"

    def __init__(self, cfg, *, http_client=None, sleep=None, clock=None):
        import httpx
        from openai import OpenAI

        self.cfg = cfg
        self.options = vllm_config_values(vars(cfg))
        self.root, self.api_base = normalize_base_url(self.options["vllm_base_url"])
        self.model = self.options["vllm_model"]
        self.max_tokens = self.options["local_max_tokens"]
        self.timeout = self.options["vllm_timeout"]
        self.context_limit = self.options["vllm_context_limit"]
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._owns_http = http_client is None
        self._http = http_client if http_client is not None else httpx.Client(
            follow_redirects=False, trust_env=False,
        )
        self._key = self.options["api_key_vllm"] or "local-vllm"
        self._client = OpenAI(
            base_url=self.api_base, api_key=self._key, max_retries=0,
            http_client=self._http, timeout=self.timeout,
        )
        self._model_limit = None
        self._deadline = None
        self.run_meta = {}

    @classmethod
    def from_cfg(cls, cfg):
        return cls(cfg)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._owns_http:
            self._client.close()

    def _remaining(self):
        remaining = self.timeout if self._deadline is None else self._deadline - self._clock()
        if remaining <= 0:
            raise ProviderUnavailableError("Batas waktu keseluruhan vLLM terlampaui.")
        return remaining

    def _http_timeout(self, *, probe=False):
        import httpx
        remaining = min(self._remaining(), 10) if probe else self._remaining()
        return httpx.Timeout(remaining, connect=min(5, remaining), pool=min(5, remaining))

    @staticmethod
    def _status_error(status):
        if status in {401, 403}:
            return ProviderConfigError("Autentikasi vLLM ditolak. Periksa VLLM_API_KEY pada client dan server.", status_code=status)
        if status == 404:
            return ProviderConfigError("Endpoint/model vLLM tidak ditemukan. Periksa base URL, model served, dan akses /tokenize.", status_code=status)
        if status == 400:
            return ProviderConfigError("vLLM menolak request. Periksa limit konteks, chat template, schema, dan versi server.", status_code=status)
        return ProviderUnavailableError(
            f"Request vLLM gagal (HTTP {status}). Periksa status server.",
            status_code=status, retryable=status in _RETRYABLE,
        )

    def _json_request(self, method, url, *, payload=None):
        import httpx
        try:
            response = self._http.request(
                method, url, json=payload,
                headers={"Authorization": "Bearer " + self._key},
                timeout=self._http_timeout(probe=True), follow_redirects=False,
            )
        except httpx.TimeoutException:
            raise ProviderUnavailableError("Probe/tokenisasi vLLM timeout.", retryable=True) from None
        except httpx.TransportError:
            raise ProviderUnavailableError("Server vLLM tidak terjangkau. Jalankan server dan periksa VLLM_BASE_URL.") from None
        if response.status_code != 200:
            raise self._status_error(response.status_code)
        try:
            data = response.json()
        except ValueError:
            raise ProviderConfigError("Endpoint vLLM mengembalikan response non-JSON.") from None
        if not isinstance(data, dict):
            raise ProviderConfigError("Format response endpoint vLLM tidak didukung.")
        return data

    def validate(self):
        """Probe only this configured server and ensure the model/alias exists."""
        data = self._json_request("GET", self.api_base + "/models")
        records = data.get("data")
        if not isinstance(records, list):
            raise ProviderConfigError("Response /v1/models tidak memuat daftar model.")
        record = next((m for m in records if isinstance(m, dict) and m.get("id") == self.model), None)
        if record is None:
            raise ProviderConfigError("VLLM_MODEL belum disajikan server. Cocokkan nama dengan /v1/models atau --served-model-name.")
        limit = record.get("max_model_len")
        self._model_limit = limit if _positive_int(limit) else None

    def _preflight_tokens(self, messages, max_tokens):
        data = self._json_request("POST", self.root + "/tokenize", payload={
            "model": self.model, "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        count = data.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ProviderConfigError("/tokenize tidak menyediakan jumlah token yang valid.")
        limits = [v for v in (data.get("max_model_len"), self._model_limit, self.context_limit) if _positive_int(v)]
        if not limits:
            raise ProviderConfigError("Limit konteks server tidak tersedia; set VLLM_CONTEXT_LIMIT sesuai konfigurasi server.")
        limit = min(limits)
        self.run_meta.update(prompt_tokens=count, context_limit=limit, max_tokens=max_tokens)
        if count + max_tokens > limit:
            remaining = limit - count
            if remaining > 0:
                advice = (f"Prompt ini menyisakan {remaining} token untuk jawaban. "
                          "Kurangi batas output jika sisa itu cukup, atau gunakan input yang lebih pendek.")
            else:
                advice = "Prompt sudah memenuhi batas konteks. Gunakan input yang lebih pendek, misalnya satu bagian video."
            raise ContextOverflowError(
                f"Konteks vLLM tidak cukup: prompt {count} + output {max_tokens} > {limit} token. "
                f"{advice} Untuk permintaan lengkap ini, model dan server perlu mendukung "
                f"konteks minimal {count + max_tokens} token."
            )

    def _completion(self, messages, *, max_tokens, temperature, top_p, schema=None):
        import openai
        kwargs = dict(
            model=self.model, messages=messages, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            timeout=self._http_timeout(),
        )
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema", "json_schema": {"name": "clips", "strict": True, "schema": schema},
            }
        try:
            return self._client.chat.completions.create(**kwargs)
        except openai.APITimeoutError:
            raise ProviderUnavailableError("Inference vLLM timeout.", retryable=True) from None
        except openai.APIConnectionError:
            raise ProviderUnavailableError("Koneksi vLLM gagal. Periksa proses server dan base URL.") from None
        except openai.APIStatusError as exc:
            raise self._status_error(exc.status_code) from None
        except openai.APIError:
            raise ProviderUnavailableError("Response vLLM tidak dapat dibaca SDK. Periksa kompatibilitas server.") from None

    def _begin_run(self, started):
        self._deadline = started + self.timeout
        self.run_meta = dict(
            actual_provider="vllm", model=self.model, attempts=0,
            fallback_used=False, status="failed", usage=[], failures=[],
        )

    def _record_failure(self, exc, cycle, stage):
        # Only retain controlled diagnostics. Provider bodies can contain user data.
        failure = {
            "retry_cycle": cycle, "stage": stage, "error_type": type(exc).__name__,
            "status_code": exc.status_code, "retryable": exc.retryable,
        }
        if "active_chunk" in self.run_meta:
            failure["chunk"] = self.run_meta["active_chunk"]
        if isinstance(exc, ClipDurationError):
            failure.update(proposed_start=exc.start, proposed_end=exc.end,
                           duration_seconds=round(exc.end - exc.start, 6))
        self.run_meta["failures"].append(failure)

    def _generate(self, messages, *, max_tokens, temperature, top_p, schema=None, parse=None,
                  _shared_run=False, _validated=False, repair_messages=None, duration_repair=None):
        started = self._clock()
        if not _shared_run:
            self._begin_run(started)
        stage = "model_probe"

        try:
            current = list(messages)
            current_schema = schema
            validated = _validated
            for attempt in range(MAX_ATTEMPTS):
                try:
                    if not validated:
                        stage = "model_probe"
                        self.validate()
                        validated = True
                    stage = "tokenize"
                    try:
                        self._preflight_tokens(current, max_tokens)
                    except ContextOverflowError:
                        if current_schema is schema:
                            raise
                        # The generic window repair was measured during planning.
                        current, current_schema = repair_messages, schema
                        self._preflight_tokens(current, max_tokens)
                    stage = "inference"
                    self.run_meta["attempts"] += 1
                    response = self._completion(
                        current, max_tokens=max_tokens, temperature=temperature,
                        top_p=top_p, schema=current_schema,
                    )
                    stage = "output_validation"
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        self.run_meta["usage"].append({
                            k: getattr(usage, k, None)
                            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                        })
                    if not response.choices:
                        raise InvalidOutputError("vLLM mengembalikan choices kosong.")
                    choice = response.choices[0]
                    if choice.finish_reason == "length":
                        # Repeating the same bounded request cannot fix truncation.
                        raise ContextOverflowError("Output vLLM terpotong pada batas token. Kurangi jumlah klip atau naikkan budget output dan konteks yang tersedia.")
                    if choice.finish_reason != "stop":
                        raise InvalidOutputError("vLLM tidak menyelesaikan jawaban teks normal.")
                    result = parse(choice.message.content, current_schema) if parse else clean_text(choice.message.content)
                    self._remaining()
                    if not _shared_run:
                        self.run_meta["status"] = "completed"
                    return result
                except InvalidOutputError as exc:
                    self._record_failure(exc, attempt + 1, stage)
                    if attempt == MAX_ATTEMPTS - 1:
                        raise
                    # Trusted diagnostic only; never echo model text into repair/logs.
                    if current_schema is schema:
                        current = repair_messages or list(messages) + [{"role": "user", "content": (
                            "Buat ulang jawaban secara lengkap. Perbaiki masalah berikut: " + str(exc)
                            + " Ikuti schema dan batas timestamp sumber; jangan menambahkan penjelasan."
                        )}]
                        if duration_repair and isinstance(exc, ClipDurationError):
                            focused = duration_repair(exc)
                            if focused is not None:
                                current, current_schema = focused
                except ProviderUnavailableError as exc:
                    self._record_failure(exc, attempt + 1, stage)
                    if not exc.retryable or attempt == MAX_ATTEMPTS - 1:
                        raise
                except ProviderError as exc:
                    self._record_failure(exc, attempt + 1, stage)
                    raise
                stage = "retry_wait"
                delay = min(2 ** attempt, self._remaining())
                self._sleep(delay)
            raise ProviderUnavailableError("Percobaan vLLM habis.")
        except ProviderError as exc:
            self.run_meta["error_type"] = type(exc).__name__
            self.run_meta["error_stage"] = stage
            raise
        finally:
            if not _shared_run:
                self.run_meta["duration_seconds"] = round(self._clock() - started, 3)
                self._deadline = None

    def _messages_fit(self, messages, max_tokens, stage, *, probe=False):
        for attempt in range(MAX_ATTEMPTS):
            try:
                if probe:
                    self.validate()
                    probe = False
                self._preflight_tokens(messages, max_tokens)
                return True
            except ContextOverflowError:
                return False
            except ProviderUnavailableError as exc:
                self._record_failure(exc, attempt + 1, stage)
                if not exc.retryable or attempt == MAX_ATTEMPTS - 1:
                    raise
                self._sleep(min(2 ** attempt, self._remaining()))
            except ProviderError as exc:
                self._record_failure(exc, attempt + 1, stage)
                raise

    def _generate_shortlist(self, full_messages, prompt, wrapped, transcript, source_end, min_duration, max_duration):
        from .chunking import plan_windows
        from .shortlist import choices_for, metadata_fields, metadata_messages, one_clip_schema, selection_messages

        stage = "shortlist_planning"
        meta = self.run_meta
        meta.update(mode="chunked", strategy="shortlist", windows=[], chunks_total=0, chunks_completed=0,
                    candidate_count=0)
        try:
            instructions, separator, tail = prompt.partition("\nTranskrip:\n")
            if not separator or not tail.startswith(transcript) or tail[len(transcript):].strip():
                raise ProviderConfigError("Shortlist memerlukan template dengan transkrip lengkap pada bagian Transkrip.")
            self._messages_fit(full_messages, self.max_tokens, stage, probe=True)
            meta["full_prompt_tokens"] = meta["prompt_tokens"]
            choice_budget = min(128, self.max_tokens)

            def fits(window):
                options = choices_for(window, min_duration, max_duration)
                if not options:
                    # The planner can probe a short cue before extending it.
                    return window.end - window.start < min_duration
                return self._messages_fit(selection_messages(window, options), choice_budget, stage)

            windows = plan_windows(transcript, fits, min_duration=min_duration, overlap_seconds=max_duration)
            meta["chunks_total"] = len(windows)
            stage = "shortlist_selection"
            nominees = []
            for index, window in enumerate(windows, 1):
                self._remaining()
                meta["active_chunk"] = index
                options = choices_for(window, min_duration, max_duration)
                if not options:
                    raise ProviderConfigError("Bagian transkrip tidak dapat dibagi menjadi klip dengan durasi yang valid.")
                messages = selection_messages(window, options)
                schema = one_clip_schema({
                    "choice": {"type": "integer", "enum": list(range(1, len(options) + 1))},
                    "viral_score": {"type": "integer", "enum": list(range(1, 101))},
                })
                print(f"[vLLM] Bagian {index}/{len(windows)}: {window.start:.1f}–{window.end:.1f} detik, "
                      f"{len(options)} pilihan.", flush=True)
                choice, = self._generate(
                    messages, max_tokens=choice_budget, temperature=0.4, top_p=1, schema=schema,
                    parse=parse_clips, _shared_run=True, _validated=True, repair_messages=messages,
                )
                nominee = options[int(choice["choice"]) - 1]
                nominees.append((nominee, choice["viral_score"]))
                meta["windows"].append({"start": window.start, "end": window.end,
                                        "prompt_tokens": meta["prompt_tokens"], "start_time": nominee.start,
                                        "end_time": nominee.end, "viral_score": choice["viral_score"]})
                meta.update(chunks_completed=index, candidate_count=index)
            meta.pop("active_chunk")
            excerpt, score = min(nominees, key=lambda item: (-item[1], item[0].start, item[0].end))
            result = dict(rank=1, viral_score=score, start_time=excerpt.start, end_time=excerpt.end)
            fields = wrapped["properties"]["clips"]["items"]["properties"]

            for editing in (True, False):
                stage = "edit_metadata" if editing else "publishing_metadata"
                group = metadata_fields(fields, editing=editing)
                messages = metadata_messages(instructions, excerpt, group, editing=editing, cfg=self.cfg)
                print(f"[vLLM] {stage}: {excerpt.start:.1f}–{excerpt.end:.1f} detik.", flush=True)

                def parse_group(content, response_schema):
                    partial, = parse_clips(content, response_schema)
                    validate_clips([result | partial], source_end=source_end, requested_clips=1,
                                   min_duration=min_duration, max_duration=max_duration)
                    return partial

                result.update(self._generate(
                    messages, max_tokens=self.max_tokens, temperature=0.5, top_p=1,
                    schema=one_clip_schema(group), parse=parse_group, _shared_run=True, _validated=True,
                ))
            # Recheck the assembled object against the exact public contract.
            clips = parse_clips(json.dumps({"clips": [result]}), wrapped)
            self._remaining()
            return clips
        except ProviderError as exc:
            meta.update(error_type=type(exc).__name__, error_stage=stage)
            if "active_chunk" in meta:
                meta["failed_chunk"] = meta.pop("active_chunk")
            raise

    def _generate_chunked(self, prompt, wrapped, *, transcript, requested_clips,
                          min_duration, max_duration, source_end, system_message):
        from .chunking import duration_repair_options, plan_windows, select_candidates

        self.run_meta.update(
            mode="chunked", full_prompt_tokens=self.run_meta.get("prompt_tokens"),
            chunks_total=0, chunks_completed=0, candidate_count=0, windows=[],
        )
        # The initial overflow is retained in failures, but is no longer the
        # terminal error once a complete, bounded plan is being attempted.
        self.run_meta.pop("error_type", None)
        self.run_meta.pop("error_stage", None)
        stage = "chunk_planning"
        try:
            instruction, separator, tail = prompt.partition("\nTranskrip:\n")
            if (not separator or not tail.startswith(transcript)
                    or tail[len(transcript):].strip()):
                raise ProviderConfigError(
                    "Chunking memerlukan template prompt dengan transkrip lengkap pada bagian Transkrip."
                )
            suffix = tail[len(transcript):]

            def messages_for(window, *, repair=False, endings=None):
                # Replace only the verified transcript suffix. Source text may
                # itself contain the delimiter or instruction-like strings.
                # The template ends in a worked JSON example with times from a different clip.
                rules = instruction
                if endings:
                    rules = rules.partition("\nSTRUKTUR JSON WAJIB (Ikuti nama field ini secara kaku):")[0]
                chunk_prompt = rules + separator + window.text + suffix
                bounds = (
                    f" Analyze only source interval [{window.start}, {window.end}] seconds."
                    " Every clip and its nested segments must stay inside this interval."
                    f" Each clip must last {min_duration} to {max_duration} seconds (end_time minus start_time)."
                    " Keep the original absolute timestamps; do not restart them at zero."
                )
                if repair:
                    # A fixed, pre-budgeted correction fits even at the context boundary.
                    bounds += (" Regenerate the complete JSON. Verify its required fields, clip durations,"
                               " non-overlap, and every hook/segment bound before answering.")
                if endings:
                    bounds += (
                        f" The previous clip duration was invalid. Use start_time={window.start}. "
                        f"Choose end_time from {endings} to finish the thought. These are final absolute "
                        "timestamps including any padding. Regenerate all metadata for the chosen interval only."
                    )
                return [
                    {"role": "system", "content": system_message + bounds},
                    {"role": "user", "content": _local_prompt(chunk_prompt)},
                ]

            def fits(window):
                return all(self._messages_fit(messages_for(window, repair=repair), self.max_tokens, stage)
                           for repair in (True, False))

            windows = list(plan_windows(
                transcript, fits, min_duration=min_duration, overlap_seconds=max_duration,
            ))
            self._remaining()
            if not windows:
                raise ProviderConfigError("Chunking tidak menghasilkan jendela transkrip yang valid.")
            self.run_meta["chunks_total"] = len(windows)
            candidates = []
            stage = "chunk_inference"
            for index, window in enumerate(windows, 1):
                self.run_meta["active_chunk"] = index
                print(f"[vLLM] Bagian {index}/{len(windows)}: "
                      f"{window.start:.1f}–{window.end:.1f} detik.", flush=True)

                def parse_window(content, response_schema):
                    clips = validate_clips(
                        parse_clips(content, response_schema), source_end=window.end,
                        requested_clips=requested_clips,
                        min_duration=min_duration, max_duration=max_duration,
                    )
                    # validate_clips also checks all nested ranges against each
                    # clip, so checking the outer lower bound is sufficient.
                    if any(clip["start_time"] < window.start for clip in clips):
                        raise InvalidOutputError("Timestamp klip berada sebelum awal jendela transkrip.")
                    return clips

                def repair_duration(error):
                    options = duration_repair_options(window, error.start, error.end, min_duration, max_duration)
                    if options is None:
                        return None
                    excerpt, ends = options
                    repair_schema = deepcopy(wrapped)
                    fields = repair_schema["properties"]["clips"]["items"]["properties"]
                    fields["start_time"]["enum"] = [excerpt.start]
                    fields["end_time"]["enum"] = ends
                    print(f"[vLLM] Bagian {index}: {error} Mencoba ulang dengan pilihan akhir dari subtitle.", flush=True)
                    return messages_for(excerpt, repair=True, endings=ends), repair_schema

                clips = self._generate(
                    messages_for(window), max_tokens=self.max_tokens, temperature=0.5,
                    top_p=1, schema=wrapped, parse=parse_window,
                    _shared_run=True, _validated=True, repair_messages=messages_for(window, repair=True),
                    duration_repair=repair_duration if requested_clips == 1 else None,
                )
                candidates.extend(clips)
                self.run_meta["windows"].append({
                    "start": window.start, "end": window.end,
                    "prompt_tokens": self.run_meta["prompt_tokens"],
                })
                self.run_meta.update(chunks_completed=index, candidate_count=len(candidates))
            self.run_meta.pop("active_chunk", None)
            stage = "candidate_selection"
            result = validate_clips(
                select_candidates(candidates, requested_clips), source_end=source_end,
                requested_clips=requested_clips,
                min_duration=min_duration, max_duration=max_duration,
            )
            self._remaining()
            return result
        except ProviderError as exc:
            self.run_meta["error_type"] = type(exc).__name__
            self.run_meta.setdefault("error_stage", stage)
            if "active_chunk" in self.run_meta:
                self.run_meta["failed_chunk"] = self.run_meta.pop("active_chunk")
            raise

    def generate_clips(self, prompt, schema, *, transcript, requested_clips, min_duration=20, max_duration=179):
        if not _positive_int(requested_clips):
            raise ProviderConfigError("Jumlah klip harus integer positif.")
        shortlist = bool(getattr(self.cfg, "vllm_shortlist", False))
        if shortlist and requested_clips != 1:
            raise ProviderConfigError("Shortlist vLLM memerlukan tepat satu klip.")
        end = transcript_end(transcript)
        if end < min_duration:
            raise ProviderConfigError("Transkrip lebih pendek dari durasi minimum klip.")
        array = deepcopy(schema)
        array.update(minItems=1, maxItems=requested_clips)
        wrapped = {"type": "object", "properties": {"clips": array}, "required": ["clips"], "additionalProperties": False}

        def parse(content, response_schema):
            return validate_clips(
                parse_clips(content, response_schema), source_end=end, requested_clips=requested_clips,
                min_duration=min_duration, max_duration=max_duration,
            )

        system_message = (
            'Return only a JSON object {"clips": [...]} matching the supplied schema. '
            'All timestamps are absolute source-video seconds. Choose non-overlapping clips. '
            'Return fewer clips only if the source cannot support the requested count.'
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": _local_prompt(prompt)},
        ]
        chunking = shortlist or bool(getattr(self.cfg, "vllm_chunk_transcript", False))
        started = self._clock()
        if chunking:
            self._begin_run(started)
            self.run_meta["mode"] = "single_pass"
        try:
            if shortlist:
                result = self._generate_shortlist(messages, prompt, wrapped, transcript, end, min_duration, max_duration)
            else:
                try:
                    options = {"_shared_run": True} if chunking else {}
                    result = self._generate(
                        messages, max_tokens=self.max_tokens, temperature=0.5, top_p=1,
                        schema=wrapped, parse=parse, **options,
                    )
                except ContextOverflowError:
                    if (not chunking or self.run_meta.get("error_stage") != "tokenize"
                            or self.run_meta.get("attempts") != 0):
                        raise
                    result = None
            # Leave the expected initial overflow out of any later error traceback.
            if result is None:
                result = self._generate_chunked(
                    prompt, wrapped, transcript=transcript, requested_clips=requested_clips,
                    min_duration=min_duration, max_duration=max_duration, source_end=end,
                    system_message=system_message,
                )
            if chunking:
                self.run_meta["status"] = "completed"
            self.run_meta.update(requested_clips=requested_clips, returned_clips=len(result))
            if len(result) < requested_clips:
                print(f"[vLLM] Menghasilkan {len(result)} dari {requested_clips} klip yang diminta; tidak ada klip tambahan yang direkayasa.")
            return result
        finally:
            if chunking:
                self.run_meta["duration_seconds"] = round(self._clock() - started, 3)
                self._deadline = None
            self.cfg.ai_run_meta = dict(self.run_meta)

    def generate_text(self, prompt, *, temperature=0.7, top_p=0.9, max_tokens=None):
        budget = self.max_tokens if max_tokens is None else max_tokens
        if not _positive_int(budget):
            raise ProviderConfigError("Budget token voiceover harus integer positif.")
        # Keep voiceover metadata separate from transcript-analysis provenance.
        try:
            return self._generate(
                [{"role": "user", "content": prompt}], max_tokens=budget,
                temperature=temperature, top_p=top_p,
            )
        finally:
            self.cfg.voiceover_run_meta = dict(self.run_meta)
