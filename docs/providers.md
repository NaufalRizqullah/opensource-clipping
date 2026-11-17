# AI providers

The application supports `gemini` (default), `nvidia`, and `vllm`. vLLM is a separate, self-hosted model server; installing the application installs its HTTP client, not the model server. Ollama is deferred.

For the experimental single-GPU Kaggle setup, start with [Gemma 4 Q4 on Kaggle](KAGGLE_VLLM.md) and its [notebook](../notebooks/kaggle-vllm-gemma4.ipynb). The preset and regression tests have been authored but **have not been run in Kaggle**. No tests, dependency installation, or inference were run on the development PC, as requested.

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `VLLM_BASE_URL` | `http://127.0.0.1:8001` | Server root; an optional trailing `/v1` is accepted |
| `VLLM_MODEL` | `google/gemma-4-E2B-it-qat-w4a16-ct` | Exact served model ID or alias from `/v1/models` |
| `VLLM_API_KEY` | empty | Optional server authentication; unrelated to OpenAI cloud keys |
| `VLLM_MAX_TOKENS` | `4096` | Maximum generated tokens per analysis request |
| `VLLM_CONTEXT_LIMIT` | unset | Optional conservative context ceiling, in tokens |
| `VLLM_TIMEOUT` | `900` | Overall analysis/retry time budget in seconds |

CLI overrides: `--vllm-model`, `--local-base-url`, `--local-max-tokens`, `--local-context-limit`, and `--local-timeout`. The context flag does **not** increase the server's `--max-model-len`.

Once a compatible server is running, use the application environment:

```bash
python main.py --url "VIDEO_URL" --ai-provider vllm --clips 1 \
  --whisper-device cpu --whisper-compute-type int8
```

This starts with one clip and keeps Whisper off the GPU occupied by vLLM. CPU transcription can be slow. A separate inference GPU/server allows other Whisper settings. Neither system RAM nor two separate GPUs should be treated as one pooled VRAM allocation.

Studio offers vLLM, a model/alias field and an output-token budget in New Job. Settings displays the operator's model/token defaults and whether authentication is configured. Endpoint, key, context ceiling and timeout are supplied by the **backend operator's environment**, not by job payloads or public Settings. Restart the backend after changing its environment. Cached jobs can reuse results without a running model server. The existing `/api/ready` remains a renderer/dependency probe and does not contact LLM providers.

Compose forwards the same variables to the backend and defaults its endpoint to `http://host.docker.internal:8001` using a host-gateway mapping. The server must listen on an interface reachable from the container; a loopback-only Kaggle helper is for same-host notebook processes. Set an explicit private endpoint for remote serving. Compose does not install/start a vLLM server.

## Request and failure behavior

- Gemini keeps its Developer API `v1beta` route, schema dialect, retry policy and fallback model. NVIDIA keeps its NIM request fields, SDK retry defaults and existing Gemini fallback. The OpenAI SDK is now a main dependency, fixing the previously missing NIM import.
- vLLM never falls back to a cloud provider. Local CLI voiceover script generation follows vLLM; cloud voiceover remains Gemini. Studio does not expose voiceover. `edge-tts`, Pexels and downloads still use the network.
- The client checks `/v1/models`, then calls `/tokenize` with the same chat messages and thinking setting as generation. Prompt tokens **plus output budget** must fit the smallest available server/client context limit. It fails clearly if no reliable limit is available; it does not silently truncate the transcript.
- Generation uses `response_format: json_schema` with a `{"clips": [...]}` object. Complete thinking/fence wrappers can be removed before strict schema validation. The public analysis result remains a list.
- Timestamp validation covers clip, hook, montage, keep-segment and B-roll bounds. Clips must be 20–179 seconds, within the transcript, and non-overlapping. Invalid timestamps are rejected rather than silently changed. Fewer valid clips are allowed with an explicit count message; empty output fails.
- There are at most three attempts across transient failures and invalid-output repair. Authentication, missing model, connection refusal, context overflow and `finish_reason=length` fail without repeated inference. SDK retries are disabled for vLLM so the application budget is not multiplied. Probe/tokenization requests have shorter timeouts within the overall budget.
- Clients do not follow redirects or inherit HTTP proxy environment settings. Use a directly reachable private endpoint or an operator-managed reverse proxy that exposes both `/v1` and `/tokenize`. Basic URL checks reject embedded credentials, query strings, literal metadata/link-local targets and unsupported schemes; these checks are not a DNS allowlist. Only trusted operators choose endpoints.

The raw result cache remains `outputs/<job>/gemini_response.json` for compatibility. `ai_run_meta.json` records local model, attempts, token usage, duration and cache reuse without transcript text. A content hash ties successful provenance to the saved cache; a failed rerun cannot claim an older result. Historical cloud runs have unknown actual model/provider where no instrumentation exists; the file does not guess the outcome of internal fallbacks. Bypass AI requires an existing cache and fails if it is missing.

## Kaggle validation

Use the app environment, not the vLLM server environment, for tests:

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests/test_provider_cloud_contracts.py tests/test_provider_metadata.py \
  tests/test_gemini_routing.py tests/test_worker.py tests/test_studio_logging.py \
  tests/test_vllm_provider.py tests/test_vllm_validation.py tests/test_vllm_cli.py \
  tests/test_vllm_web.py tests/test_kaggle_vllm.py tests/test_local_eval.py
python scripts/eval_local_llm.py --transcript tests/fixtures/eval_transcript_short.txt \
  --clips 1 --output /kaggle/working/vllm-eval-short.json
RUN_LOCAL_LLM_TESTS=1 python -m pytest -m local_llm tests/test_vllm_live.py
```

The regular tests use fake modules or mock HTTP transports and no inference server. The live test skips unless explicitly opted in; after opt-in, an unreachable server fails the test. The evaluation writes generated clips, timing/schema success, metadata-warning count and usage to a JSON report. Review Indonesian/English quality and narrative coherence yourself; schema compliance is not a quality benchmark. Long-transcript evaluation, peak-memory measurements, chunking/staged generation and CPU serving presets remain future work.

The Kaggle helper pins the official QAT W4A16 compressed-tensors checkpoint and its revision, plus vLLM 0.32.0. Q4 weights do not make the KV cache or all runtime allocations 4-bit. Source compatibility and quantized weight size do not prove the full model/pipeline fits 16 GB. See the official [Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html), [structured-output API](https://docs.vllm.ai/en/stable/features/structured_outputs/), and [CPU installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/cpu/). CPU servers require a matching build and separately verified model/quantization settings; the Kaggle GPU helper deliberately rejects CPU-only hosts and unsupported GPUs.
