# Kaggle / Colab Studio server

Use [`notebooks/kaggle-studio-server.ipynb`](../notebooks/kaggle-studio-server.ipynb) for the browser-based Studio. The historical name `Kaggle_Studio_Server` refers to this notebook. The CLI notebooks and `main.py` remain separate entry points.

## Setup and startup

1. Select a Linux x86_64 runtime with Python 3.10–3.12 (prefer 3.12), enable Internet, and select a GPU for CUDA transcription/rendering.
2. Run the clone/setup cells. The repository is cloned into a named directory without deleting the working directory. Existing checkouts and outputs are retained; an existing checkout is not automatically pulled or reset.
3. Setup creates `.cache/kaggle-studio/venv` without access to the notebook's installed packages or reliance on distro `ensurepip`. It checks the target interpreter, bootstraps missing pip into that venv, then installs `requirements-kaggle.txt` with `constraints-kaggle.txt`, runs `pip check`, and installs a checksum-verified Deno runtime for yt-dlp's YouTube JavaScript challenges. Installer errors stop the cell and remain visible.
4. Supply secrets through Kaggle Secrets, Colab userdata, or the environment. `NGROK_AUTHTOKEN` is needed for the tunnel; Google or NVIDIA credentials depend on the selected AI provider. The notebook passes them through the process environment, without writing a `.env` file or printing their values. Existing `.env` files are ignored by this managed server.
5. Run the server cell and keep it running. It launches Uvicorn in a **fresh Python subprocess**, checks `/api/ready`, and only then opens the tunnel and prints **BACKEND SERVER READY** with its Public URL.
6. In [Studio](https://naufalrizqullah.github.io/opensource-clipping/studio/), choose **Connect**, paste the URL, and select **Test & Connect**. Pages refresh after a successful connection. A failed attempt preserves the previously saved URL.

The first import/readiness check can take a few minutes. Stop Server requests graceful shutdown; an active native operation may delay process exit. Stopping the notebook cell also applies the supervisor's shutdown timeout and closes the managed backend and tunnel. A busy port is reported instead of connecting the notebook to an unrelated older process. API jobs are serialized even if `MAX_CONCURRENT_JOBS` requests a higher value: renderer temporary files and some settings are shared within the process. Use one Uvicorn process per checkout.

Tunnel requests use the full upstream address `http://127.0.0.1:8000` (or the selected port). pyngrok 9 defaults to the v3 Endpoints API, which can incorrectly expand a bare `127.0.0.1:8000` into `http://localhost:127.0.0.1:8000`. Studio selects HTTPS with the v3 `url: https://` shorthand, or the legacy `bind_tls` option for v2 configurations/older pyngrok. It accepts only an HTTPS public URL. This preserves the supported pyngrok 7.2–9.x range without forcing a dependency downgrade. The routing distinction is documented in [pyngrok's connect implementation](https://pyngrok.readthedocs.io/en/latest/_modules/pyngrok/ngrok.html); the [ngrok v3 URL reference](https://ngrok.com/docs/gateway/agent/config/v3#url) documents scheme-only URLs.

## Why the environment is isolated

`numpy<2` in a requirements file is insufficient when the notebook already has native libraries loaded or incompatible preinstalled packages. The reported `numpy.dtype size changed` error occurred when importing MediaPipe during rendering, after AI analysis had succeeded.

The Kaggle constraints keep NumPy 1.26.4, MediaPipe 0.10.21, OpenCV 4.11, JAX 0.4.35, and the Pyannote 3.x API on compatible dependency families. Torch 2.5.1, Torchaudio 2.5.1, and Torchvision 0.20.1 are kept together. General CLI requirements and CLI arguments are unchanged. This is a constrained environment, not a complete transitive lockfile; actual runtime import checks are still required.

The readiness probe checks NumPy, the MediaPipe/Studio renderer, Whisper/engine imports, and FFmpeg in a separate process. It does not download models, call paid AI APIs, validate credentials, or prove that GPU inference, gated Hugging Face models, or a particular video will succeed.

The managed process uses `MPLBACKEND=Agg`. MediaPipe imports Matplotlib, but this server does not display notebook figures. Inheriting `module://matplotlib_inline.backend_inline` from Jupyter can make Matplotlib 3.11.2 raise `ValueError` in an isolated environment without `matplotlib-inline`. This failure was reproduced with NumPy 1.26.4 and Matplotlib 3.11.2; selecting `Agg` allowed the same import to succeed. The override applies to child processes only, leaving the notebook's plotting backend and the CLI unchanged. See [Matplotlib's backend environment variable](https://matplotlib.org/stable/install/environment_variables_faq.html#mplbackend).

## Health, jobs, and cancellation

- `GET /api/health` reports API liveness and system information. A healthy HTTP server alone does not mean its renderer works.
- `GET /api/ready` returns HTTP 200 with `ready: true`, or HTTP 503 with safe per-check diagnostics. Import failures identify the attempted module, category, dependency versions, and traceback module/function/line locations, including nested causes. Raw exception messages, source lines, locals, and filesystem paths are excluded. Results are cached for that server process. Restart after changing packages; modifying a running notebook's packages does not repair loaded native modules.
- `POST /api/jobs` checks readiness before queuing work. NVIDIA credentials can be used without a Google key. Cached AI reuse does not require an AI key.
- **Reuse Job** accepts an existing terminal job with its original source still present. Active jobs cannot be reused. **Bypass AI** requires `gemini_response.json`; switching it off performs fresh analysis. Reuse replaces that job's state and can overwrite its rendered outputs, so download results you want to keep first.
- Detected `youtube_cookies.txt` is forwarded to the downloader. Treat cookies as credentials and provide them through the notebook's optional cookie input, never commit them.
- **Cancel Job** requests cancellation. The current download, transcription, AI retry sequence, or render can finish before the next safe checkpoint. The record remains visible with `cancel_requested` until the worker stops; a subsequent delete removes the terminal record. It does not delete output files.
- A failed clip cannot make a job appear fully completed. Successful clips remain available with the failed job, and `render_manifest.json` records per-clip outcomes. A backend restart marks interrupted jobs failed instead of leaving them running forever.
- Job persistence uses atomic file replacement. It is designed for one server process, not a shared multi-process queue or database.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `Error in sitecustomize` / `No module named 'wrapt'` | Kaggle's Python startup hook imports `wrapt` from the base standard-library path, even in an isolated venv. Rerun setup with the updated `requirements-kaggle.txt`, which includes `wrapt`. Early bootstrap warnings can occur before it is installed. Keep system site packages disabled. Studio disables Kaggle's automatic Gemini proxy hook to preserve its configured API-key/provider behavior. |
| `NGROK_AUTHTOKEN: not configured` / `NGROK_AUTHTOKEN is missing` | The server has not received the token. In Kaggle Add-ons → Secrets, use the exact label `NGROK_AUTHTOKEN` and attach/enable it for this notebook. Rerun Configure API Keys, confirm its status is `configured` (no value is printed), then rerun Start Backend Server. A Python variable alone is not exported to child processes. |
| Readiness is 200, then `PyngrokNgrokHTTPError` / `tunnel.failed` | The backend passed readiness, but the ngrok agent rejected tunnel creation. Update the checkout to include the full upstream URL and v2/v3 HTTPS option fix, then rerun Start Backend Server. Check `error.http_status`, `error.ngrok_error_code`, and `error.agent_error_code` in `kaggle-studio.jsonl` if it still fails. The older generic exception alone does not establish an invalid token, account limit, or network problem; use the actual ngrok code to distinguish them. The code fix does not require rebuilding the ML environment. |
| `ensurepip ... returned non-zero exit status 1` during venv creation | Update `scripts/kaggle_studio.py` to the repair revision and rerun the dependency setup cell. A valid partial venv is repaired in place; no clone, output deletion, or kernel package install is needed. The screenshot/error alone does not identify why that runtime's `ensurepip` failed. |
| `Renderer readiness failed (renderer: ValueError)` | FastAPI started, but an import failed before opening the tunnel. The older message hides the exact cause; it does not prove a NumPy ABI problem. Update the checkout to include the managed `MPLBACKEND=Agg` fix and richer diagnostics, then restart the server. No reinstall is needed for the backend-setting fix. If it still fails, inspect the reported `import`, `category`, `trace`, and versions before changing dependencies. |
| Confirmed NumPy/MediaPipe ABI error (`category=numpy_abi_mismatch`) | Stop the backend and rerun setup in a fresh isolated environment. To retain an existing environment for inspection, use a new `--venv` directory for **both** setup and start. Do not install into the running notebook kernel. |
| All clips fail with `raw_box_tensor` / `num_boxes_`, then `Graph has errors` | The old full-range model is incompatible with the MediaPipe 0.10.21 Tasks detector graph. Update the code and restart the backend. It now downloads the pinned short-range model to `blaze_face_short_range_v1.tflite` on first use; the old full-range cache is left untouched. No environment rebuild is needed. Reuse the failed job with **Bypass AI** if its source and `gemini_response.json` remain available. Saved errors identify this assertion as `mediapipe_model_mismatch`, including under `error.causes` when wrapped by the inference check. |
| Setup fails | Fix the first visible dependency/download error and rerun setup. No ready banner or tunnel should appear after a failed check. |
| Missing FFmpeg | Run the notebook's checked FFmpeg installation cell, then restart the server. |
| Backend Not Connected | Keep the notebook server cell running and use its latest Public URL. The URL can change after a restart. |
| Gemini preview model returns 404 for `v1`, then fallback says JSON mode requires `v1beta` | Update the checkout with the explicit Gemini API routing fix and restart the backend server cell so the worker imports the new code. Analysis and voiceover now use the API-key Developer API with `v1beta`, including fallbacks. No environment rebuild, key replacement, or model change is required for this routing fix. Reconnect to the latest tunnel URL, then retry the failed job. |
| Gemini still returns 404 on `v1beta` | Check that the selected model supports `generateContent` and is available to your account using Google's model list/documentation. A 404 alone does not prove the model was retired or the key is invalid. Main and fallback availability can differ; choose accessible models rather than repeatedly retrying the same 404. |
| AI analysis appears slow | Providers may retry transient errors such as Gemini 503. Follow notebook logs and the job's analysis progress; this is distinct from renderer readiness. Cancellation waits for the current AI operation to return. |
| SSE fails | Studio sends the ngrok bypass header on its fetch-based event stream and retries it. Polling continues as a fallback. |
| Video preview fails but API calls work | Native video/link requests cannot set the ngrok bypass header. Use **Open video / Download**, continue through the tunnel warning if shown, and retry. Media remains streamed rather than copied entirely into browser RAM. |
| Python 3.13+ runtime | Use Python 3.12 for the constrained Kaggle environment; setup stops early on unsupported versions. |

For a local diagnostic in the managed environment, run `python -m web.api.readiness --check` with that environment's Python. It exits nonzero if a required import or FFmpeg fails and does not start a tunnel.

Gemini routing is explicit because [Google GenAI SDK 3.0.0 changed its default to `v1`](https://github.com/googleapis/python-genai/releases/tag/v3.0.0), whereas the reported analysis requests rejected JSON mode and requested `v1beta`. Setting `HttpOptions(api_version="v1beta")` also overrides an inherited `GOOGLE_GENAI_API_VERSION=v1`; `vertexai=False` keeps this API-key integration on the Developer API. Readiness records the installed SDK version under `diagnostics.versions.google-genai` but does not contact Gemini or validate model access. Saved Gemini exceptions retain their numeric HTTP status as `error.http_status` (or under `error.causes` for wrapped failures).

Face detector construction now includes a synthetic image inference before the detector is cached. This runs when rendering or speaker estimation first requests the detector; `/api/ready` still checks imports and FFmpeg, without downloading or testing model weights. Short-range faces are the supported default for this Tasks graph; distant/small-face tracking may be weaker than the former intended full-range behavior. See the [face-tracking guide](../wiki/4-Face-Tracking-and-Auto-Framing.md) for the existing YOLO alternative.

In the notebook, use the same environment builder as startup so the diagnostic also uses the server's headless Matplotlib backend:

```python
from scripts.kaggle_studio import runtime_environment

result = subprocess.run(
    [str(STUDIO_PYTHON), "-m", "web.api.readiness", "--check"],
    cwd=PROJECT_DIR,
    env=runtime_environment(STUDIO_ENV),
    check=False,
)
print("Readiness exit code:", result.returncode)
```

The preserved checkout is not updated by rerunning the clone cell. After updating its files, restart the notebook kernel (and rerun the configuration cells) or reload the imported bootstrap module before starting again. Otherwise Python can retain the older `runtime_environment` function.

## Saved flow logs

Structured diagnostic files are created automatically when setup or the backend starts. Each line is one JSON event (`.jsonl`), written immediately and retained after the process exits:

| File under the project directory | Recorded flow |
| --- | --- |
| `logs/kaggle-studio.jsonl` | Platform/venv validation, pip bootstrap and fallback, dependency install/check, Deno setup, FastAPI launch, readiness results, tunnel startup and shutdown, process exit codes. |
| `logs/studio-api.jsonl` | API module import before FastAPI starts, application startup/shutdown, HTTP requests, native import readiness, job queue/cancellation, pipeline stages, per-clip results and caught renderer errors/fallbacks. |

In Kaggle the default directory is `/kaggle/working/opensource-clipping/logs/`. Open the notebook's **Files/Output** browser and download both `.jsonl` files. Download rotated files (`.jsonl.1`, `.jsonl.2`, `.jsonl.3`) too if the relevant event has moved there. The notebook includes a cell listing these files after stopping the server. The files are ordinary text and can be attached for diagnosis without conversion.

Each file rotates at approximately **5 MiB**, with **three backups** (about 20 MiB per file family). Existing logs are appended, not cleared on restart. The `logs/` directory is ignored by Git. Set `OSC_LOG_DIR` before setup/start to use a different writable directory. Use a single managed server per checkout; supervisor and API have separate files to avoid competing rotations between those two processes.

To follow an error:

1. Find a `level: ERROR` event or a failed readiness check. HTTP 4xx responses and recoverable fallbacks appear as warnings.
2. Match `run_id` between supervisor and API files. Match `job_id` to follow one job through download → transcribe → analyze → metadata → renderer import → optional diarization → render preparation → render → finalize. A reused job keeps its ID, so also check timestamps and `job_started` boundaries.
3. Follow `request_id` from an HTTP request to its `job_queued` event, then use `job_id` for thread-worker events. Handled HTTP responses expose the generated ID as `X-Request-ID`; the default unhandled-500 response may not carry that header, but the failed request event still has the ID.
4. Inspect `stage`, `error.type`, `error.code`, and the last entries in `error.traceback`. Each frame identifies a Python module, function and line number; `causes` retains nested exception locations. Readiness uses the corresponding fields under `diagnostics.checks`. Numeric subprocess exit codes and OS error numbers are retained when available. Tunnel failures also preserve numeric HTTP status, a strict `ERR_NGROK_*` code and the numeric agent error code when the SDK supplies them. `tunnel.configured` records the API schema and pyngrok version. These codes are extracted without saving response bodies, URLs or headers.
5. For rendering, `clip_index` is the clip's processing position and `clip_rank` is its metadata rank. `elapsed_ms` shows stage/clip duration. `renderer_fallback` means processing can continue; the final `job_completed`, `job_failed`, or `job_cancelled` event determines the outcome.

For example, a failed AI stage produces `stage_failed` with `stage: analyze`, the job ID, exception class and traceback locations, followed by `job_failed`. If the process crashes abruptly, the last started stage and supervisor exit code help locate the interruption even when Python cannot produce a traceback.

The saved files contain operational metadata only. They omit credential values, headers, request bodies, source URLs, transcripts, AI responses, raw exception messages, command arguments, source-code lines, local variables and absolute traceback paths. Third-party console output is not copied into these files. Keep the notebook console log too for pip/FFmpeg output or failures before the Python helper starts (for example clone/system-package setup). A full disk or unwritable log directory emits a fixed warning and does not replace the original application result.

The managed launcher is `python -m scripts.studio_server`; it configures logging before importing the API so import-time errors are saved. Manual `uvicorn web.api.app:app` remains supported. Existing `main.py` CLI usage does not enable these file logs; renderer hooks activate only inside an API job context.

Pip recovery first uses the host pip's supported [`--python` option](https://pip.pypa.io/en/stable/topics/python-option/) to install into the target environment. If host pip is absent or incompatible, setup downloads the official [PyPA `get-pip.py` bootstrap](https://pip.pypa.io/en/stable/installation/#get-pip-py) over HTTPS and runs it with the target interpreter. It never installs packages into the notebook interpreter. A functioning existing target pip is reused.

The tunnel exposes the existing Studio API, including settings and shutdown. No new authentication mechanism is introduced by this repair; treat the tunnel URL as private and stop it when finished.

## Regression tests

Use a separate local virtual environment, install `requirements-test.txt`, then run:

```sh
python -m pytest tests -q
node --test tests/studio.test.cjs
```

The suite isolates the job store, disables `.env` loading, and mocks model, AI, and download operations. `pytest.ini` restricts default discovery to `tests/`; operational uploader scripts elsewhere are not part of this suite. The CLI help smoke test runs without site packages, and another smoke test starts the real Uvicorn entry point in a fresh interpreter.

Tunnel regression tests use the real pyngrok SDK to construct requests, with agent process/network access replaced by local test doubles. They check the upstream URL, v2/v3 HTTPS options, rejection/cleanup behavior, and safe error-code logging. They do not open a public tunnel or validate account credentials.

Gemini regression tests also use the real SDK with a local mock HTTP transport, synthetic credentials, and blocked outbound sockets. They verify Developer API `v1beta` URLs, JSON request configuration, primary/fallback routing, transient-error retries, and voiceover behavior without paid requests.

Face detector regression tests cover CLI/Studio model selection with a stale full-range cache, first-inference validation, failure cleanup without caching a broken graph, detector reuse, and speaker-estimation image types. Native CPU validation with MediaPipe 0.10.21 reproduced the full-range assertion and successfully ran repeated inference with the pinned short-range model; this is separate from the lightweight mocked test suite.

These tests do not replace a Kaggle GPU smoke test: start the notebook, verify the ready banner, connect Studio, run a short video, inspect/download a clip, and exercise cancel/reuse. Validate each selected AI provider and optional diarization model with your own credentials in that runtime.
