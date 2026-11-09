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

## Why the environment is isolated

`numpy<2` in a requirements file is insufficient when the notebook already has native libraries loaded or incompatible preinstalled packages. The reported `numpy.dtype size changed` error occurred when importing MediaPipe during rendering, after AI analysis had succeeded.

The Kaggle constraints keep NumPy 1.26.4, MediaPipe 0.10.21, OpenCV 4.11, JAX 0.4.35, and the Pyannote 3.x API on compatible dependency families. Torch 2.5.1, Torchaudio 2.5.1, and Torchvision 0.20.1 are kept together. General CLI requirements and CLI arguments are unchanged. This is a constrained environment, not a complete transitive lockfile; actual runtime import checks are still required.

The readiness probe checks NumPy, the MediaPipe/Studio renderer, Whisper/engine imports, and FFmpeg in a separate process. It does not download models, call paid AI APIs, validate credentials, or prove that GPU inference, gated Hugging Face models, or a particular video will succeed.

## Health, jobs, and cancellation

- `GET /api/health` reports API liveness and system information. A healthy HTTP server alone does not mean its renderer works.
- `GET /api/ready` returns HTTP 200 with `ready: true`, or HTTP 503 with safe per-check diagnostics. Results are cached for that server process. Restart after changing packages; modifying a running notebook's packages does not repair loaded native modules.
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
| `ensurepip ... returned non-zero exit status 1` during venv creation | Update `scripts/kaggle_studio.py` to the repair revision and rerun the dependency setup cell. A valid partial venv is repaired in place; no clone, output deletion, or kernel package install is needed. The screenshot/error alone does not identify why that runtime's `ensurepip` failed. |
| NumPy/MediaPipe ABI error or HTTP 503 from `/api/ready` | Stop the backend and rerun setup in a fresh isolated environment. To retain an existing environment for inspection, use a new `--venv` directory for **both** setup and start. Do not install into the running notebook kernel. |
| Setup fails | Fix the first visible dependency/download error and rerun setup. No ready banner or tunnel should appear after a failed check. |
| Missing FFmpeg | Run the notebook's checked FFmpeg installation cell, then restart the server. |
| Backend Not Connected | Keep the notebook server cell running and use its latest Public URL. The URL can change after a restart. |
| AI analysis appears slow | Providers may retry transient errors such as Gemini 503. Follow notebook logs and the job's analysis progress; this is distinct from renderer readiness. Cancellation waits for the current AI operation to return. |
| SSE fails | Studio sends the ngrok bypass header on its fetch-based event stream and retries it. Polling continues as a fallback. |
| Video preview fails but API calls work | Native video/link requests cannot set the ngrok bypass header. Use **Open video / Download**, continue through the tunnel warning if shown, and retry. Media remains streamed rather than copied entirely into browser RAM. |
| Python 3.13+ runtime | Use Python 3.12 for the constrained Kaggle environment; setup stops early on unsupported versions. |

For a local diagnostic in the managed environment, run `python -m web.api.readiness --check` with that environment's Python. It exits nonzero if a required import or FFmpeg fails and does not start a tunnel.

Pip recovery first uses the host pip's supported [`--python` option](https://pip.pypa.io/en/stable/topics/python-option/) to install into the target environment. If host pip is absent or incompatible, setup downloads the official [PyPA `get-pip.py` bootstrap](https://pip.pypa.io/en/stable/installation/#get-pip-py) over HTTPS and runs it with the target interpreter. It never installs packages into the notebook interpreter. A functioning existing target pip is reused.

The tunnel exposes the existing Studio API, including settings and shutdown. No new authentication mechanism is introduced by this repair; treat the tunnel URL as private and stop it when finished.

## Regression tests

Use a separate local virtual environment, install `requirements-test.txt`, then run:

```sh
python -m pytest tests -q
node --test tests/studio.test.cjs
```

The suite isolates the job store, disables `.env` loading, and mocks model, AI, and download operations. `pytest.ini` restricts default discovery to `tests/`; operational uploader scripts elsewhere are not part of this suite. The CLI help smoke test runs without site packages, and another smoke test starts the real Uvicorn entry point in a fresh interpreter.

These tests do not replace a Kaggle GPU smoke test: start the notebook, verify the ready banner, connect Studio, run a short video, inspect/download a clip, and exercise cancel/reuse. Validate each selected AI provider and optional diarization model with your own credentials in that runtime.
