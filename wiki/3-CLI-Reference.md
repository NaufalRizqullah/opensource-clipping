# ⚙️ CLI Reference

Complete reference for all command-line arguments. Run `python main.py --help` for a quick overview.

---

## Core Settings

| Argument | Default | Description |
|---|---|---|
| `--url`, `-u` | — | Video URL to process (**Required**) |
| `--source` | `youtube` | Video source platform: `youtube`, `tiktok`, `instagram`, `gdrive` |
| `--clips`, `-n` | `7` | Number of highlight clips to generate |
| `--ratio`, `-r` | `9:16` | Output aspect ratio: `9:16`, `16:9`, `1:1`, `3:4`, `4:5` |
| `--source-height` | `max` | Preferred source download max height (`max`, `1080`, `1440`, `2160`) |
| `--render-height` | `1080` | Target output render height (`1080`, `1440`, `2160`, `source`) |

---

## AI Provider Settings

| Argument | Default | Description |
|---|---|---|
| `--ai-provider` | `gemini` | AI provider for analysis: `gemini`, `nvidia`, or `vllm` |
| `--gemini-model` | `gemini-3-flash-preview` | Gemini model name |
| `--gemini-fallback-model` | `gemini-2.5-flash` | Fallback model if main model fails |
| `--nvidia-model` | `deepseek-ai/deepseek-v4-pro` | Model name for NVIDIA NIM API |
| `--vllm-model` | `VLLM_MODEL` or `google/gemma-4-E2B-it-qat-w4a16-ct` | Exact served model ID/alias |
| `--local-base-url` | `VLLM_BASE_URL` or `http://127.0.0.1:8001` | vLLM server root, optional `/v1` suffix |
| `--local-max-tokens` | `VLLM_MAX_TOKENS` or `4096` | Analysis output-token budget |
| `--local-context-limit` | `VLLM_CONTEXT_LIMIT` or unset | Conservative client ceiling; does not increase server context |
| `--local-timeout` | `VLLM_TIMEOUT` or `900` | Overall request/retry budget, seconds |
| `--load-gemini-json` | `False` | Load saved `gemini_response.json` to bypass AI call |

vLLM does not require Google/NVIDIA keys and never falls back to cloud. Optional authentication uses `VLLM_API_KEY`. CLI voiceover follows vLLM when selected; TTS still requires Internet. The experimental [Kaggle Gemma 4 Q4 notebook](../notebooks/kaggle-vllm-gemma4.ipynb) isolates the server environment. See [provider behavior and validation](../docs/providers.md).

---

## Content & Hook Settings

| Argument | Default | Description |
|---|---|---|
| `--words-per-sub` | `5` | Max words per karaoke subtitle group |
| `--hook-duration` | `3` | Hook teaser duration (seconds) |
| `--hook-source` | `None` | Path or URL for custom hook video (.mp4) |
| `--hook-source-start` | `0.0` | Start time in seconds for custom hook |
| `--no-broll` | — | Disable B-roll footage |
| `--no-hook` | — | Disable hook glitch teaser (V1) |
| `--no-bgm` | — | Disable background music |
| `--no-subs` | — | Disable all subtitle rendering |
| `--no-karaoke` | — | Use clean text instead of karaoke highlight |

---

## Hook V2 & Segment Trimming

| Argument | Default | Description |
|---|---|---|
| `--hook-v2` | `False` | Enable Multi-Hook Intro V2 mode |
| `--hook-v2-items` | `3` | Number of micro-hooks to generate |
| `--hook-v2-style` | `controversial_fast_glitch` | Style prompt for AI hook selection |
| `--white-flash-duration` | `0.12` | Duration of flash transition between hooks (seconds) |
| `--no-segment-trim` | `False` | Disable AI segment trimming (render full clip) |
| `--silence-trim` | `False` | Aggressively trim silence/dead air |

---

## Subtitle & Typography

| Argument | Default | Description |
|---|---|---|
| `--font-style` | `HORMOZI` | Font preset: `DEFAULT`, `STORYTELLER`, `HORMOZI`, `CINEMATIC` |
| `--advanced-text` | `False` | Enable kinetic typography (word scaling & animation) |
| `--advanced-text-hook` | `False` | Enable kinetic typography on hook teaser |

---

## BGM (Background Music)

| Argument | Default | Description |
|---|---|---|
| `--bgm-mode` | `ducking` | `ducking` (auto-lower during speech) or `background` (constant low volume) |

---

## Video Effects (Ambient Glow)

| Argument | Default | Description |
|---|---|---|
| `--edge-glow` | `False` | Apply ambient edge glow to the entire output video (hook, clip, broll, voiceover). By default, glow only appears on voice-over intros. |
| `--edge-glow-mode` | `smooth` | Edge glow rendering strategy: `default` (original 10s loop, may stutter at loop points), `smooth` (10s loop with auto-adjusted speed for seamless loop), `full` (renders full duration, heavier but zero stutter). |

---

## Watermark Settings

| Argument | Default | Description |
|---|---|---|
| `--watermark` | `False` | Enable watermark overlay on rendered clips |
| `--text` | `None` | Watermark text to overlay (e.g. 'Channel Name') |
| `--image` | `None` | Path to watermark image (PNG with alpha recommended, also supports JPG, JPEG, WEBP) |
| `--opacity` | `70` | Watermark opacity in percent (1-100) |
| `--position` | `center-right`| Watermark position (9 anchors: `top-left`, `center-right`, etc.) |
| `--padding` | `0` | Watermark padding from the nearest edge in pixels |
| `--watermark-font-size` | `0` | Watermark font size in pixels (0 = auto ~3% frame height) |
| `--watermark-scale` | `15` | Image watermark height as % of frame height (1-100) |

---

## Podcast / Split-Screen

| Argument | Default | Description |
|---|---|---|
| `--split-screen` | `False` | Enable split-screen mode (9:16 only) |
| `--dynamic-split` | `False` | Auto-toggle between full and split based on speakers |
| `--split-trigger` | `diarization` | Trigger: `diarization` (audio) or `face` (visual count) |
| `--diarization-speakers` | `auto` | Number of speakers or `auto` for visual detection |
| `--camera-switch` | `False` | Enable camera-switch mode (cinematic speaker switching) |
| `--switch-hold-duration` | `2.0` | Min seconds before switching speakers |
| `--split-zoom` | `1.0` | Manual zoom factor for split panels |
| `--split-v-align` | `0.5` | Vertical alignment (0.0=top, 0.5=center, 1.0=bottom) |
| `--split-auto-zoom` | `False` | Auto-zoom to isolate each speaker |
| `--split-max-zoom` | `2.5` | Maximum zoom limit for auto-zoom |

---

## Whisper (Transcription)

| Argument | Default | Description |
|---|---|---|
| `--whisper-model` | `large-v3` | Faster-Whisper model size |
| `--whisper-device` | `cuda` | Device: `cuda`, `cpu`, `auto` |
| `--whisper-compute-type` | `float16` | Compute type: `float32`, `float16`, `int8` |
| `--use-dlp-subs` | — | Use YouTube's built-in subtitles (skips Whisper if found) |

---

## Face Detection & Tracking

| Argument | Default | Description |
|---|---|---|
| `--face-detector` | `mediapipe` | AI model: `mediapipe` (CPU) or `yolo` (GPU) |
| `--yolo-size` | `8m` | YOLO model size: `8n`, `8s`, `8m`, `8n_v2`, `9c` |
| `--box-face-detection` | `False` | Draw yellow bounding boxes (debug) |
| `--static-crop` | `False` | Disable face tracking, use static center crop |

---

## Camera Tracking Tuning

| Argument | Default | Description |
|---|---|---|
| `--track-step` | `0.25` | Face detection frequency (seconds) |
| `--track-deadzone` | `0.15` | Camera deadzone ratio |
| `--track-smooth` | `0.30` | Camera catch-up speed factor |
| `--track-jitter` | `5` | Pixel threshold to ignore micro-shakes |
| `--track-snap` | `0.25` | Jump threshold for hard cut between speakers |
| `--track-conf` | `0.55` | Face detection confidence threshold |
| `--track-smooth-window` | `12` | Frame window for layout stability (~0.5s) |
| `--scene-cut-threshold` | `18` | Sensitivity for camera-cut detection |
| `--track-iou-threshold` | `0.2` | Overlap threshold for merging duplicate detections |

---

## Video Quality

| Argument | Default | Description |
|---|---|---|
| `--video-bitrate` | `auto` | Target bitrate (e.g. `8M`, `12M`, `auto`) |
| `--video-sharpen` | — | Apply subtle sharpening filter |
| `--video-cq` | `23` | NVENC CQ quality (lower = sharper) |
| `--video-crf` | `20` | libx264 CRF quality (lower = sharper) |
| `--video-preset` | `auto` | Encoder preset (NVENC: `p1`-`p7`, x264: `ultrafast`-`veryslow`) |
| `--video-scale-algo` | `lanczos` | Resize algorithm: `lanczos`, `bicubic`, `bilinear`, `area` |

---

## Developer / Debug Mode

| Argument | Default | Description |
|---|---|---|
| `--dev-mode` | `False` | Enable 16:9 visualization for 9:16 tracking |
| `--dev-mode-with-output` | `False` | Generate both final + dev dashboard simultaneously |
| `--dev-mode-with-output-merge` | `False` | Merged side-by-side ultrawide output |
| `--track-lines` | `False` | Draw crosshair tracking lines from face box |

---

