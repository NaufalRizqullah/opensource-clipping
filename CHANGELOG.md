# Changelog

All notable changes to the **OpenSource Clipping** project will be documented in this file.

**The Standard Structure (SemVer)**
- **Major (X.y.z)**: Incremented for incompatible API changes (breaking changes).
- **Minor (x.Y.z)**: Incremented for new functionality introduced in a backward-compatible manner.
- **Patch (x.y.Z)**: Incremented for backward-compatible bug fixes or minor patches.

## [v0.7.8] - 2026-04-18

### Fixed
- **Dev-Mode Stability**: Fixed `UnboundLocalError: cannot access local variable 'frame_utama_siap'` that occurred in `--dev-mode` when B-roll transitions were triggered.

---

## [v0.7.7] - 2026-04-18

> ⚠️ **Experimental**: The dev-mode stabilizer visualization is currently experimental.

### Added
- **Developer Visualization Mode** (`--dev-mode`): New flag for 9:16 target ratio that renders a 16:9 context view. It visualizes the "Director's view" of the stabilization process by dimming the background outside the 9:16 crop, drawing boundary lines, and labeling tracking targets. Useful for fine-tuning AI tracking speed and deadzones.

---

## [v0.7.6] - 2026-04-18

### Added
- **Face Detection Visualization**: Introduced the `--box-face-detection` CLI flag. This draws a yellow bounding box around every detected face in the source frame, with smooth linear interpolation between detection intervals. Useful for debugging face tracking accuracy.

---

## [v0.7.5] - 2026-04-18

### Added
- **Configurable Tracking Parameters**: Exposed internal camera tracking constants as CLI flags. If not provided, the system defaults to the optimized values introduced in v0.7.4.
  - `--track-step`: Face detection frequency in seconds.
  - `--track-deadzone`: Camera deadzone ratio.
  - `--track-smooth`: Camera smoothing/catch-up factor.
  - `--track-jitter`: Micro-jitter pixel threshold.
  - `--track-snap`: Jump threshold for hard cuts between speakers.

---

## [v0.7.4] - 2026-04-18

### Improved
- **Optimized Face Tracking & Camera Responsiveness**: Fine-tuned internal camera parameters for tighter and faster centering during face tracking.
  - **Higher Detection Frequency**: Increased face check rate to every 0.25s (previously 0.5s) to reduce camera lag in dynamic scenes.
  - **Tighter Deadzone**: Reduced the safe "no-move" zone to 15% (previously 25%), ensuring the subject stays closer to the center of the frame.
  - **Responsive Catch-up**: Increased smoothing speed to 30%, making the camera follow movement more assertively while maintaining fluid motion.
  - **Micro-jitter Prevention**: Increased the jitter threshold to 5px to ensure a steady shot despite high-frequency AI detection updates.

---

## [v0.7.3] - 2026-04-17

### Added
- **No-Subs Mode** (`--no-subs`): New flag to disable subtitle rendering in the final video output. The transcription process (Whisper) still runs to enable AI analysis and ranking, but the text is not burned into the video. Useful for creators who want clean B-roll or their own manual captioning later.

---

## [v0.7.2] - 2026-04-13

### Improved
- **Multi-Speaker Multi-Scene Podcast Support**: Both `--split-screen` and `--camera-switch` now handle podcast formats with **3+ speakers across multiple scenes** (e.g., 2 speakers in one camera shot + 1 speaker in a separate solo shot). Use `--diarization-speakers 3` to enable.
  - **Split-Screen**: Per-speaker frozen frame cache — each speaker now has their own fallback crop instead of a single shared one. Both panels (top & bottom) can independently fall back to their speaker's last valid frame when the speaker is not visible in the current scene.
  - **Camera-Switch**: Scene-aware simultaneous speech — when 2+ speakers talk simultaneously but are in **different scenes** (one is solo-scene type), the system stays on the current speaker instead of switching to blurred pillarbox. Wide-shot only triggers when all active speakers share the same physical frame.
  - **Hybrid Visual Auto-Detection**: Added the `auto` option (which is now the default) for `--diarization-speakers`. When set to `auto`, the system performs a rapid visual scan of 20 sampled frames to find the maximum number of people physically appearing together. This number is then injected dynamically into Pyannote as a boundary guide, optimizing accuracy and significantly reducing "over-segmentation" issues.
  - **Split-Screen**: Refactored panel rendering into a reusable `_build_panel()` helper; both panels now use identical fallback logic.

### Notes
- Backward compatible with existing 2-speaker podcast workflows — no changes needed for standard usage.
- For 3-speaker podcasts, set `--diarization-speakers 3` to let Pyannote detect all speakers correctly.

---

## [v0.7.1] - 2026-04-08

### Added
- **TikTok Indonesian Metadata Support**: Added new fields `tiktok_title_id` and `tiktok_caption_id` specifically for Indonesian-localized TikTok content.
- **Enhanced Validations**: Added warnings if English fields contain Indonesian text or if Indonesian fields contain non-Indonesian text to ensure strict locale compliance.
- **Enriched Preview**: Updated `print_preview()` to display the new TikTok ID fields in the CLI output.

---

## [v0.7.0] - 2026-04-06

> ⚠️ **Experimental**: The camera-switch feature in this version is experimental and may be rolled back or undergo significant changes in future updates.

### Added
- **Camera-Switch Mode** (`--camera-switch`): New full 9:16 rendering mode for podcast-style videos. Uses Pyannote speaker diarization to detect who is speaking at each moment and automatically switches the crop to focus on the active speaker — similar to a live director cutting between camera angles.
  - **Single speaker active** → full 9:16 crop centred and face-tracked on that speaker
  - **Both speakers simultaneously** → **blurred pillarbox** (original 16:9 frame centred with blurred background filling the 9:16 canvas — no black bars)
  - **No one speaking** → holds on the last active speaker
  - **Minimum hold duration** (`--switch-hold-duration`, default `2.0` s) prevents flickering when speakers alternate rapidly
- **Blurred Pillarbox Helper** (`_make_blurred_pillarbox`): Internal renderer that produces TikTok/Reels-style blurred letterbox/pillarbox by using a scaled + Gaussian-blurred version of the source frame as background, with the original frame composited at the centre.
- **`get_active_speakers()` in `diarization.py`**: New helper that returns *all* speakers active at a given timestamp (vs. the existing `get_active_speaker()` which returns only the first). Enables detection of simultaneous speech for the blurred pillarbox transition.

### Changed
- **Diarization trigger in `runner.py`**: Speaker diarization is now also triggered when `--camera-switch` is active (previously only triggered for `--split-screen`).

### Notes
- `--camera-switch` and `--split-screen` are mutually exclusive; if both flags are passed, `--split-screen` takes precedence.
- Requires `HF_TOKEN` set in `.env` (same as split-screen).

---

## [v0.6.3] - 2026-04-06

### Fixed
- **Pyannote 3.1 Compatibility**: Fixed `'DiarizeOutput' object has no attribute 'itertracks'` by correctly extracting the `Annotation` object from the newer `DiarizeOutput` wrapper in `clipping/diarization.py`.

---

## [v0.6.2] - 2026-04-06

### Fixed
- **Diarization Robustness**: Fixed `'DiarizeOutput' object has no attribute 'itertracks'` by adding logic to handle wrapped annotation objects in newer Pyannote versions and adding debug diagnostic info.

---

## [v0.6.1] - 2026-04-06

### Fixed
- **Pyannote API Compatibility**: Fixed `Pipeline.from_pretrained() got an unexpected keyword argument 'use_auth_token'` error by updating to the newer `token` parameter, with automatic fallback to `use_auth_token` for older versions.

---

## [v0.6.0] - 2026-04-06

> ⚠️ **Experimental**: The split-screen feature in this version is experimental and may be rolled back or undergo significant changes in future updates. Use with discretion.

### Added
- **Podcast Split-Screen Mode**: Introduced `--split-screen` flag that activates automatic **Pyannote speaker diarization** to detect 2 speakers in podcast-style videos. When enabled on `9:16` ratio, the renderer produces a **top-bottom split-screen** layout where each panel independently face-tracks a different speaker. The active speaker's panel is highlighted with a yellow border while the inactive speaker's panel is subtly darkened.
- **Diarization Module**: New `clipping/diarization.py` module powered by `pyannote/speaker-diarization-3.1` with GPU acceleration support, automatic segment merging, and active speaker lookup.
- **CLI Parameters**: Added `--split-screen` (enable split-screen mode) and `--diarization-speakers` (set expected number of speakers, default 2).
- **Graceful Fallback**: If diarization fails (missing `HF_TOKEN`, model error, or only 1 speaker detected), the system automatically falls back to the standard single-panel renderer.

### Dependencies
- Added `pyannote.audio`, `torch`, `torchaudio` to `requirements.txt`.

---

## [v0.5.3] - 2026-04-05

### Fixed / Added
- **Decoupled Subtitle Fetching**: Completely separated the video downloader from the subtitle downloader in `yt-dlp`. Subtitles are now independently grabbed utilizing `skip_download: True`. This ensures that even if YouTube blocks the subtitle request (`HTTP Error 429: Too Many Requests`), the video will still be downloaded without interruption, falling back to Whisper flawlessly.
- **Multilingual Priority Ladder**: Implemented a sequential priority loop that searches for English (`en`) subtitles first, followed by Indonesian (`id`) if unavailable. `runner.py` now supports automatic regional `.json3` mapping via glob regex matching.

---

## [v0.5.2] - 2026-04-05

### Fixed
- **JSON3 Word-Level Slicing**: Upgraded the YouTube `json3` subtitles parsing system to intelligently split broader subtitle sentences into isolated space-separated characters while distributing original timestamps evenly. This ensures kinetic karaoke annotations emphasize authentically, precisely matching `faster-whisper`'s per-word word tracking level.

---

## [v0.5.1] - 2026-04-05

### Fixed
- **JSON3 Text Sanitizer**: Implemented a regex sanitizer for YouTube auto-generated captions (`parse_youtube_json3_subs`) to automatically strip unprintable glyphs, emojis, music notes, and zero-width identifiers (e.g. `\u200b`). This prevents `FFmpeg` from crashing during the subtitle burn-in phase due to unsupported font fallbacks (`failed to find any fallback with glyph`).

---

## [v0.5.0] - 2026-04-05

### Added
- **YouTube DLP Subtitles**: Added the `--use-dlp-subs` flag to prioritize parsing YouTube's built-in manual and auto-generated `json3` subtitles, completely bypassing `faster-whisper` and drastically speeding up the audio-transcription phase of the clipping pipeline.

---

## [v0.4.2] - 2026-04-05

### Added
- **Gemini Fallback Model**: Added robust automatic fallback mechanism to retry with a secondary AI model (`--gemini-fallback-model`) if the main Gemini content engine exhausts its retry limits.

---

## [v0.4.1] - 2026-04-05

### Added
- **YOLOv8 GPU Face Tracking**: Integrated PyTorch-based YOLO tracking (via `ultralytics`) as a high-powered alternative for face cropping (`--face-detector yolo`). Features auto-downloading of models (`8n`, `8s`, `8m`, `8n_v2`, `9c`) from Hugging Face for dynamic GPU execution.

---

## [v0.4.0] - 2026-04-05

### Added
- **Standalone YouTube Auto-Uploader**: Created a dedicated CLI (`run_upload.py`) with a `youtube_uploader` package for automated background uploading and scheduling without interrupting the rendering pipeline.

### Changed
- **Centralized Outputs Directory**: Refactored the engine (`config.py`, `studio.py`, `runner.py`) so all generated media, thumbnails, and manifests are now neatly sandboxed inside an `outputs/` folder.
- **MediaPipe Tracking Upgrade**: Upgraded the smart auto-framing AI from the basic short-range model to the more robust `BlazeFace (Full-Range)` model. This resolves face-loss issues on wide-shot inputs and podcast frames.
- **Typographic Sweet Spots**: Fine-tuned default subtitles margins and CSS scaling (adjustable in `config.py`).

### Removed
- Deprecated and removed the obsolete `v2-youtube-clipping.ipynb` legacy notebook.

---

## [v0.3.0] - 2026-04-04

### Added
- **Modular CLI Architecture**: Completely refactored the project from a monolithic Jupyter Notebook into a clean, CLI-based Python project (`main.py` & `clipping/` package).
- **Dynamic Scaling & Font Pairing (Kinetic Typography)**: Specific core words within the running subtitles now scale dynamically for emphasis (e.g., "I eat *RICE*").
- **Contextual B-roll Injection**: The system now automatically fetches and splices contextual B-roll footage from Pexels based on relevant keywords spoken in the clip.
- **Multi-language Documentation**: Added `README.md` (English) and `README_ID.md` (Indonesian), complete with execution guides for Google Colab.
- **Environment Management**: Introduced `.env` support to securely manage API keys, removing hardcoded/Colab userdata from the core logic.

### Changed
- **Gemini Model**: Configured support for both `gemini-3-flash-preview` and `gemini-2.5-flash` via config parameters.
- **Dependencies**: Streamlined dependency management, supporting both `uv` package manager and standard `requirements.txt` / `pyproject.toml`.

---

## [Planned / Upcoming Features]

- **[Planned] Wefluence Integration**: Building an automated batch clipping system that pulls source videos directly from Google Drive and YouTube, crafting compelling video compilations.
- **[Done → v0.7.0] Auto Camera Switch (Full 9:16)**: Implemented as `--camera-switch` flag with blurred pillarbox for simultaneous speech.
- **[Planned] 16:9 Speaker Switch**: Automatic Active Speaker detection for 16:9 that performs full frame cuts/switches (Speaker A / Speaker B / Wide Shot).
