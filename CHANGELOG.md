# Changelog

All notable changes to the **OpenSource Clipping** project will be documented in this file.

**The Standard Structure (SemVer)**
- **Major (X.y.z)**: Incremented for incompatible API changes (breaking changes).
- **Minor (x.Y.z)**: Incremented for new functionality introduced in a backward-compatible manner.
- **Patch (x.y.Z)**: Incremented for backward-compatible bug fixes or minor patches.

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
- **[Planned] Auto Camera Switch (Full 9:16)**: Automatic active speaker detection that goes full-frame 9:16 and switches between speakers based on who is currently talking (instead of split-screen).
- **[Planned] 16:9 Speaker Switch**: Automatic Active Speaker detection for 16:9 that performs full frame cuts/switches (Speaker A / Speaker B / Wide Shot).
