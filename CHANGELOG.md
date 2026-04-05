# Changelog

All notable changes to the **OpenSource Clipping** project will be documented in this file.

**The Standard Structure (SemVer)**
- **Major (X.y.z)**: Incremented for incompatible API changes (breaking changes).
- **Minor (x.Y.z)**: Incremented for new functionality introduced in a backward-compatible manner.
- **Patch (x.y.Z)**: Incremented for backward-compatible bug fixes or minor patches.

## [v0.5.3] - 2026-04-05

### Fixed / Added
- **Resilient yt-dlp Subtitle Fallback**: Added a fail-safe `try/except` block enclosing the YouTube subtitle extraction phase. If YouTube blocks the download request due to strict IP rate limiting (`HTTP Error 429: Too Many Requests`), the pipeline dynamically recovers, downloads the video alone, and smoothly delegates word-tracking back to the active Whisper AI engine without fatal crashes.
- **Multilingual DLP Parsing**: Enlarged `yt-dlp` target subtitle language configuration to prioritize Indonesian (`id`) alongside English (`en`). Modified `runner.py`'s parser logic using regex mapping (`*.json3`) dynamically grabbing any regional configuration file.

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
- **[Planned] Auto-Split Screen for Podcasts**: Introducing intelligent face-tracking and framing for podcast styled videos:
  - **9:16 (Vertical)**: Automatically splits the screen (Top/Bottom) to show both speakers or goes full-screen 9:16 with automatic camera switching based on the active speaker's voice.
  - **16:9 (Horizontal)**: Automatic Active Speaker detection that performs full frame cuts/switches based on who is currently talking (Speaker A / Speaker B / Wide Shot).
