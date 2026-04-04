# Changelog

All notable changes to the **OpenSource Clipping** project will be documented in this file.

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
