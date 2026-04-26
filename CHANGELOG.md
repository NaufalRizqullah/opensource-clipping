# Changelog

All notable changes to the **OpenSource Clipping** project will be documented in this file.

**The Standard Structure (SemVer)**
- **Major (X.y.z)**: Incremented for incompatible API changes (breaking changes).
- **Minor (x.Y.z)**: Incremented for new functionality introduced in a backward-compatible manner.
- **Patch (x.y.Z)**: Incremented for backward-compatible bug fixes or minor patches.

## [v0.9.15] - 2026-04-26

### Added
- **Smart Separation Auto-Zoom**: Implemented dynamic zoom for split-screen and camera-switch modes. The system now adjusts individual panel zoom levels based on neighboring face proximity to ensure the main subject remains isolated and centered, even in crowded podcast shots.
- **Dynamic Vertical Tracking**: Added vertical alignment support across all podcast rendering pipelines. The camera now tracks both X and Y coordinates with configurable biasing via `--split-v-align`.
- **Exposed Split Controls**: Added manual controls for split-screen optimization including `--split-zoom`, `--split-v-align`, `--split-auto-zoom`, and `--split-max-zoom`.

### Fixed
- **Standardized Face Tracking Data**: Overhauled the internal tracking dictionary structure to use absolute center coordinates (`cx`, `cy`) and distance metrics (`dist`) consistently. This definitively resolves the `KeyError` bugs encountered during rapid layout transitions.
- **Robust Auto-Zoom Smoothing**: Integrated independent zoom smoothing for each panel, preventing "vignetting" or sudden jumps in framing during chaotic multi-person scenes.

---

## [v0.9.6] - 2026-04-22

### Fixed
- **Instant Split→Full Transition**: Eliminated the visible "ghost frame" where the split-screen layout persisted for a fraction of a second showing only 1 person in a 2-panel view before switching to full 9:16 crop. The `split→full` transition now fires on the same frame the face count drops to 1, bypassing both the majority-vote window and `MIN_HOLD` timer. The reverse direction (`full→split`) remains guarded to prevent flicker.

---

## [v0.9.8] - 2026-04-25

### Added
- **Configurable Source Download Quality**: Added `--source-height` CLI flag to control preferred source resolution at download time (e.g. `1080`, `1440`, `2160`).

### Changed
- **Default Source Strategy**: If `--source-height` is not provided, downloader now prefers the highest available source quality (`max`) to preserve sharpness after 16:9 → 9:16 crop.

---

## [v0.9.13] - 2026-04-26

### Added
- **Multi-Cloud AI Providers**: Integrated NVIDIA NIM API support (DeepSeek-V3/V4, Llama-3, etc) as an alternative to Gemini.
- **Guided JSON Generation**: Implemented strict JSON schema enforcement for NVIDIA models to guarantee 100% output accuracy.
- **AI-Agnostic Metadata Validator**: Added intelligent field aliasing (e.g., mapping 'peringkat' to 'rank') and auto-flattening for cross-model stability.
- **Failover Logic**: Automated failover from NVIDIA to Gemini if the provider is unavailable.

---

## [v0.9.12] - 2026-04-26

### Added
- **Auto-Bitrate Management**: Implemented resolution-aware bitrate selection (4M/8M/12M/20M) to optimize files for TikTok/Reels algorithms.
- **Subtle Sharpening Filter**: Added `--video-sharpen` flag to enhance clarity and texture detail in final renders.
- **NVENC Quality Tuning**: Switched to VBR (Variable Bitrate) mode with peak rate headroom for better motion handling.

---

## [v0.9.11] - 2026-04-26

### Fixed
- **Glitch Transition Resolution Bug**: Fixed an issue where the glitch teaser disappeared when using high-resolution rendering (2K/4K). The glitch video is now dynamically scaled to match the main video's output dimensions before concatenation.

---

## [v0.9.10] - 2026-04-26

### Added
- **Dynamic Render Resolution**: Added `--render-height` flag to allow custom output heights (e.g. `1080`, `1440`, `2160`, or `source`).
- **Smart Result Scaling**: Subtitles, layouts, and tracking windows now scale proportionally when rendering at high resolutions (2K/4K).
- **Download Verification**: Added real-time logging of the actual resolution and codec selected by `yt-dlp` to ensure quality targets are met.

### Fixed
- **AV1 Codec Incompatibility**: Explicitly excluded AV1 codec (`av01`) from download selection to prevent rendering crashes on platforms without hardware AV1 decoding (e.g., Colab T4). Prioritizes VP9 for high-res assets.

---

## [v0.9.9] - 2026-04-25

### Added
- **Video Quality Tuning Flags**: Added `--video-cq`, `--video-crf`, `--video-preset`, and `--video-scale-algo` for sharper output tuning on both normal and `--dynamic-split` rendering paths.

### Changed
- **Encoder Selection Integration**: `detect_video_encoder()` now respects runtime config tuning values while preserving backward-compatible defaults when flags are not provided.
- **Render Scaling Quality**: OpenCV resize operations in Studio rendering now use configurable interpolation (default `lanczos`) to preserve more detail after 16:9 → 9:16 crops.

---

## [v0.9.7] - 2026-04-23

### Changed
- **Studio Modularization (Non-Breaking)**: Refactored the monolithic `clipping/studio.py` into smaller internal modules under `clipping/studio/` (`core.py`, `helpers.py`, `ffmpeg_utils.py`) while keeping `clipping/studio.py` as the thin primary compatibility entry point.
- **Code Organization**: Moved internal rendering, subtitle, asset handling, and FFmpeg helper implementations into cohesive modules without changing CLI parameters, pipeline order, output naming, or public function contracts used by `runner.py`.

### Documentation
- **Docstring Coverage**: Added/expanded docstrings on public and non-trivial internal Studio functions to improve maintainability and onboarding without changing runtime behavior.

---

## [v0.9.5] - 2026-04-21

### Added
- **Manual Custom Hook Override**: Added `--hook-source` flag to use a specific external video file (.mp4) as the teaser hook for all generated clips.
- **Hook Tailoring**: Added `--hook-source-start` to define the exact starting point within the custom hook video.
- **Clean B-Roll Style**: Custom hooks now automatically skip subtitle rendering to preserve the original visual quality (ideal for quotes or cinematic assets).

---

## [v0.9.4] - 2026-04-21

### Added
- **Gemini JSON Caching**: The engine now automatically saves the raw Gemini AI analysis to `gemini_response.json` in the output directory.
- **Reproducibility Flag**: Added `--load-gemini-json` CLI argument to bypass AI generation and load results from a local file, enabling faster reproduction and debugging of specific clip layouts.

---

## [v0.9.3] - 2026-04-20

### Improved
- **Visual Merge Dashboard (Experimental)**: Added white bounding boxes, padded margins, and legend titles to the `--dev-mode-with-output-merge` view. 
- **Resolution Update**: Merged output resolution is now `2648x1220` to accommodate the new visual framing.
- **Note**: This enhanced styling is currently experimental. Users can revert to the standard side-by-side view from v0.9.2 if preferred.

---

## [v0.9.2] - 2026-04-20

### Added
- **Dual-Render Pipeline**: Added `--dev-mode-with-output` to generate both standard video and the dev dashboard simultaneously as two separate files.
- **Merged Dashboard**: Added `--dev-mode-with-output-merge` combining the standard output and the dev dashboard side-by-side into a single ultrawide video feed (2527x1080 resolution).

---

## [v0.9.1] - 2026-04-20

### Improved
- **Director's Console (Clear Window)**: Added a "spotlight" effect in Dev Mode where the active crop area is shown at 100% brightness while the background remains dimmed.
- **Dual Layout HUD**: Both Solo and Split crop boundaries are now visible simultaneously in Dev Mode, with high-contrast highlighting for the active layout.

---

## [v0.9.0] - 2026-04-20

### Added
- **Director's Console (Dev Mode)**: Implemented a professional debugging visualization for the `split-screen` / `dynamic-split` pipeline. Includes real-time HUD showing face counts, scene difference values, and layout switch hold timers.
- **Visual Stability Calibration**: Added yellow face-boxes and white crop outlines in dev-mode for precision tuning.

---

## [v0.8.9] - 2026-04-19

### Fixed
- **Instant Initial Layout**: Fixed a bug where even a solo shot would stay in split-screen for 2 seconds (default `switch-hold-duration`) at the very start of a clip. Now the system can make an instant correct layout decision at `t=0`.

---

## [v0.8.8] - 2026-04-19

### Improved
- **Enhanced Scene-Cut Sensitivity**: Lowered the visual difference threshold from 30 to 18 to better detect camera cuts in videos with dark themes/backgrounds.
- **Aggressive Face Merging**: Increased IoU overlap sensitivity to 0.2 to prevent near-face ghosting in close-up shots.

---

## [v0.8.6] - 2026-04-19

### Fixed
- **Instant Layout Reset on Scene Cuts**: Added a visual scene-cut detector that clears the layout stability history immediately when a camera cut occurs. This eliminates the "lag" where the video stayed in split-screen for ~1s after a cut to a solo shot.

---

## [v0.8.5] - 2026-04-19

### Added
- **Exposed Tracking Tuning Parameters**: Added new CLI flags for manual stability adjustment:
  - `--track-conf`: Confidence threshold for face detection.
  - `--track-smooth-window`: Frame window for majority-vote layout stability (includes time conversion guide in README).

---

## [v0.8.4] - 2026-04-19

### Added
- **Experimental Stability Filter for `--dynamic-split`**: Added three-layer protection against "ghost" split-screen triggers:
  - Higher YOLO confidence threshold (0.55).
  - IoU box merger to prevent duplicate face detections for the same person.
  - Majority-vote layout smoothing (12-frame window) to prevent flickering during dynamic transitions.

---

## [v0.8.3] - 2026-04-19

### Fixed
- **Off-Center Solo Crops**: Fixed the bug where speakers were off-center in full 9:16 solo mode within split-screen.
- **Aligned Tracking Parameters**: Split-screen tracking now uses identical default parameters (Deadzone, Smoothing, Step) as the standard hybrid mode for a consistent "feel".
- **Coordinate Overhaul**: Implemented absolute center-X tracking (`cx`) to support dynamic layout-independent centering.
- **Stability Tuning**: Improved logic to work better with `--face-detector yolo` for profile-heavy podcast scenarios.

---

## [v0.8.2] - 2026-04-19

### Added
- **Visual-Based Split Trigger** (`--split-trigger face`): Alternative way to decide when to split the screen.
  - Switches to Split layout when 2+ faces are detected.
  - Switches to Full layout when 1 face is detected.
  - **No HF_TOKEN required**: Can run without Speaker Diarization.
  - Highly efficient as it uses existing face tracking data.
- **Improved Resiliency**: Diarization helpers are now more robust when no speaker data is available.

---

## [v0.8.1] - 2026-04-19

### Added
- **Dynamic Split-Screen Mode** (`--dynamic-split`): New feature for split-screen layouts that automatically switches between full-screen and split-screen based on speaker activity. 
  - If 1 speaker is active, the system renders a full 9:16 crop on that speaker.
  - If 2+ speakers are active, the system renders the standard top-bottom split.
  - Includes a "hold duration" to prevent flickering during rapid dialogue.
- **Subtitle Tracking for Split-Screen**: Subtitles now correctly follow the speaker's face in both full and split layouts when using `--dynamic-split`.

---

## [v0.8.0] - 2026-04-19

### Fixed
- **Dev-Mode Subtitle Alignment**: Subtitles now dynamically follow the 9:16 tracking window in `--dev-mode`, ensuring they stay centered within the highlighted box rather than the full 16:9 frame.

---

## [v0.7.9] - 2026-04-18

### Added
- **Crosshair Tracking Lines** (`--track-lines`): New visualization feature that draws horizontal and vertical yellow lines extending from the face box to the boundaries of the 9:16 crop window. Automatically enabled in `--dev-mode`.

---

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
