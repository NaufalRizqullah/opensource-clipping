# 🎬 OpenSource Clipping v0.3.0

**Ultimate AI Auto-Clipper & Teaser Generator** — an open-source content factory that transforms long-form videos into cinematic short-form highlights with hook teasers, karaoke subtitles, and auto-thumbnails.

> 🇮🇩 [Baca dalam Bahasa Indonesia](README_ID.md)

---

## ✨ Features

| Feature | Description |
|---|---|
| **AI Transcriber** | Word-level transcription using **Faster-Whisper** (large-v3) |
| **AI Content Curator** | **Google Gemini** analyzes context, picks the most viral moments, and generates metadata |
| **Smart Auto-Framing** | **MediaPipe** face-tracking with Smooth Pan, Deadzone & anti-jitter algorithms |
| **Cinematic Teaser Hook** | 3-second hook with dark overlay, cinematic bars, and **TV Glitch** transition |
| **Karaoke Subtitles** | Word-by-word highlighted `.ASS` subtitles (Alex Hormozi / Veed style) |
| **Kinetic Typography** | AI-driven word emphasis with bounce/stagger animations & dual-font system |
| **B-Roll Integration** | Auto-fetches contextual stock footage from **Pexels** with crossfade & Ken Burns |
| **Auto-BGM & Ducking** | AI-matched background music from Pixabay with sidechain ducking |
| **Auto-Thumbnail** | Frame extraction with dark overlay and large title text |
| **Cross-Platform Metadata** | YouTube title/description/tags + TikTok caption — all in English |

## 📋 Prerequisites

- **Python** 3.10+
- **FFmpeg** installed and available in PATH
- **CUDA GPU** recommended (for Whisper; CPU fallback available)
- **Google Gemini API Key** ([get one here](https://aistudio.google.com/apikey))
- **Pexels API Key** (optional, for B-roll — [get one here](https://www.pexels.com/api/))

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-username/opensource-clipping.git
cd opensource-clipping

# 2. Install dependencies (pick one)
pip install -r requirements.txt          # pip / Colab
# uv sync                               # or use uv (reads pyproject.toml)

# 3. Set up API keys
cp .env.sample .env
# Edit .env and add your GOOGLE_API_KEY

# 4. Run with defaults
python main.py
# uv run main.py                        # if using uv

# 5. Or customize
python main.py --url "https://youtube.com/watch?v=VIDEO_ID" --clips 5 --ratio 16:9
```

## ⚙️ CLI Options

```
python main.py --help
```

| Argument | Default | Description |
|---|---|---|
| `--url`, `-u` | *(preset URL)* | YouTube video URL to process |
| `--clips`, `-n` | `7` | Number of highlight clips to generate |
| `--ratio`, `-r` | `9:16` | Output aspect ratio (`9:16` or `16:9`) |
| `--words-per-sub` | `5` | Max words per karaoke subtitle group |
| `--hook-duration` | `3` | Hook teaser duration (seconds) |
| `--font-style` | `HORMOZI` | Font preset (`DEFAULT`, `STORYTELLER`, `HORMOZI`, `CINEMATIC`) |
| `--no-broll` | — | Disable B-roll footage |
| `--no-hook` | — | Disable hook glitch teaser |
| `--no-bgm` | — | Disable background music |
| `--no-karaoke` | — | Use clean text instead of karaoke highlight |
| `--advanced-text` | `False` | Enable kinetic typography (word scaling & animation) |
| `--whisper-model` | `large-v3` | Whisper model size |
| `--whisper-device` | `cuda` | Whisper device (`cuda`, `cpu`, `auto`) |
| `--gemini-model` | `gemini-3-flash-preview` | Gemini model name |

## 📂 Project Structure

```
opensource-clipping/
├── main.py                  # CLI entry point
├── pyproject.toml           # Dependencies & metadata
├── .env.sample              # API key template
├── .gitignore
├── README.md                # English docs
├── README_ID.md             # Indonesian docs
└── clipping/
    ├── __init__.py
    ├── config.py            # Master configuration & argparse
    ├── engine.py            # Download → Transcribe → Gemini AI
    ├── metadata.py          # QA metadata normalization
    ├── studio.py            # Video render engine (face-track, subs, B-roll, BGM)
    └── runner.py            # Pipeline orchestrator
```

## 🔄 Pipeline Flow

```mermaid
graph LR
    A[YouTube URL] --> B[Download Video]
    B --> C[Whisper Transcription]
    C --> D[Gemini AI Analysis]
    D --> E[Metadata QA]
    E --> F[Render Loop]
    F --> G[Face-Track Crop]
    F --> H[B-Roll + BGM]
    F --> I[ASS Subtitles]
    F --> J[Hook + Glitch]
    G & H & I & J --> K[Final MP4 + Thumbnail]
```

## 📤 Output

For each clip, the pipeline generates:

| File | Description |
|---|---|
| `highlight_rank_N_ready.mp4` | Final rendered clip with subtitles, B-roll, BGM |
| `thumbnail_rank_N.jpg` | Auto-generated thumbnail with title text |
| `render_manifest.json` | Manifest with metadata for all clips |
| `metadata_preview.json` | Gemini-generated metadata (titles, tags, captions) |

## 🎵 Font Styles

| Style | Main Font | Emphasis Font | Best For |
|---|---|---|---|
| `HORMOZI` | Montserrat | Anton | Business / motivational |
| `STORYTELLER` | Inter | Lora | Narrative / storytelling |
| `CINEMATIC` | Roboto | Bebas Neue | Film / dramatic |
| `DEFAULT` | Montserrat Black | Montserrat Medium | General purpose |

## 📺 YouTube Upload (Notebook Only)

Cell 5 of the original notebook (`OpenSource_Clipping.ipynb`) contains a YouTube uploader with scheduling support. This is **not** included in the CLI project — run it separately from the notebook after rendering is complete.

## 📄 License

Open source. Feel free to use, modify, and distribute.
