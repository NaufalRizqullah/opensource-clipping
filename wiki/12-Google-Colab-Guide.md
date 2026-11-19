# ☁️ Google Colab Guide

If you don't have a local GPU, Google Colab provides free access to NVIDIA T4 GPUs — perfect for running the full pipeline.

---

## Quick Start (3 Cells)

For the browser-based **Studio** on a Python 3.13 kernel, use the latest [Studio server notebook](../notebooks/kaggle-studio-server.ipynb). Its **Prepare Python 3.12** cell creates/reuses a separate interpreter before installing the application, without changing the kernel. It also includes optional local vLLM startup through `START_LOCAL_VLLM=True`; leave that off for Gemini/NVIDIA or an existing vLLM endpoint. See the [Studio guide](../docs/KAGGLE_STUDIO.md). The CLI examples below and other CLI notebooks do not include this bootstrap and still require a compatible application Python environment.

Open a new [Google Colab](https://colab.research.google.com/) notebook and set the Runtime to **T4 GPU**.

### Cell 1: Setup & Clone

```python
!rm -rf ./* ./.*
!git clone https://github.com/NaufalRizqullah/opensource-clipping.git .
!pip install -r requirements.txt
```

### Cell 2: Setup API Keys

```python
import os
from pathlib import Path
from google.colab import userdata

# Store your keys in Colab Secrets first (key icon in sidebar)
GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")

env_text = f"GOOGLE_API_KEY={GOOGLE_API_KEY}\n"
Path(".env").write_text(env_text, encoding="utf-8")
```

> **Tip:** Click the 🔑 (Secrets) icon in the Colab sidebar to add your API keys securely.

### Cell 3: Run

```python
URL_YOUTUBE = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
JUMLAH_CLIP = 7
RASIO = "9:16"
FONT_STYLE = "DEFAULT"
GEMINI_MODEL = "gemini-3-flash-preview"

!python main.py \
  --url "{URL_YOUTUBE}" \
  --clips {JUMLAH_CLIP} \
  --ratio "{RASIO}" \
  --font-style "{FONT_STYLE}" \
  --hook-duration 3 \
  --words-per-sub 5 \
  --gemini-model "{GEMINI_MODEL}" \
  --no-bgm
```

---

## Recommended Configurations

### Standard Clipping (Best Accuracy)

```python
URL_YOUTUBE = "https://www.youtube.com/watch?v=VIDEO_ID"
JUMLAH_CLIP = 7
RASIO = "9:16"
FONT_STYLE = "DEFAULT"
GEMINI_MODEL = "gemini-2.0-flash"

!python main.py \
  --url "{URL_YOUTUBE}" \
  --clips {JUMLAH_CLIP} \
  --ratio "{RASIO}" \
  --font-style "{FONT_STYLE}" \
  --hook-duration 3 \
  --words-per-sub 5 \
  --face-detector yolo \
  --gemini-model "{GEMINI_MODEL}" \
  --no-bgm \
  --no-subs \
  --no-broll \
  --use-dlp-subs
```

### Split-Screen Podcast

```python
URL_YOUTUBE = "https://www.youtube.com/watch?v=PODCAST_ID"
JUMLAH_CLIP = 3
RASIO = "9:16"
FONT_STYLE = "DEFAULT"
GEMINI_MODEL = "gemini-2.0-flash"

!python main.py \
  --url "{URL_YOUTUBE}" \
  --clips {JUMLAH_CLIP} \
  --ratio "{RASIO}" \
  --font-style "{FONT_STYLE}" \
  --hook-duration 3 \
  --words-per-sub 5 \
  --gemini-model "{GEMINI_MODEL}" \
  --no-bgm \
  --no-subs \
  --no-broll \
  --split-screen \
  --dynamic-split \
  --split-trigger face \
  --face-detector yolo \
  --use-dlp-subs
```

---

## Kaggle Compatibility

When running on **Kaggle** (which has limited T4 configurations), use `float32` for Whisper:

```python
WHISPER_COMPUTE_TYPE = "float32"

!python main.py \
  --url "{URL_YOUTUBE}" \
  --clips 5 \
  --whisper-compute-type "{WHISPER_COMPUTE_TYPE}" \
  --no-bgm
```

---

## Adding Optional API Keys

To enable more features, add additional secrets:

```python
from google.colab import userdata

GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
PEXELS_API_KEY = userdata.get("PEXELS_API_KEY")    # For B-roll
HF_TOKEN = userdata.get("HF_TOKEN")                # For podcast modes

env_text = f"""GOOGLE_API_KEY={GOOGLE_API_KEY}
PEXELS_API_KEY={PEXELS_API_KEY}
HF_TOKEN={HF_TOKEN}
"""
Path(".env").write_text(env_text, encoding="utf-8")
```

---

## Downloading Results

After rendering, download the generated clips:

```python
# Download all clips from outputs
from google.colab import files
import glob

for f in glob.glob("outputs/**/highlight_*_ready.mp4", recursive=True):
    files.download(f)
```

Or zip everything:

```python
!zip -r results.zip outputs/
files.download("results.zip")
```

---

## Tips & Notes

- **Runtime Timeout**: Free Colab sessions disconnect after ~90 minutes of inactivity. Keep the browser tab active.
- **GPU Memory**: If you run out of GPU memory, try using `--whisper-model medium` instead of `large-v3`.
- **Persistent Storage**: Use Google Drive to save outputs across sessions:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  # Then copy outputs to Drive
  !cp -r outputs/ /content/drive/MyDrive/clipping_results/
  ```
- **Ready-to-use Notebook**: Check `notebooks/Lib_OpenSource_Clipping.ipynb` in the repo for a pre-configured template.

---

## See Also

- [Getting Started](Getting-Started) — Local installation
- [Video Quality & Rendering](Video-Quality-and-Rendering) — Quality tuning for Colab
