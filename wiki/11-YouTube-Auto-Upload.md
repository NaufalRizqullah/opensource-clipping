# 📤 YouTube Auto-Upload

OpenSource Clipping includes a standalone YouTube auto-uploader with scheduling support, allowing you to automatically publish generated clips with full metadata.

---

## Prerequisites

1. A Google Cloud project with the **YouTube Data API v3** enabled
2. OAuth 2.0 credentials (`youtube_token.json`) configured for your YouTube channel
3. Generated clips in the `outputs/` directory with a `render_manifest.json`

> **First time?** Follow the complete **[YouTube API Setup Guide](YouTube-API-Setup-Guide)** for step-by-step instructions with screenshots on creating your Google Cloud project, OAuth credentials, and generating the token.

---

## Setup

### 1. Generate YouTube Token

Run the token generator to authenticate your YouTube account:

```bash
python youtube_uploader/generate_youtube_token.py
```

This will open a browser window for Google OAuth. After authorizing, a `youtube_token.json` file will be created.

### 2. Place the Token

Move the generated token file into the `.credentials/` directory:

```bash
mkdir -p .credentials
mv youtube_token.json .credentials/
```

---

## Usage

### Basic Upload

After rendering clips, run from the repository root:

```bash
python scripts/run_upload.py
```

This will:
1. Read `outputs/render_manifest.json` for clip metadata
2. Upload each clip to YouTube with AI-generated titles, descriptions, and tags
3. Schedule uploads at 24-hour intervals by default (Asia/Makassar)

The former safety feature (daily limits and manual approval) has been removed.
The old `--safety-config` and `--no-approval` options stop with an error.

### Custom Scheduling

```bash
# 12-hour intervals with Jakarta timezone
python scripts/run_upload.py --interval-hours 12 --tz-name "Asia/Jakarta"

# Test with only the first video
python scripts/run_upload.py --test-mode
```

### All Options

```bash
python scripts/run_upload.py --help
```

---

## Metadata

The uploader uses the AI-generated metadata from `metadata_preview.json`:

- **Title** — YouTube-optimized title
- **Description** — SEO-friendly description with relevant context
- **Tags** — Keyword tags for discoverability
- **TikTok Caption** — Also generated for cross-platform publishing

---

## Rescheduling

Need to reschedule already-uploaded videos?

```bash
python youtube_uploader/reschedule_youtube.py
```

This tool can reschedule videos that are still in `scheduled` or `private` status.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Token expired | Re-run `generate_youtube_token.py` |
| API quota exceeded | Wait 24 hours or use a different API project |
| Rate limited (429) | Increase interval between uploads |
| Upload fails silently | Check `outputs/render_manifest.json` for valid file paths |

---

## See Also

- [Getting Started](Getting-Started) — Initial setup
- [CLI Reference](CLI-Reference) — Pipeline options
