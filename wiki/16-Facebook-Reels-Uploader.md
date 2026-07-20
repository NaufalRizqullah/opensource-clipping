# 📱 Facebook Reels Uploader

OpenSource Clipping includes a standalone Facebook Pages Reels auto-uploader and scheduler, enabling you to publish generated clips directly to your Facebook Page via the Meta Graph API.

---

## Prerequisites

To use the Facebook Uploader, you need:
1. A **Facebook Page** where you have admin/editor access.
2. A **Page Access Token** with `pages_manage_posts`, `pages_read_engagement`, and `pages_show_list` permissions.
3. The **Page ID** of your Facebook Page.
4. Generated clips in the `outputs/` directory with a `render_manifest.json`.

---

## Setup

Set up your environment variables in the `.env` file at the root of the repository:

```env
META_PAGE_ID="your_facebook_page_id"
META_PAGE_ACCESS_TOKEN="your_page_access_token_here"
META_GRAPH_VERSION="v25.0" # Optional, defaults to v25.0
```

---

## Usage

### Basic Upload

After rendering your clips, run the Facebook uploader CLI tool:

```bash
python run_fb_upload.py
```

This will:
1. Validate your Page Access Token.
2. Read the `outputs/render_manifest.json` for pending clips.
3. Automatically schedule the clips at 5-hour intervals (by default).
4. Synchronize the `fb_upload_status` in `outputs/render_manifest_fb_uploaded.json`.

### Custom Scheduling and Modes

You can adjust the interval between scheduled posts or test the uploader with a single video:

```bash
# 3-hour intervals with Jakarta timezone
python run_fb_upload.py --interval-hours 3 --tz-name "Asia/Jakarta"

# Test mode: only upload the FIRST item in the manifest
python run_fb_upload.py --test-mode
```

### All Options

```bash
python run_fb_upload.py --help
```

---

## Rate Limits & Safety Rules

The uploader automatically enforces Meta's safety guidelines and API limits to prevent account restrictions:

- **24-Hour Rolling Quota**: Meta limits Reels to a maximum of **30 videos per 24-hour window**. The script safely counts your recent uploads and stops if you reach this limit.
- **Minimum Schedule Time**: Reels must be scheduled at least **10 minutes** into the future. The uploader automatically bumps schedules that are too close.
- **Maximum Schedule Time**: Reels can be scheduled up to **29 days** in advance. If the queue goes beyond 29 days, the batch will safely stop.

---

## API Upload Flow Details

For transparency and troubleshooting, the uploader follows Meta's 4-step upload flow:
1. **Initialize Session**: `upload_phase=start` is sent to get a `video_id` and an upload URL.
2. **Binary Upload**: The physical `.mp4` file is securely transmitted via OAuth stream.
3. **Finish & Metadata**: `upload_phase=finish` is sent with the title, description, and `scheduled_publish_time`.
4. **Status Polling**: The tool actively polls the Meta API until `publishing_phase.status` is `complete` (whether `published` or `scheduled`), ensuring your video was accepted without errors.

---

## Status Synchronization

If you stop the script halfway or an upload gets stuck in processing, running the tool again will safely resume operations. It checks existing `fb_upload_status` fields (like `uploaded` or `pending`) and asks Meta for the current state (e.g., updating it to `published` or `scheduled`).

---

## See Also

- [YouTube Auto-Upload](YouTube-Auto-Upload) — Cross-post to YouTube Shorts
- [CLI Reference](CLI-Reference) — Pipeline options
