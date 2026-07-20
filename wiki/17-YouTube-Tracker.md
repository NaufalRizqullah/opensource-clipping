# 📺 YouTube Tracker

OpenSource Clipping includes a lightweight, local **YouTube Playlist Snapshot Tracker**. It helps you track which YouTube videos you've already used for clipping, ensuring you don't process duplicate content across different playlists or manual picks.

> **Snapshot Behavior:** This is a local tracker, not a live background sync. Data is only fetched from YouTube when you explicitly add a playlist or click the "Pull Again" button. All data is securely stored locally in a SQLite database.

---

## Prerequisites

1. Python 3.10+
2. `yt-dlp` installed (comes automatically with the project requirements)

---

## Launching the Server

The tracker features a clean, dark-mode web dashboard. To launch it, run:

```bash
python youtube_tracker/server.py
```

Then, open **http://127.0.0.1:8765** in your browser.

---

## Key Features

### 1. Track Playlists
- Go to the Dashboard and paste any YouTube playlist URL.
- Click **Add Playlist**. The tracker fetches metadata for every video and stores it locally.
- **Refresh / Pull Again:** Clicking the "🔄 Pull Again" button re-fetches the playlist. New videos are added, missing ones are marked (never deleted), and your local statuses remain intact.

### 2. Manual Video Tracking
- Switch to the **+ Manual Video** tab on the Dashboard.
- Paste a single YouTube video URL.
- It will be added directly to the "Manual Videos" virtual playlist.

### 3. Global Status Management
Each video has a global status across the entire tracker (even if it appears in multiple playlists):
- **Unused** — Not yet processed.
- **Candidate** — Marked for future clipping.
- **Used** — Successfully clipped and processed.
- **Skipped** — Intentionally bypassed.

Click the status pill or ✏️ button to edit. You can add extra metadata such as:
- The generated Clip Title
- The Used Date
- Local Output Path
- Published URL
- Custom Notes

### 4. Search and Channels
- **Global Search:** Search instantly across video titles, channel names, custom notes, and clip titles.
- **Channels View:** Browse your database grouped by the original YouTube channel. Easily filter by "Not Used Yet" to find fresh, unclipped content from a specific creator.
- **Duplicates View:** Automatically identify videos that appear in multiple playlists across your database.

### 5. Instant Clipping Command
Ready to clip a video? Click the 📋 copy button on any video card to instantly copy the processing command:
```bash
python main.py --url "https://www.youtube.com/watch?v=..."
```
*(You can configure your default preferred flags in the Settings panel)*

### 6. Exporting
You can back up or export your tracking data from the Settings page:
- **Export JSON** — Full metadata dump.
- **Export CSV** — Simple spreadsheet of videos and their statuses.

---

## Database Schema Overview

The SQLite database is located at `youtube_tracker/youtube_tracker.sqlite3`.

| Table | Purpose |
|-------|---------|
| `channels` | Original YouTube channels |
| `sources` | Playlists, manual source, and channel sources |
| `videos` | Unique videos (deduplicated by `youtube_video_id`) |
| `source_videos` | Link table mapping videos to playlists |
| `video_status` | The global status per video (unused/candidate/used/skipped) |
| `pull_runs` | The sync history of playlist refreshes |
| `pull_run_videos` | The exact list of videos seen during a specific pull |
| `settings` | Key-value store for UI settings and default flags |

> **Resetting:** If you need to completely restart, you can delete `youtube_tracker/youtube_tracker.sqlite3`. A new empty database will be created the next time you start the server.
