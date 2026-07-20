"""
youtube_tracker/youtube_fetcher.py
YouTube metadata fetcher using yt-dlp Python API.
Never downloads audio/video — metadata only.
"""

import re
import json
import time
import sys

MAX_RETRIES = 2
RETRY_DELAY_BASE = 2  # seconds, exponential backoff: 2, 4, 8


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

_VIDEO_PATTERNS = [
    re.compile(r'(?:youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/shorts/)([A-Za-z0-9_-]{11})'),
]

_PLAYLIST_PATTERNS = [
    re.compile(r'(?:youtube\.com/playlist\?.*list=)([A-Za-z0-9_-]+)'),
    re.compile(r'(?:youtube\.com/watch\?.*list=)([A-Za-z0-9_-]+)'),
]


def extract_video_id(url_or_id):
    """Extract YouTube video ID from a URL or return the ID if already bare."""
    if not url_or_id:
        return None
    url_or_id = url_or_id.strip()
    # Bare 11-char ID
    if re.match(r'^[A-Za-z0-9_-]{11}$', url_or_id):
        return url_or_id
    for pat in _VIDEO_PATTERNS:
        m = pat.search(url_or_id)
        if m:
            return m.group(1)
    return None


def extract_playlist_id(url_or_id):
    """Extract YouTube playlist ID from a URL or return the ID if already bare."""
    if not url_or_id:
        return None
    url_or_id = url_or_id.strip()
    # Bare playlist ID (starts with PL, UU, OL, LL, FL, etc.)
    if re.match(r'^[A-Za-z0-9_-]{10,}$', url_or_id) and '/' not in url_or_id:
        return url_or_id
    for pat in _PLAYLIST_PATTERNS:
        m = pat.search(url_or_id)
        if m:
            return m.group(1)
    return None


def validate_youtube_url(url):
    """Check if a URL is a valid YouTube URL. Returns 'video', 'playlist', or None."""
    if not url:
        return None
    if extract_playlist_id(url):
        return "playlist"
    if extract_video_id(url):
        return "video"
    return None

def fetch_channel_thumbnail(channel_url):
    """Scrape the channel profile picture from a YouTube channel URL via og:image."""
    if not channel_url:
        return None
    try:
        import urllib.request
        req = urllib.request.Request(channel_url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, timeout=5).read().decode('utf-8', errors='ignore')
        m = re.search(r'<meta property="og:image" content="([^"]+)"', html)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Metadata normalizers
# ---------------------------------------------------------------------------

def _safe_int(val):
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_str(val):
    if val is None:
        return None
    return str(val).strip() if val else None


def _best_thumbnail(info):
    """Pick the best thumbnail URL from yt-dlp info dict."""
    thumb = info.get("thumbnail")
    if thumb:
        return thumb
    thumbs = info.get("thumbnails")
    if thumbs and isinstance(thumbs, list):
        # Prefer higher resolution
        for t in reversed(thumbs):
            if isinstance(t, dict) and t.get("url"):
                return t["url"]
    return None


def normalize_video_metadata(info):
    """Normalize yt-dlp video info dict to our standard format."""
    if not info:
        return None

    video_id = info.get("id") or info.get("display_id")
    if not video_id:
        return None

    channel_id = info.get("channel_id") or info.get("uploader_id")
    channel_name = info.get("channel") or info.get("uploader")
    channel_url = info.get("channel_url") or info.get("uploader_url")

    # Handle for channel
    handle = info.get("uploader_id")
    if handle and not handle.startswith("@"):
        handle = None  # Not a handle

    url = info.get("webpage_url") or info.get("url") or f"https://www.youtube.com/watch?v={video_id}"

    # Duration
    duration = _safe_int(info.get("duration"))

    # Upload date
    upload_date = _safe_str(info.get("upload_date"))
    if upload_date and len(upload_date) == 8:
        upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"

    # Build a lightweight raw_json (exclude large fields)
    raw = {}
    for k in ("id", "title", "description", "duration", "upload_date",
              "channel", "channel_id", "channel_url", "uploader", "uploader_id",
              "view_count", "like_count", "comment_count", "categories", "tags",
              "webpage_url", "availability", "live_status"):
        if k in info:
            raw[k] = info[k]

    return {
        "video_id": video_id,
        "title": _safe_str(info.get("title")) or "Untitled",
        "url": url,
        "thumbnail": _best_thumbnail(info),
        "duration_seconds": duration,
        "upload_date": upload_date,
        "published_at": upload_date,
        "channel_id": channel_id,
        "channel_name": channel_name,
        "channel_url": channel_url,
        "handle": handle,
        "description": _safe_str(info.get("description")),
        "raw_json": raw,
    }


def normalize_playlist_metadata(info):
    """Normalize yt-dlp playlist info dict to our standard format."""
    if not info:
        return None

    playlist_id = info.get("id")
    channel_id = info.get("channel_id") or info.get("uploader_id")
    channel_name = info.get("channel") or info.get("uploader")
    channel_url = info.get("channel_url") or info.get("uploader_url")

    raw = {}
    for k in ("id", "title", "description", "channel", "channel_id",
              "channel_url", "uploader", "uploader_id", "webpage_url",
              "playlist_count", "modified_date"):
        if k in info:
            raw[k] = info[k]

    return {
        "playlist_id": playlist_id,
        "title": _safe_str(info.get("title")) or "Untitled Playlist",
        "url": info.get("webpage_url") or f"https://www.youtube.com/playlist?list={playlist_id}",
        "thumbnail": _best_thumbnail(info),
        "channel_id": channel_id,
        "channel_name": channel_name,
        "channel_url": channel_url,
        "raw_json": raw,
    }


# ---------------------------------------------------------------------------
# YouTubeFetcher class
# ---------------------------------------------------------------------------

class YouTubeFetcher:
    """Fetches YouTube metadata using yt-dlp. Never downloads media."""

    def __init__(self):
        self._ydl_module = None

    def _get_ydl(self):
        """Lazy-import yt_dlp."""
        if self._ydl_module is None:
            try:
                import yt_dlp
                self._ydl_module = yt_dlp
            except ImportError:
                raise RuntimeError(
                    "yt-dlp is not installed. Install it with: pip install yt-dlp"
                )
        return self._ydl_module

    def fetch_video(self, url_or_id, _retries=MAX_RETRIES):
        """
        Fetch metadata for a single video with retry logic.
        Returns normalized video dict or raises an error.
        """
        yt_dlp = self._get_ydl()

        vid_id = extract_video_id(url_or_id)
        url = f"https://www.youtube.com/watch?v={vid_id}" if vid_id else url_or_id

        opts = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "ignoreerrors": False,
        }

        last_error = None
        for attempt in range(1, _retries + 1):
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=False)

                if not info:
                    raise ValueError(f"Could not fetch metadata for: {url_or_id}")

                normalized = normalize_video_metadata(info)
                if not normalized:
                    raise ValueError(f"Could not normalize metadata for: {url_or_id}")

                return normalized

            except Exception as e:
                last_error = e
                if attempt < _retries:
                    delay = RETRY_DELAY_BASE ** attempt
                    print(f"  [Retry {attempt}/{_retries}] Failed to fetch {url_or_id}: {e} — retrying in {delay}s...",
                          file=sys.stderr)
                    time.sleep(delay)

        raise last_error

    def fetch_playlist_generator(self, url_or_id, should_fetch_callback=None):
        """
        Fetch playlist metadata and yield it, then fetch ALL video entries and yield them.
        Uses playlistend=-1 to disable the default 100-item limit.
        Returns a generator yielding dicts:
          {"type": "playlist", "playlist": {...}, "total_entries": int}
          {"type": "video", "video": {...}, "position": int}
        """
        yt_dlp = self._get_ydl()

        pl_id = extract_playlist_id(url_or_id)
        url = f"https://www.youtube.com/playlist?list={pl_id}" if pl_id else url_or_id

        # Step 1: Fetch playlist with flat extraction to get ALL video IDs
        opts_flat = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": "in_playlist",
            "ignoreerrors": True,
            "playlist_items": "1:",
            "lazy_playlist": False,
        }
        

        print(f"  [Playlist] Fetching playlist index: {url}", file=sys.stderr)

        with yt_dlp.YoutubeDL(opts_flat) as ydl:
            pl_info = ydl.extract_info(url, download=False)

        if not pl_info:
            raise ValueError(f"Could not fetch playlist: {url_or_id}")

        playlist_meta = normalize_playlist_metadata(pl_info)

        # Materialize entries — yt-dlp sometimes returns a lazy generator
        raw_entries = pl_info.get("entries") or []
        if hasattr(raw_entries, '__next__') or hasattr(raw_entries, '__iter__'):
            entries = list(raw_entries)
        else:
            entries = raw_entries

        total_entries = len(entries)
        print(f"  [Playlist] Found {total_entries} entries in playlist", file=sys.stderr)

        yield {
            "type": "playlist",
            "playlist": playlist_meta,
            "total_entries": total_entries,
        }

        # Step 2: For each entry, fetch full video metadata with retry
        opts_video = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "ignoreerrors": False,
        }

        for i, entry in enumerate(entries):
            if not entry:
                continue

            entry_id = entry.get("id") or entry.get("url")
            if not entry_id:
                continue

            # Skip full fetch if we already have it (fast mode)
            if should_fetch_callback and not should_fetch_callback(entry_id):
                yield {
                    "type": "shallow_video",
                    "video_id": entry_id,
                    "position": i
                }
                continue

            # Progress log
            progress = f"[{i + 1}/{total_entries}]"
            print(f"  {progress} Fetching video: {entry_id}", file=sys.stderr)

            # Retry loop for this video
            success = False
            last_error = None

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    video_url = f"https://www.youtube.com/watch?v={entry_id}"
                    with yt_dlp.YoutubeDL(opts_video) as ydl:
                        video_info = ydl.extract_info(video_url, download=False)

                    if video_info:
                        normalized = normalize_video_metadata(video_info)
                        if normalized:
                            normalized["_playlist_position"] = i
                            yield {
                                "type": "video",
                                "video": normalized,
                                "position": i
                            }
                            success = True
                            break
                    else:
                        raise ValueError(f"No info returned for {entry_id}")

                except Exception as e:
                    last_error = e
                    if attempt < MAX_RETRIES:
                        delay = RETRY_DELAY_BASE ** attempt
                        print(f"  [Retry {attempt}/{MAX_RETRIES}] Failed to fetch {entry_id}: {e} — retrying in {delay}s...",
                              file=sys.stderr)
                        time.sleep(delay)

            if not success:
                print(f"  [Error] Exhausted retries for {entry_id}: {last_error}", file=sys.stderr)
