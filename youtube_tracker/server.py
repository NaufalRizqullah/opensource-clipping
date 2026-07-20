"""
youtube_tracker/server.py
Lightweight HTTP server for the YouTube Tracker.
Uses only Python standard library (http.server).
Serves static files and JSON API endpoints.
"""

import json
import os
import re
import sys
import mimetypes
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Ensure package-level imports work when running as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db
from youtube_fetcher import YouTubeFetcher

HOST = "127.0.0.1"
PORT = 8765
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

fetcher = YouTubeFetcher()


# ---------------------------------------------------------------------------
# Pull orchestration
# ---------------------------------------------------------------------------

import threading

def _run_async_pull(url, source_id, pull_run_id):
    """Background worker thread for fetching a playlist."""
    stats = {
        "total_entries": 0,
        "videos_found": 0,
        "videos_added": 0,
        "videos_updated": 0,
        "videos_already_exists": 0,
        "videos_missing_from_latest_pull": 0,
        "fetch_errors": 0,
    }
    errors = []

    try:
        gen = fetcher.fetch_playlist_generator(url)
        first = next(gen)
        
        pl_meta = first["playlist"]
        total_entries = first["total_entries"]
        stats["total_entries"] = total_entries

        # Upsert owner channel
        owner_ch_id = None
        if pl_meta.get("channel_id"):
            owner_ch_id = db.upsert_channel({
                "channel_id": pl_meta["channel_id"],
                "channel_name": pl_meta.get("channel_name"),
                "channel_url": pl_meta.get("channel_url"),
            })

        # Upsert source
        sid = db.upsert_source({
            "source_key": f"playlist:{pl_meta['playlist_id']}",
            "source_type": "playlist",
            "playlist_id": pl_meta["playlist_id"],
            "title": pl_meta["title"],
            "url": pl_meta["url"],
            "thumbnail": pl_meta.get("thumbnail"),
            "owner_channel_db_id": owner_ch_id,
            "owner_channel_name": pl_meta.get("channel_name"),
            "owner_channel_url": pl_meta.get("channel_url"),
            "raw_json": pl_meta.get("raw_json"),
        })
        
        source_id = sid

        for item in gen:
            if item["type"] == "video":
                vmeta = item["video"]
                i = item["position"]
                
                stats["videos_found"] += 1
                db.update_pull_run_progress(pull_run_id, stats["videos_found"], total_entries)
                
                ch_db_id = None
                if vmeta.get("channel_id"):
                    ch_db_id = db.upsert_channel({
                        "channel_id": vmeta["channel_id"],
                        "channel_name": vmeta.get("channel_name"),
                        "channel_url": vmeta.get("channel_url"),
                        "handle": vmeta.get("handle"),
                    })

                vmeta["channel_db_id"] = ch_db_id
                video_db_id, was_new = db.upsert_video(vmeta)
                if was_new:
                    stats["videos_added"] += 1
                else:
                    stats["videos_already_exists"] += 1

                db.ensure_video_status(video_db_id)
                db.record_pull_run_video(pull_run_id, video_db_id, position=i)
                db.link_source_video(
                    source_id, video_db_id, position=i,
                    metadata_source="playlist_pull",
                    pull_run_id=pull_run_id,
                )

        missing_count = db.mark_missing_videos_for_source(source_id, pull_run_id)
        stats["videos_missing_from_latest_pull"] = missing_count

        db.update_source_last_pulled(source_id)
        stats["fetch_errors"] = len(errors)
        db.finish_pull_run(pull_run_id, status="success", stats=stats)

    except Exception as e:
        import traceback
        traceback.print_exc()
        db.finish_pull_run(pull_run_id, status="failed", error_message=str(e))

def _start_async_pull(url, source_id=None):
    # If source doesn't exist yet, we create a dummy source or just start pull run on source 0
    # Wait, pull_runs requires a valid source_id. Let's create a placeholder source if it's new.
    if not source_id:
        source_id = db.upsert_source({
            "source_type": "playlist",
            "source_key": f"temp:{url}",
            "title": "Loading Playlist...",
            "url": url,
        })
    pull_run_id = db.create_pull_run(source_id)
    t = threading.Thread(target=_run_async_pull, args=(url, source_id, pull_run_id), daemon=True)
    t.start()
    return {"ok": True, "source_id": source_id, "pull_run_id": pull_run_id, "status": "running"}


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class TrackerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for API + static files."""

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[{self.log_date_time_string()}] {format % args}")

    # -- Helpers ---

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status, message):
        self._send_json({"error": message}, status)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw)

    def _parse_path(self):
        parsed = urlparse(self.path)
        return parsed.path, parse_qs(parsed.query)

    # -- Static file serving ---

    def _serve_static(self, filepath):
        if not os.path.isfile(filepath):
            self.send_error(404, "File not found")
            return

        mime, _ = mimetypes.guess_type(filepath)
        if mime is None:
            mime = "application/octet-stream"

        with open(filepath, "rb") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    # -- Routing ---

    def _route_get(self, path, query):
        """Route GET requests."""

        # Static files
        if path == "/" or path == "":
            return self._serve_static(os.path.join(STATIC_DIR, "index.html"))

        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            return self._serve_static(os.path.join(STATIC_DIR, rel))

        # API routes
        if path == "/api/health":
            return self._send_json({"status": "ok", "server": "youtube_tracker"})

        if path == "/api/sources":
            sources = db.get_sources_with_stats()
            stats = db.get_global_stats()
            return self._send_json({"sources": sources, "stats": stats})

        # GET /api/sources/:id
        m = re.match(r'^/api/sources/(\d+)$', path)
        if m:
            source = db.get_source(int(m.group(1)))
            if not source:
                return self._send_error_json(404, "Source not found")
            return self._send_json(source)

        # GET /api/sources/:id/videos
        m = re.match(r'^/api/sources/(\d+)/videos$', path)
        if m:
            filters = {
                "status": query.get("status", ["all"])[0],
                "present": query.get("present", ["all"])[0],
                "q": query.get("q", [None])[0],
                "sort": query.get("sort", ["position"])[0],
            }
            videos = db.get_source_videos(int(m.group(1)), filters)
            return self._send_json({"videos": videos})

        if path == "/api/videos/recently_used":
            videos = db.get_recently_used_videos()
            return self._send_json({"videos": videos})

        if path == "/api/videos/random_unused":
            video = db.get_random_unused_video()
            if not video:
                return self._send_error_json(404, "No unused videos found")
            return self._send_json(video)

        # GET /api/videos/:youtube_video_id
        m = re.match(r'^/api/videos/([A-Za-z0-9_-]+)$', path)
        if m:
            video = db.get_video_detail(m.group(1))
            if not video:
                return self._send_error_json(404, "Video not found")
            return self._send_json(video)

        # Channels
        if path == "/api/channels":
            channels = db.get_channels_with_stats()
            return self._send_json({"channels": channels})

        # Pull Runs
        if path == "/api/sources/pull_runs/active":
            conn = db.get_connection()
            rows = conn.execute("SELECT * FROM pull_runs ORDER BY started_at DESC LIMIT 50").fetchall()
            conn.close()
            return self._send_json({"pull_runs": [dict(r) for r in rows]})


        m = re.match(r'^/api/channels/(\d+)$', path)
        if m:
            channel = db.get_channel(int(m.group(1)))
            if not channel:
                return self._send_error_json(404, "Channel not found")
            return self._send_json({"channel": channel})

        m = re.match(r'^/api/channels/(\d+)/videos$', path)
        if m:
            filters = {
                "status": query.get("status", ["all"])[0],
                "q": query.get("q", [None])[0],
            }
            channel = db.get_channel(int(m.group(1)))
            if not channel:
                return self._send_error_json(404, "Channel not found")
            videos = db.get_channel_videos(int(m.group(1)), filters)
            return self._send_json({
                "channel": channel,
                "stats": {
                    "total": channel.get("total_videos", 0),
                    "used": channel.get("used_count", 0),
                    "unused": channel.get("unused_count", 0),
                    "candidate": channel.get("candidate_count", 0),
                    "skipped": channel.get("skipped_count", 0),
                    "not_used_yet": channel.get("not_used_yet", 0),
                },
                "videos": videos,
            })

        # Search
        if path == "/api/search":
            q = query.get("q", [""])[0]
            status = query.get("status", [None])[0]
            if not q:
                return self._send_json({"videos": []})
            videos = db.search_global(q, status)
            return self._send_json({"videos": videos})

        # Duplicates
        if path == "/api/duplicates":
            status = query.get("status", [None])[0]
            result = db.get_duplicates(status)
            return self._send_json(result)

        # Export
        if path == "/api/export/json":
            data = db.export_json()
            return self._send_json(data)

        if path == "/api/export/csv":
            csv_data = db.export_csv_rows()
            body = csv_data.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", "attachment; filename=youtube_tracker_export.csv")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # Settings
        if path == "/api/settings":
            settings = db.get_settings()
            return self._send_json(settings)

        return self._send_error_json(404, "Not found")

    def _route_post(self, path, query):
        """Route POST requests."""

        # Add playlist
        if path == "/api/sources/playlist":
            body = self._read_body()
            url = body.get("url", "").strip()
            if not url:
                return self._send_error_json(400, "URL is required")

            from youtube_fetcher import extract_playlist_id
            pl_id = extract_playlist_id(url)
            if not pl_id:
                return self._send_error_json(400, "Invalid playlist URL")

            try:
                # Start async pull
                result = _start_async_pull(url)
                return self._send_json(result, 202)
            except Exception as e:
                traceback.print_exc()
                return self._send_error_json(500, f"Failed to start fetch: {str(e)}")

        # Refresh/Pull Again
        m = re.match(r'^/api/sources/(\d+)/refresh$', path)
        if m:
            source_id = int(m.group(1))
            source = db.get_source(source_id)
            if not source:
                return self._send_error_json(404, "Source not found")

            url = source.get("url")
            if not url:
                return self._send_error_json(400, "Source has no URL to refresh")

            try:
                result = _start_async_pull(url, source_id)
                return self._send_json(result, 202)
            except Exception as e:
                traceback.print_exc()
                return self._send_error_json(500, f"Pull failed: {str(e)}")

        m = re.match(r'^/api/channels/(\d+)/avatar$', path)
        if m:
            channel_id = int(m.group(1))
            channel = db.get_channel(channel_id)
            if not channel or not channel.get("url"):
                return self._send_error_json(404, "Channel not found or no URL")
            from youtube_fetcher import fetch_channel_thumbnail
            t = fetch_channel_thumbnail(channel["url"])
            if t:
                db.update_channel_thumbnail(channel_id, t)
            return self._send_json({"thumbnail_url": t})

        # Add manual video
        if path == "/api/videos/manual":
            body = self._read_body()
            url = body.get("url", "").strip()
            if not url:
                return self._send_error_json(400, "URL is required")

            from youtube_fetcher import extract_video_id as evi
            vid_id = evi(url)
            if not vid_id:
                return self._send_error_json(400, "Invalid video URL")

            # Check if video already exists
            existing = db.get_video_detail(vid_id)
            if existing:
                return self._send_error_json(409, "Video ini sudah pernah ditambahkan!")

            try:
                vmeta = fetcher.fetch_video(url)

                ch_db_id = None
                if vmeta.get("channel_id"):
                    ch_db_id = db.upsert_channel({
                        "channel_id": vmeta["channel_id"],
                        "channel_name": vmeta.get("channel_name"),
                        "channel_url": vmeta.get("channel_url"),
                        "handle": vmeta.get("handle"),
                    })

                vmeta["channel_db_id"] = ch_db_id
                video_db_id, was_new = db.upsert_video(vmeta)
                db.ensure_video_status(video_db_id)

                manual_source_id = db.create_or_get_manual_source()
                db.link_source_video(
                    manual_source_id, video_db_id,
                    metadata_source="manual",
                )

                return self._send_json({
                    "ok": True,
                    "video_id": vmeta["video_id"],
                    "video_db_id": video_db_id,
                    "was_new": was_new,
                    "title": vmeta.get("title"),
                }, 201)

            except Exception as e:
                traceback.print_exc()
                return self._send_error_json(500, f"Failed to fetch video: {str(e)}")

        # Generate clipping command
        if path == "/api/command":
            body = self._read_body()
            yt_vid_id = body.get("youtube_video_id", "").strip()
            if not yt_vid_id:
                return self._send_error_json(400, "youtube_video_id is required")
            cmd = db.build_clipping_command(yt_vid_id)
            if cmd is None:
                return self._send_error_json(404, "Video not found")
            return self._send_json({"command": cmd})

        return self._send_error_json(404, "Not found")

    def _route_patch(self, path, query):
        """Route PATCH requests."""

        # Bulk update video status
        if path == "/api/videos/bulk_status":
            body = self._read_body()
            video_ids = body.get("video_ids", [])
            status = body.get("status")
            if not video_ids or not status:
                return self._send_error_json(400, "video_ids and status are required")
            result = db.update_bulk_video_status(video_ids, status)
            return self._send_json(result)

        # Update single video status
        m = re.match(r'^/api/videos/([A-Za-z0-9_-]+)/status$', path)
        if m:
            yt_vid_id = m.group(1)
            body = self._read_body()
            result = db.update_video_status(yt_vid_id, body)
            if result is None:
                return self._send_error_json(404, "Video not found")
            return self._send_json(result)

        # Update settings
        if path == "/api/settings":
            body = self._read_body()
            db.update_settings(body)
            return self._send_json({"ok": True})

        return self._send_error_json(404, "Not found")

    def _route_delete(self, path, query):
        """Route DELETE requests."""

        m = re.match(r'^/api/sources/(\d+)$', path)
        if m:
            source_id = int(m.group(1))
            source = db.get_source(source_id)
            if not source:
                return self._send_error_json(404, "Source not found")
            db.delete_source(source_id)
            return self._send_json({"ok": True, "deleted": source_id})

        return self._send_error_json(404, "Not found")

    # -- HTTP methods ---

    def do_GET(self):
        try:
            path, query = self._parse_path()
            self._route_get(path, query)
        except Exception as e:
            traceback.print_exc()
            self._send_error_json(500, str(e))

    def do_POST(self):
        try:
            path, query = self._parse_path()
            self._route_post(path, query)
        except Exception as e:
            traceback.print_exc()
            self._send_error_json(500, str(e))

    def do_PATCH(self):
        try:
            path, query = self._parse_path()
            self._route_patch(path, query)
        except Exception as e:
            traceback.print_exc()
            self._send_error_json(500, str(e))

    def do_DELETE(self):
        try:
            path, query = self._parse_path()
            self._route_delete(path, query)
        except Exception as e:
            traceback.print_exc()
            self._send_error_json(500, str(e))

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  YouTube Tracker — Local Playlist Snapshot Tracker")
    print("=" * 60)

    # Initialize database
    db.init_db()
    print(f"  Database : {db.DB_PATH}")
    print(f"  Static   : {STATIC_DIR}")
    print(f"  Server   : http://{HOST}:{PORT}")
    print("=" * 60)

    server = HTTPServer((HOST, PORT), TrackerHandler)
    try:
        print(f"\n  🚀 Server running at http://{HOST}:{PORT}")
        print("  Press Ctrl+C to stop.\n")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        import os
        os._exit(0)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
