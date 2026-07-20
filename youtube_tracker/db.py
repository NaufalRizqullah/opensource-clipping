"""
youtube_tracker/db.py
Database layer for the YouTube Tracker.
Uses SQLite with parameterized queries throughout.
"""

import csv
import io
import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "youtube_tracker.sqlite3")


def get_connection():
    """Get a new SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS channels (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  youtube_channel_id TEXT UNIQUE,
  name TEXT,
  url TEXT,
  handle TEXT,
  thumbnail_url TEXT,
  raw_json TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sources (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_type TEXT NOT NULL CHECK (
    source_type IN ('playlist', 'manual', 'channel')
  ),
  source_key TEXT UNIQUE,
  youtube_playlist_id TEXT,
  title TEXT NOT NULL,
  url TEXT,
  thumbnail_url TEXT,

  owner_channel_db_id INTEGER,
  owner_channel_name TEXT,
  owner_channel_url TEXT,

  last_pulled_at TEXT,
  raw_json TEXT,

  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

  FOREIGN KEY (owner_channel_db_id)
    REFERENCES channels(id)
    ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS videos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  youtube_video_id TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL,
  url TEXT NOT NULL,
  thumbnail_url TEXT,
  duration_seconds INTEGER,
  upload_date TEXT,
  published_at TEXT,
  description TEXT,

  channel_db_id INTEGER,
  channel_name TEXT,
  channel_url TEXT,

  raw_json TEXT,

  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

  FOREIGN KEY (channel_db_id)
    REFERENCES channels(id)
    ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS pull_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id INTEGER NOT NULL,
  started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  finished_at TEXT,
  status TEXT NOT NULL DEFAULT 'running' CHECK (
    status IN ('running', 'success', 'failed')
  ),
  videos_found INTEGER DEFAULT 0,
  videos_added INTEGER DEFAULT 0,
  videos_updated INTEGER DEFAULT 0,
  videos_already_exists INTEGER DEFAULT 0,
  videos_missing_from_latest_pull INTEGER DEFAULT 0,
  progress_current INTEGER DEFAULT 0,
  progress_total INTEGER DEFAULT 0,
  error_message TEXT,

  FOREIGN KEY (source_id)
    REFERENCES sources(id)
    ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS source_videos (
  source_id INTEGER NOT NULL,
  video_db_id INTEGER NOT NULL,
  position INTEGER,

  first_seen_pull_run_id INTEGER,
  last_seen_pull_run_id INTEGER,

  added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_seen_at TEXT,

  is_present_latest_pull INTEGER NOT NULL DEFAULT 1,
  missing_since TEXT,

  metadata_source TEXT,

  PRIMARY KEY (source_id, video_db_id),

  FOREIGN KEY (source_id)
    REFERENCES sources(id)
    ON DELETE CASCADE,

  FOREIGN KEY (video_db_id)
    REFERENCES videos(id)
    ON DELETE CASCADE,

  FOREIGN KEY (first_seen_pull_run_id)
    REFERENCES pull_runs(id)
    ON DELETE SET NULL,

  FOREIGN KEY (last_seen_pull_run_id)
    REFERENCES pull_runs(id)
    ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS pull_run_videos (
  pull_run_id INTEGER NOT NULL,
  video_db_id INTEGER NOT NULL,
  position INTEGER,
  seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (pull_run_id, video_db_id),

  FOREIGN KEY (pull_run_id)
    REFERENCES pull_runs(id)
    ON DELETE CASCADE,

  FOREIGN KEY (video_db_id)
    REFERENCES videos(id)
    ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS video_status (
  video_db_id INTEGER PRIMARY KEY,
  status TEXT NOT NULL DEFAULT 'unused' CHECK (
    status IN ('unused', 'candidate', 'used', 'skipped')
  ),
  used_at TEXT,
  clip_title TEXT,
  local_output_path TEXT,
  published_url TEXT,
  notes TEXT,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

  FOREIGN KEY (video_db_id)
    REFERENCES videos(id)
    ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS clips (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  video_db_id INTEGER NOT NULL,
  title TEXT,
  local_output_path TEXT,
  published_url TEXT,
  platform TEXT,
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

  FOREIGN KEY (video_db_id)
    REFERENCES videos(id)
    ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tags (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS video_tags (
  video_db_id INTEGER NOT NULL,
  tag_id INTEGER NOT NULL,

  PRIMARY KEY (video_db_id, tag_id),

  FOREIGN KEY (video_db_id)
    REFERENCES videos(id)
    ON DELETE CASCADE,

  FOREIGN KEY (tag_id)
    REFERENCES tags(id)
    ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT
);

CREATE INDEX IF NOT EXISTS idx_channels_youtube_channel_id
  ON channels(youtube_channel_id);

CREATE INDEX IF NOT EXISTS idx_sources_source_key
  ON sources(source_key);

CREATE INDEX IF NOT EXISTS idx_sources_type
  ON sources(source_type);

CREATE INDEX IF NOT EXISTS idx_videos_youtube_video_id
  ON videos(youtube_video_id);

CREATE INDEX IF NOT EXISTS idx_videos_channel_db_id
  ON videos(channel_db_id);

CREATE INDEX IF NOT EXISTS idx_source_videos_source_id
  ON source_videos(source_id);

CREATE INDEX IF NOT EXISTS idx_source_videos_video_db_id
  ON source_videos(video_db_id);

CREATE INDEX IF NOT EXISTS idx_source_videos_present_latest
  ON source_videos(is_present_latest_pull);

CREATE INDEX IF NOT EXISTS idx_video_status_status
  ON video_status(status);

CREATE INDEX IF NOT EXISTS idx_pull_runs_source_id
  ON pull_runs(source_id);
"""


def init_db():
    """Create all tables and indexes if they don't exist."""
    conn = get_connection()
    try:
        conn.executescript(_SCHEMA_SQL)
        
        # Reset any stuck "running" pull runs to "failed" on startup
        conn.execute("""
            UPDATE pull_runs
            SET status = 'failed',
                error_message = 'Interrupted by server restart'
            WHERE status = 'running'
        """)
        
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

def upsert_channel(meta):
    """Insert or update a channel. Returns the channel's db id."""
    yt_id = meta.get("channel_id")
    if not yt_id:
        return None

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id FROM channels WHERE youtube_channel_id = ?", (yt_id,)
        ).fetchone()

        now = _now()
        raw = json.dumps(meta.get("raw_json")) if meta.get("raw_json") else None

        if row:
            conn.execute("""
                UPDATE channels
                SET name = COALESCE(?, name),
                    url = COALESCE(?, url),
                    handle = COALESCE(?, handle),
                    thumbnail_url = COALESCE(?, thumbnail_url),
                    raw_json = COALESCE(?, raw_json),
                    updated_at = ?
                WHERE id = ?
            """, (
                meta.get("channel_name"),
                meta.get("channel_url"),
                meta.get("handle"),
                meta.get("thumbnail"),
                raw,
                now,
                row["id"],
            ))
            conn.commit()
            return row["id"]
        else:
            cur = conn.execute("""
                INSERT INTO channels
                    (youtube_channel_id, name, url, handle, thumbnail_url, raw_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                yt_id,
                meta.get("channel_name"),
                meta.get("channel_url"),
                meta.get("handle"),
                meta.get("thumbnail"),
                raw,
                now, now,
            ))
            conn.commit()
            return cur.lastrowid
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def upsert_source(meta):
    """Insert or update a source (playlist/channel). Returns source db id."""
    source_key = meta.get("source_key")
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id FROM sources WHERE source_key = ?", (source_key,)
        ).fetchone()

        now = _now()
        raw = json.dumps(meta.get("raw_json")) if meta.get("raw_json") else None

        if row:
            conn.execute("""
                UPDATE sources
                SET title = COALESCE(?, title),
                    url = COALESCE(?, url),
                    thumbnail_url = COALESCE(?, thumbnail_url),
                    youtube_playlist_id = COALESCE(?, youtube_playlist_id),
                    owner_channel_db_id = COALESCE(?, owner_channel_db_id),
                    owner_channel_name = COALESCE(?, owner_channel_name),
                    owner_channel_url = COALESCE(?, owner_channel_url),
                    raw_json = COALESCE(?, raw_json),
                    updated_at = ?
                WHERE id = ?
            """, (
                meta.get("title"),
                meta.get("url"),
                meta.get("thumbnail"),
                meta.get("playlist_id"),
                meta.get("owner_channel_db_id"),
                meta.get("owner_channel_name"),
                meta.get("owner_channel_url"),
                raw,
                now,
                row["id"],
            ))
            conn.commit()
            return row["id"]
        else:
            cur = conn.execute("""
                INSERT INTO sources
                    (source_type, source_key, youtube_playlist_id, title, url,
                     thumbnail_url, owner_channel_db_id, owner_channel_name,
                     owner_channel_url, raw_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                meta.get("source_type", "playlist"),
                source_key,
                meta.get("playlist_id"),
                meta.get("title", "Untitled"),
                meta.get("url"),
                meta.get("thumbnail"),
                meta.get("owner_channel_db_id"),
                meta.get("owner_channel_name"),
                meta.get("owner_channel_url"),
                raw,
                now, now,
            ))
            conn.commit()
            return cur.lastrowid
    finally:
        conn.close()


def create_or_get_manual_source():
    """Ensure the 'Manual Videos' source exists and return its id."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id FROM sources WHERE source_key = 'manual'"
        ).fetchone()
        if row:
            return row["id"]
        now = _now()
        cur = conn.execute("""
            INSERT INTO sources
                (source_type, source_key, title, created_at, updated_at)
            VALUES ('manual', 'manual', 'Manual Videos', ?, ?)
        """, (now, now))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------

def upsert_video(meta):
    """Insert or update a video. Returns (video_db_id, was_new)."""
    yt_vid_id = meta.get("video_id")
    if not yt_vid_id:
        raise ValueError("video_id is required")

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id FROM videos WHERE youtube_video_id = ?", (yt_vid_id,)
        ).fetchone()

        now = _now()
        raw = json.dumps(meta.get("raw_json")) if meta.get("raw_json") else None

        if row:
            conn.execute("""
                UPDATE videos
                SET title = COALESCE(?, title),
                    url = COALESCE(?, url),
                    thumbnail_url = COALESCE(?, thumbnail_url),
                    duration_seconds = COALESCE(?, duration_seconds),
                    upload_date = COALESCE(?, upload_date),
                    published_at = COALESCE(?, published_at),
                    description = COALESCE(?, description),
                    channel_db_id = COALESCE(?, channel_db_id),
                    channel_name = COALESCE(?, channel_name),
                    channel_url = COALESCE(?, channel_url),
                    raw_json = COALESCE(?, raw_json),
                    updated_at = ?
                WHERE id = ?
            """, (
                meta.get("title"),
                meta.get("url"),
                meta.get("thumbnail"),
                meta.get("duration_seconds"),
                meta.get("upload_date"),
                meta.get("published_at"),
                meta.get("description"),
                meta.get("channel_db_id"),
                meta.get("channel_name"),
                meta.get("channel_url"),
                raw,
                now,
                row["id"],
            ))
            conn.commit()
            return row["id"], False
        else:
            url = meta.get("url") or f"https://www.youtube.com/watch?v={yt_vid_id}"
            cur = conn.execute("""
                INSERT INTO videos
                    (youtube_video_id, title, url, thumbnail_url, duration_seconds,
                     upload_date, published_at, description,
                     channel_db_id, channel_name, channel_url,
                     raw_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                yt_vid_id,
                meta.get("title", "Untitled"),
                url,
                meta.get("thumbnail"),
                meta.get("duration_seconds"),
                meta.get("upload_date"),
                meta.get("published_at"),
                meta.get("description"),
                meta.get("channel_db_id"),
                meta.get("channel_name"),
                meta.get("channel_url"),
                raw,
                now, now,
            ))
            conn.commit()
            return cur.lastrowid, True
    finally:
        conn.close()


def ensure_video_status(video_db_id):
    """Ensure a video_status row exists for the given video."""
    conn = get_connection()
    try:
        conn.execute("""
            INSERT OR IGNORE INTO video_status (video_db_id, status, updated_at)
            VALUES (?, 'unused', ?)
        """, (video_db_id, _now()))
        conn.commit()
    finally:
        conn.close()


def update_video_status(youtube_video_id, payload):
    """Update status fields for a video by its youtube_video_id."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id FROM videos WHERE youtube_video_id = ?", (youtube_video_id,)
        ).fetchone()
        if not row:
            return None

        vid_id = row["id"]
        ensure_video_status(vid_id)

        # Retrieve current status & used_at
        current = conn.execute(
            "SELECT status, used_at FROM video_status WHERE video_db_id = ?", (vid_id,)
        ).fetchone()

        payload_copy = dict(payload)
        if "status" in payload_copy:
            new_status = payload_copy["status"]
            if new_status == "used":
                if "used_at" not in payload_copy and (not current or not current["used_at"]):
                    payload_copy["used_at"] = _now()
            else:
                if "used_at" not in payload_copy:
                    payload_copy["used_at"] = None

        fields = []
        params = []
        for key in ("status", "used_at", "clip_title", "local_output_path",
                     "published_url", "notes"):
            if key in payload_copy:
                fields.append(f"{key} = ?")
                params.append(payload_copy[key])

        if not fields:
            return {"ok": True}

        fields.append("updated_at = ?")
        params.append(_now())
        params.append(vid_id)

        conn.execute(
            f"UPDATE video_status SET {', '.join(fields)} WHERE video_db_id = ?",
            params,
        )
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


def update_bulk_video_status(youtube_video_ids, status):
    """Update the status for multiple videos at once."""
    if not youtube_video_ids:
        return {"ok": True, "updated": 0}

    conn = get_connection()
    try:
        # Get video DB IDs for the given youtube IDs
        placeholders = ','.join(['?'] * len(youtube_video_ids))
        rows = conn.execute(
            f"SELECT id FROM videos WHERE youtube_video_id IN ({placeholders})",
            youtube_video_ids
        ).fetchall()
        
        vid_ids = [r["id"] for r in rows]
        if not vid_ids:
            return {"ok": True, "updated": 0}

        # Ensure video_status exists
        now = _now()
        for vid_id in vid_ids:
            conn.execute("""
                INSERT OR IGNORE INTO video_status (video_db_id, status, updated_at)
                VALUES (?, 'unused', ?)
            """, (vid_id, now))

        # Bulk update
        vid_placeholders = ','.join(['?'] * len(vid_ids))
        if status == 'used':
            conn.execute(
                f"UPDATE video_status SET status = ?, used_at = COALESCE(used_at, ?), updated_at = ? WHERE video_db_id IN ({vid_placeholders})",
                [status, now, now] + vid_ids
            )
        else:
            conn.execute(
                f"UPDATE video_status SET status = ?, used_at = NULL, updated_at = ? WHERE video_db_id IN ({vid_placeholders})",
                [status, now] + vid_ids
            )
        conn.commit()
        return {"ok": True, "updated": len(vid_ids)}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pull Runs
# ---------------------------------------------------------------------------

def create_pull_run(source_id):
    """Create a new pull run record. Returns the pull_run_id."""
    conn = get_connection()
    try:
        cur = conn.execute("""
            INSERT INTO pull_runs (source_id, started_at, status, progress_current, progress_total)
            VALUES (?, ?, 'running', 0, 0)
        """, (source_id, _now()))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()

def update_pull_run_progress(pull_run_id, current, total):
    """Update progress metrics for a running pull."""
    conn = get_connection()
    try:
        conn.execute("""
            UPDATE pull_runs
            SET progress_current = ?, progress_total = ?
            WHERE id = ?
        """, (current, total, pull_run_id))
        conn.commit()
    finally:
        conn.close()

def finish_pull_run(pull_run_id, status="success", stats=None, error_message=None):
    """Mark a pull run as finished."""
    conn = get_connection()
    try:
        stats = stats or {}
        conn.execute("""
            UPDATE pull_runs
            SET finished_at = ?,
                status = ?,
                videos_found = ?,
                videos_added = ?,
                videos_updated = ?,
                videos_already_exists = ?,
                videos_missing_from_latest_pull = ?,
                error_message = ?
            WHERE id = ?
        """, (
            _now(),
            status,
            stats.get("videos_found", 0),
            stats.get("videos_added", 0),
            stats.get("videos_updated", 0),
            stats.get("videos_already_exists", 0),
            stats.get("videos_missing_from_latest_pull", 0),
            error_message,
            pull_run_id,
        ))
        conn.commit()
    finally:
        conn.close()


def record_pull_run_video(pull_run_id, video_db_id, position=None):
    """Record that a video was seen in a specific pull run."""
    conn = get_connection()
    try:
        conn.execute("""
            INSERT OR IGNORE INTO pull_run_videos (pull_run_id, video_db_id, position, seen_at)
            VALUES (?, ?, ?, ?)
        """, (pull_run_id, video_db_id, position, _now()))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Source-Video Links
# ---------------------------------------------------------------------------

def link_source_video(source_id, video_db_id, position=None,
                      metadata_source=None, pull_run_id=None):
    """Link a video to a source, or update the link if it already exists."""
    conn = get_connection()
    try:
        now = _now()
        existing = conn.execute(
            "SELECT 1 FROM source_videos WHERE source_id = ? AND video_db_id = ?",
            (source_id, video_db_id),
        ).fetchone()

        if existing:
            conn.execute("""
                UPDATE source_videos
                SET position = COALESCE(?, position),
                    last_seen_pull_run_id = COALESCE(?, last_seen_pull_run_id),
                    last_seen_at = ?,
                    is_present_latest_pull = 1,
                    missing_since = NULL,
                    metadata_source = COALESCE(?, metadata_source)
                WHERE source_id = ? AND video_db_id = ?
            """, (position, pull_run_id, now, metadata_source,
                  source_id, video_db_id))
        else:
            conn.execute("""
                INSERT INTO source_videos
                    (source_id, video_db_id, position, first_seen_pull_run_id,
                     last_seen_pull_run_id, added_at, last_seen_at,
                     is_present_latest_pull, metadata_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, (source_id, video_db_id, position, pull_run_id,
                  pull_run_id, now, now, metadata_source))
        conn.commit()
    finally:
        conn.close()


def mark_missing_videos_for_source(source_id, current_pull_run_id):
    """Mark videos not in the current pull run as missing. Returns count."""
    conn = get_connection()
    try:
        now = _now()
        cur = conn.execute("""
            UPDATE source_videos
            SET is_present_latest_pull = 0,
                missing_since = CASE
                    WHEN missing_since IS NULL THEN ?
                    ELSE missing_since
                END
            WHERE source_id = ?
              AND video_db_id NOT IN (
                  SELECT video_db_id FROM pull_run_videos
                  WHERE pull_run_id = ?
              )
              AND is_present_latest_pull = 1
        """, (now, source_id, current_pull_run_id))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Queries: Sources
# ---------------------------------------------------------------------------

def get_sources_with_stats():
    """Get all sources with video/status counts."""
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT
                s.id, s.source_type, s.source_key, s.youtube_playlist_id,
                s.title, s.url, s.thumbnail_url,
                s.owner_channel_db_id, s.owner_channel_name, s.owner_channel_url,
                s.last_pulled_at, s.created_at,
                COUNT(DISTINCT sv.video_db_id) AS total_videos,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'used' THEN 1 ELSE 0 END) AS used_count,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'unused' THEN 1 ELSE 0 END) AS unused_count,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'candidate' THEN 1 ELSE 0 END) AS candidate_count,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'skipped' THEN 1 ELSE 0 END) AS skipped_count
            FROM sources s
            LEFT JOIN source_videos sv ON sv.source_id = s.id
            LEFT JOIN video_status vs ON vs.video_db_id = sv.video_db_id
            GROUP BY s.id
            ORDER BY s.source_type ASC, s.title ASC
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_source(source_id):
    """Get a single source by id with stats and latest pull info."""
    conn = get_connection()
    try:
        row = conn.execute("""
            SELECT
                s.*,
                COUNT(DISTINCT sv.video_db_id) AS total_videos,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'used' THEN 1 ELSE 0 END) AS used_count,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'unused' THEN 1 ELSE 0 END) AS unused_count,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'candidate' THEN 1 ELSE 0 END) AS candidate_count,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'skipped' THEN 1 ELSE 0 END) AS skipped_count
            FROM sources s
            LEFT JOIN source_videos sv ON sv.source_id = s.id
            LEFT JOIN video_status vs ON vs.video_db_id = sv.video_db_id
            WHERE s.id = ?
            GROUP BY s.id
        """, (source_id,)).fetchone()
        if not row:
            return None
        result = dict(row)

        # Latest pull run
        pr = conn.execute("""
            SELECT * FROM pull_runs
            WHERE source_id = ? ORDER BY id DESC LIMIT 1
        """, (source_id,)).fetchone()
        result["latest_pull_run"] = dict(pr) if pr else None
        return result
    finally:
        conn.close()


def get_source_videos(source_id, filters=None):
    """Get videos for a source with optional filters."""
    filters = filters or {}
    conn = get_connection()
    try:
        sql = """
            SELECT
                v.id, v.youtube_video_id, v.title, v.url, v.thumbnail_url,
                v.duration_seconds, v.upload_date, v.channel_db_id,
                v.channel_name, v.channel_url,
                COALESCE(vs.status, 'unused') AS status,
                vs.used_at, vs.clip_title, vs.notes, vs.local_output_path,
                vs.published_url,
                sv.position, sv.is_present_latest_pull, sv.missing_since,
                sv.added_at AS source_added_at
            FROM source_videos sv
            JOIN videos v ON v.id = sv.video_db_id
            LEFT JOIN video_status vs ON vs.video_db_id = v.id
            WHERE sv.source_id = ?
        """
        params = [source_id]

        status = filters.get("status")
        if status and status != "all":
            if status == "not_used_yet":
                sql += " AND COALESCE(vs.status, 'unused') IN ('unused', 'candidate', 'skipped')"
            else:
                sql += " AND COALESCE(vs.status, 'unused') = ?"
                params.append(status)

        present = filters.get("present")
        if present == "present":
            sql += " AND sv.is_present_latest_pull = 1"
        elif present == "missing":
            sql += " AND sv.is_present_latest_pull = 0"

        search = filters.get("q")
        if search:
            sql += " AND (v.title LIKE ? OR v.channel_name LIKE ?)"
            like = f"%{search}%"
            params.extend([like, like])

        sort = filters.get("sort", "position")
        sort_map = {
            "position": "sv.position ASC, v.title ASC",
            "title": "v.title ASC",
            "channel": "v.channel_name ASC, v.title ASC",
            "duration": "v.duration_seconds DESC",
            "status": "status ASC, v.title ASC",
            "date": "v.upload_date DESC",
        }
        sql += f" ORDER BY {sort_map.get(sort, 'sv.position ASC, v.title ASC')}"

        rows = conn.execute(sql, params).fetchall()
        videos = []
        for r in rows:
            v = dict(r)
            # Count how many sources this video appears in
            sc = conn.execute(
                "SELECT COUNT(*) AS cnt FROM source_videos WHERE video_db_id = ?",
                (v["id"],)
            ).fetchone()
            v["source_count"] = sc["cnt"]
            videos.append(v)
        return videos
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Queries: Channels
# ---------------------------------------------------------------------------

def get_channels_with_stats():
    """Get all channels with video/status counts."""
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT
                c.id, c.youtube_channel_id, c.name, c.url, c.handle,
                c.thumbnail_url,
                COUNT(DISTINCT v.id) AS total_videos,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'used' THEN 1 ELSE 0 END) AS used_count,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'unused' THEN 1 ELSE 0 END) AS unused_count,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'candidate' THEN 1 ELSE 0 END) AS candidate_count,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'skipped' THEN 1 ELSE 0 END) AS skipped_count
            FROM channels c
            LEFT JOIN videos v ON v.channel_db_id = c.id
            LEFT JOIN video_status vs ON vs.video_db_id = v.id
            GROUP BY c.id
            HAVING total_videos > 0
            ORDER BY total_videos DESC
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_channel_thumbnail(channel_id, thumbnail_url):
    conn = get_connection()
    try:
        conn.execute("UPDATE channels SET thumbnail_url = ? WHERE id = ?", (thumbnail_url, channel_id))
        conn.commit()
    finally:
        conn.close()

def get_channel(channel_id):
    """Get a single channel with stats."""
    conn = get_connection()
    try:
        row = conn.execute("""
            SELECT
                c.*,
                COUNT(DISTINCT v.id) AS total_videos,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'used' THEN 1 ELSE 0 END) AS used_count,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'unused' THEN 1 ELSE 0 END) AS unused_count,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'candidate' THEN 1 ELSE 0 END) AS candidate_count,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'skipped' THEN 1 ELSE 0 END) AS skipped_count
            FROM channels c
            LEFT JOIN videos v ON v.channel_db_id = c.id
            LEFT JOIN video_status vs ON vs.video_db_id = v.id
            WHERE c.id = ?
            GROUP BY c.id
        """, (channel_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["not_used_yet"] = result["unused_count"] + result["candidate_count"] + result["skipped_count"]
        return result
    finally:
        conn.close()


def get_channel_videos(channel_id, filters=None):
    """Get videos for a channel with optional filters."""
    filters = filters or {}
    conn = get_connection()
    try:
        sql = """
            SELECT
                v.id, v.youtube_video_id, v.title, v.url, v.thumbnail_url,
                v.duration_seconds, v.upload_date, v.channel_db_id,
                v.channel_name, v.channel_url,
                COALESCE(vs.status, 'unused') AS status,
                vs.used_at, vs.clip_title, vs.notes
            FROM videos v
            LEFT JOIN video_status vs ON vs.video_db_id = v.id
            WHERE v.channel_db_id = ?
        """
        params = [channel_id]

        status = filters.get("status")
        if status and status != "all":
            if status == "not_used_yet":
                sql += " AND COALESCE(vs.status, 'unused') IN ('unused', 'candidate', 'skipped')"
            else:
                sql += " AND COALESCE(vs.status, 'unused') = ?"
                params.append(status)

        search = filters.get("q")
        if search:
            sql += " AND v.title LIKE ?"
            params.append(f"%{search}%")

        sql += " ORDER BY v.created_at DESC"

        rows = conn.execute(sql, params).fetchall()
        videos = []
        for r in rows:
            v = dict(r)
            # Get sources this video appears in
            sources = conn.execute("""
                SELECT s.id, s.title, s.source_type
                FROM source_videos sv
                JOIN sources s ON s.id = sv.source_id
                WHERE sv.video_db_id = ?
                ORDER BY s.source_type ASC, s.title ASC
            """, (v["id"],)).fetchall()
            v["sources"] = [dict(s) for s in sources]
            videos.append(v)
        return videos
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Queries: Video detail
# ---------------------------------------------------------------------------

def get_video_detail(youtube_video_id):
    """Get full video detail by youtube_video_id."""
    conn = get_connection()
    try:
        row = conn.execute("""
            SELECT
                v.*,
                COALESCE(vs.status, 'unused') AS status,
                vs.used_at, vs.clip_title, vs.notes,
                vs.local_output_path AS status_local_output_path,
                vs.published_url AS status_published_url,
                c.name AS ch_name, c.url AS ch_url, c.handle AS ch_handle
            FROM videos v
            LEFT JOIN video_status vs ON vs.video_db_id = v.id
            LEFT JOIN channels c ON c.id = v.channel_db_id
            WHERE v.youtube_video_id = ?
        """, (youtube_video_id,)).fetchone()
        if not row:
            return None
        result = dict(row)

        # Sources
        sources = conn.execute("""
            SELECT s.id, s.title, s.source_type, sv.position,
                   sv.is_present_latest_pull, sv.missing_since
            FROM source_videos sv
            JOIN sources s ON s.id = sv.source_id
            WHERE sv.video_db_id = ?
            ORDER BY s.source_type ASC, s.title ASC
        """, (result["id"],)).fetchall()
        result["sources"] = [dict(s) for s in sources]
        return result
    finally:
        conn.close()

def get_random_unused_video():
    """Get 1 random unused video, prioritizing newest."""
    conn = get_connection()
    try:
        # Get top 50 newest unused videos across all sources
        sql = """
            SELECT
                v.id, v.youtube_video_id, v.title, v.url, v.thumbnail_url,
                v.duration_seconds, v.upload_date, v.channel_name, v.channel_url,
                COALESCE(vs.status, 'unused') AS status,
                vs.used_at, vs.clip_title, vs.notes
            FROM videos v
            JOIN video_status vs ON vs.video_db_id = v.id
            WHERE vs.status = 'unused'
            ORDER BY COALESCE(v.published_at, v.upload_date, v.created_at) DESC
            LIMIT 50
        """
        rows = conn.execute(sql).fetchall()
        if not rows:
            return None
        
        # Pick one randomly from the 50
        import random
        r = random.choice(rows)
        v = dict(r)

        # Get sources this video appears in
        sources = conn.execute("""
            SELECT s.id, s.title, s.source_type
            FROM source_videos sv
            JOIN sources s ON s.id = sv.source_id
            WHERE sv.video_db_id = ?
        """, (v["id"],)).fetchall()
        v["sources"] = [dict(s) for s in sources]
        
        return v
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_global(query, status=None):
    """Search videos, channels, sources by text."""
    conn = get_connection()
    try:
        like = f"%{query}%"
        sql = """
            SELECT
                v.id, v.youtube_video_id, v.title, v.url, v.thumbnail_url,
                v.duration_seconds, v.channel_name, v.channel_db_id,
                COALESCE(vs.status, 'unused') AS status,
                vs.used_at, vs.clip_title, vs.notes
            FROM videos v
            LEFT JOIN video_status vs ON vs.video_db_id = v.id
            WHERE (v.title LIKE ?
                   OR v.channel_name LIKE ?
                   OR vs.clip_title LIKE ?
                   OR vs.notes LIKE ?)
        """
        params = [like, like, like, like]

        if status and status != "all":
            if status == "not_used_yet":
                sql += " AND COALESCE(vs.status, 'unused') IN ('unused', 'candidate', 'skipped')"
            else:
                sql += " AND COALESCE(vs.status, 'unused') = ?"
                params.append(status)

        sql += " ORDER BY v.title ASC LIMIT 200"

        rows = conn.execute(sql, params).fetchall()
        videos = []
        for r in rows:
            v = dict(r)
            sources = conn.execute("""
                SELECT s.id, s.title, s.source_type
                FROM source_videos sv
                JOIN sources s ON s.id = sv.source_id
                WHERE sv.video_db_id = ?
            """, (v["id"],)).fetchall()
            v["sources"] = [dict(s) for s in sources]
            videos.append(v)
        return videos
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def get_duplicates(status=None):
    """Get videos that appear in more than one source."""
    conn = get_connection()
    try:
        sql = """
            SELECT
                v.id, v.youtube_video_id, v.title, v.url, v.thumbnail_url,
                v.duration_seconds, v.channel_name, v.channel_db_id,
                COALESCE(vs.status, 'unused') AS status,
                vs.used_at, vs.clip_title, vs.notes,
                COUNT(sv.source_id) AS source_count
            FROM videos v
            JOIN source_videos sv ON sv.video_db_id = v.id
            LEFT JOIN video_status vs ON vs.video_db_id = v.id
        """
        params = []
        where_clauses = []

        if status and status != "all":
            if status == "not_used_yet":
                where_clauses.append("COALESCE(vs.status, 'unused') IN ('unused', 'candidate', 'skipped')")
            else:
                where_clauses.append("COALESCE(vs.status, 'unused') = ?")
                params.append(status)

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        sql += """
            GROUP BY v.id
            HAVING COUNT(sv.source_id) > 1
            ORDER BY source_count DESC, v.title ASC
        """

        rows = conn.execute(sql, params).fetchall()
        videos = []
        total_links = 0
        for r in rows:
            v = dict(r)
            total_links += v["source_count"]
            sources = conn.execute("""
                SELECT s.id, s.title, s.source_type, s.url,
                       sv.position, sv.is_present_latest_pull, sv.missing_since
                FROM source_videos sv
                JOIN sources s ON s.id = sv.source_id
                WHERE sv.video_db_id = ?
                ORDER BY s.source_type ASC, s.title ASC
            """, (v["id"],)).fetchall()
            v["sources"] = [dict(s) for s in sources]
            videos.append(v)

        return {
            "stats": {
                "duplicate_videos": len(videos),
                "total_duplicate_source_links": total_links,
            },
            "videos": videos,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Recently Used
# ---------------------------------------------------------------------------

def get_recently_used_videos(limit=100):
    """Get videos that have status = 'used' ordered by used_at DESC, updated_at DESC."""
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT
                v.id, v.youtube_video_id, v.title, v.url, v.thumbnail_url,
                v.duration_seconds, v.upload_date, v.channel_name, v.channel_db_id,
                COALESCE(vs.status, 'unused') AS status,
                vs.used_at, vs.clip_title, vs.notes
            FROM videos v
            JOIN video_status vs ON vs.video_db_id = v.id
            WHERE vs.status = 'used'
            ORDER BY COALESCE(vs.used_at, vs.updated_at) DESC, vs.updated_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        
        videos = []
        for r in rows:
            v = dict(r)
            # Fetch sources for each video
            sources = conn.execute("""
                SELECT s.id, s.title, s.source_type
                FROM source_videos sv
                JOIN sources s ON s.id = sv.source_id
                WHERE sv.video_db_id = ?
            """, (v["id"],)).fetchall()
            v["sources"] = [dict(s) for s in sources]
            videos.append(v)
        return videos
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Global Stats
# ---------------------------------------------------------------------------

def get_global_stats():
    """Get global statistics for the dashboard."""
    conn = get_connection()
    try:
        stats = {}
        stats["total_sources"] = conn.execute(
            "SELECT COUNT(*) AS c FROM sources"
        ).fetchone()["c"]
        stats["total_videos"] = conn.execute(
            "SELECT COUNT(*) AS c FROM videos"
        ).fetchone()["c"]
        stats["total_channels"] = conn.execute(
            "SELECT COUNT(*) AS c FROM channels"
        ).fetchone()["c"]

        status_counts = conn.execute("""
            SELECT
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'used' THEN 1 ELSE 0 END) AS total_used,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'unused' THEN 1 ELSE 0 END) AS total_unused,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'candidate' THEN 1 ELSE 0 END) AS total_candidate,
                SUM(CASE WHEN COALESCE(vs.status, 'unused') = 'skipped' THEN 1 ELSE 0 END) AS total_skipped
            FROM videos v
            LEFT JOIN video_status vs ON vs.video_db_id = v.id
        """).fetchone()
        stats["total_used"] = status_counts["total_used"] or 0
        stats["total_unused"] = status_counts["total_unused"] or 0
        stats["total_candidate"] = status_counts["total_candidate"] or 0
        stats["total_skipped"] = status_counts["total_skipped"] or 0
        return stats
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def get_settings():
    """Get all settings as a dict."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
        return {r["key"]: r["value"] for r in rows}
    finally:
        conn.close()


def update_settings(payload):
    """Update settings from a dict."""
    conn = get_connection()
    try:
        for key, value in payload.items():
            conn.execute("""
                INSERT INTO settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?
            """, (key, value, value))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Clipping Command Builder
# ---------------------------------------------------------------------------

def build_clipping_command(youtube_video_id):
    """Build the clipping command for a video using settings."""
    conn = get_connection()
    try:
        video = conn.execute(
            "SELECT url FROM videos WHERE youtube_video_id = ?",
            (youtube_video_id,)
        ).fetchone()
        if not video:
            return None

        settings = get_settings()
        cmd = f'python main.py --url "{video["url"]}"'

        setting_flags = {
            "clips": "--clips",
            "ratio": "--ratio",
            "font_style": "--font-style",
            "output_dir": "--output-dir",
        }
        bool_flags = {
            "no_bgm": "--no-bgm",
            "split_screen": "--split-screen",
        }

        for key, flag in setting_flags.items():
            val = settings.get(key)
            if val:
                cmd += f" {flag} {val}"

        for key, flag in bool_flags.items():
            val = settings.get(key)
            if val and val.lower() in ("true", "1", "yes"):
                cmd += f" {flag}"

        return cmd
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_json():
    """Export full summary as JSON-serializable dict."""
    conn = get_connection()
    try:
        sources = get_sources_with_stats()
        stats = get_global_stats()

        videos = conn.execute("""
            SELECT
                v.id, v.youtube_video_id, v.title, v.url, v.thumbnail_url,
                v.duration_seconds, v.upload_date, v.channel_name, v.channel_url,
                COALESCE(vs.status, 'unused') AS status,
                vs.used_at, vs.clip_title, vs.notes
            FROM videos v
            LEFT JOIN video_status vs ON vs.video_db_id = v.id
            ORDER BY v.title ASC
        """).fetchall()

        return {
            "exported_at": _now(),
            "stats": stats,
            "sources": sources,
            "videos": [dict(v) for v in videos],
        }
    finally:
        conn.close()


def export_csv_rows():
    """Export video list as CSV string."""
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT
                v.youtube_video_id, v.title, v.url, v.duration_seconds,
                v.upload_date, v.channel_name, v.channel_url,
                COALESCE(vs.status, 'unused') AS status,
                vs.used_at, vs.clip_title, vs.notes, vs.local_output_path,
                vs.published_url
            FROM videos v
            LEFT JOIN video_status vs ON vs.video_db_id = v.id
            ORDER BY v.title ASC
        """).fetchall()

        output = io.StringIO()
        writer = csv.writer(output)
        if rows:
            writer.writerow(rows[0].keys())
            for r in rows:
                writer.writerow(tuple(r))
        return output.getvalue()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Source last_pulled_at
# ---------------------------------------------------------------------------

def update_source_last_pulled(source_id):
    """Update the last_pulled_at timestamp for a source."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE sources SET last_pulled_at = ?, updated_at = ? WHERE id = ?",
            (_now(), _now(), source_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_source(source_id):
    """Delete a source and its source_videos links (videos remain)."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        conn.commit()
        return True
    finally:
        conn.close()
