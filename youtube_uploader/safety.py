"""
youtube_uploader.safety — Upload Safety Guardrails

Prevents aggressive upload patterns that trigger YouTube's spam detection.
Enforces daily limits, minimum intervals, queue caps, and manual approval.
"""

import json
import os
from datetime import datetime, date, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

def _get_tz(tz_name: str):
    """Get timezone, falling back to UTC if ZoneInfo/tzdata unavailable."""
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(tz_name)
    except Exception:
        # tzdata package may not be installed on Windows
        return timezone.utc


# ==============================================================================
# CONFIG LOADER
# ==============================================================================

DEFAULT_SAFETY_CONFIG = {
    "max_upload_per_run": 1,
    "max_upload_per_day": 2,
    "interval_hours_min": 24,
    "max_scheduled_queue": 7,
    "require_manual_approval": True,
    "upload_log_file": "outputs/upload_history.json",
}


def load_safety_config(config_path: str = "upload_safety.json") -> dict:
    """
    Load safety config from JSON file, falling back to defaults.

    Parameters
    ----------
    config_path : str
        Path to the safety config JSON file.

    Returns
    -------
    dict
        Merged config with defaults for any missing keys.
    """
    config = dict(DEFAULT_SAFETY_CONFIG)

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            # Strip $schema key if present
            user_config.pop("$schema", None)
            config.update(user_config)
            print(f"🛡️ Safety config loaded from: {config_path}")
        except Exception as e:
            print(f"⚠️ Gagal membaca {config_path}: {e}. Menggunakan default safety config.")
    else:
        print(f"ℹ️ {config_path} tidak ditemukan. Menggunakan default safety config.")

    return config


# ==============================================================================
# UPLOAD HISTORY TRACKING
# ==============================================================================

def load_upload_history(log_file: str) -> list[dict]:
    """Load upload history from JSON log file."""
    if not os.path.exists(log_file):
        return []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_upload_history(log_file: str, history: list[dict]) -> None:
    """Save upload history to JSON log file."""
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def record_upload(log_file: str, video_id: str, title: str, tz_name: str = "Asia/Makassar") -> None:
    """Record a successful upload to the history log."""
    history = load_upload_history(log_file)
    tz = _get_tz(tz_name)
    now = datetime.now(tz)

    history.append({
        "video_id": video_id,
        "title": title,
        "uploaded_at": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "date": now.strftime("%Y-%m-%d"),
    })

    save_upload_history(log_file, history)


# ==============================================================================
# SAFETY CHECKS
# ==============================================================================

def count_uploads_today(log_file: str, tz_name: str = "Asia/Makassar") -> int:
    """Count how many videos were uploaded today."""
    history = load_upload_history(log_file)
    tz = _get_tz(tz_name)
    today_str = datetime.now(tz).strftime("%Y-%m-%d")

    return sum(1 for entry in history if entry.get("date") == today_str)


def check_daily_limit(safety_config: dict, tz_name: str = "Asia/Makassar") -> tuple[bool, int, int]:
    """
    Check if daily upload limit has been reached.

    Returns
    -------
    tuple[bool, int, int]
        (is_allowed, uploads_today, max_per_day)
    """
    log_file = safety_config["upload_log_file"]
    max_per_day = safety_config["max_upload_per_day"]
    uploads_today = count_uploads_today(log_file, tz_name)

    return uploads_today < max_per_day, uploads_today, max_per_day


def enforce_min_interval(requested_hours: int, safety_config: dict) -> int:
    """
    Ensure upload interval is at least the minimum configured.

    Returns the effective interval (capped to minimum).
    """
    min_hours = safety_config["interval_hours_min"]
    effective = max(requested_hours, min_hours)

    if effective != requested_hours:
        print(
            f"🛡️ Interval diubah dari {requested_hours}h → {effective}h "
            f"(minimum safety: {min_hours}h)"
        )

    return effective


def limit_pending_items(items: list, safety_config: dict) -> list:
    """
    Limit the number of items to upload per run.

    Returns truncated list.
    """
    max_per_run = safety_config["max_upload_per_run"]

    if len(items) > max_per_run:
        print(
            f"🛡️ Membatasi upload: {len(items)} item → {max_per_run} item per run "
            f"(safety limit: max_upload_per_run={max_per_run})"
        )
        return items[:max_per_run]

    return items


def check_queue_limit(
    youtube,
    safety_config: dict,
    tz_name: str = "Asia/Makassar",
) -> tuple[bool, int, int]:
    """
    Check if the number of scheduled (future) videos on YouTube
    exceeds the configured queue limit.

    Returns
    -------
    tuple[bool, int, int]
        (is_allowed, current_queue_count, max_queue)
    """
    from .uploader import get_latest_scheduled_publish_time, parse_rfc3339_to_local

    max_queue = safety_config["max_scheduled_queue"]

    # Count scheduled videos by scanning uploads playlist
    try:
        tz = _get_tz(tz_name)
        now_local = datetime.now(tz)

        channel_resp = youtube.channels().list(part="contentDetails", mine=True).execute()
        channel_items = channel_resp.get("items", [])
        if not channel_items:
            return True, 0, max_queue

        uploads_playlist_id = (
            channel_items[0]
            .get("contentDetails", {})
            .get("relatedPlaylists", {})
            .get("uploads")
        )
        if not uploads_playlist_id:
            return True, 0, max_queue

        scheduled_count = 0
        page_token = None

        for _ in range(5):  # max 5 pages = 250 videos
            playlist_resp = youtube.playlistItems().list(
                part="contentDetails",
                playlistId=uploads_playlist_id,
                maxResults=50,
                pageToken=page_token
            ).execute()

            playlist_items = playlist_resp.get("items", [])
            if not playlist_items:
                break

            video_ids = [
                row.get("contentDetails", {}).get("videoId")
                for row in playlist_items
                if row.get("contentDetails", {}).get("videoId")
            ]

            if video_ids:
                videos_resp = youtube.videos().list(
                    part="status",
                    id=",".join(video_ids),
                    maxResults=50
                ).execute()

                for video in videos_resp.get("items", []):
                    status = video.get("status", {})
                    publish_at = status.get("publishAt")
                    privacy = status.get("privacyStatus")

                    if not publish_at or privacy != "private":
                        continue

                    dt_local = parse_rfc3339_to_local(publish_at, tz_name)
                    if dt_local and dt_local > now_local:
                        scheduled_count += 1

            page_token = playlist_resp.get("nextPageToken")
            if not page_token:
                break

        return scheduled_count < max_queue, scheduled_count, max_queue

    except Exception as e:
        print(f"⚠️ Gagal mengecek scheduled queue: {e}")
        # Fail-open: allow upload if check fails
        return True, 0, max_queue


# ==============================================================================
# MANUAL APPROVAL
# ==============================================================================

def prompt_manual_approval(item: dict, publish_at_local=None) -> bool:
    """
    Show video details and ask user for Y/N confirmation before upload.

    Returns True if approved, False if rejected.
    """
    title = (
        item.get("youtube_title_final")
        or item.get("title_inggris")
        or item.get("title_indonesia")
        or f"Clip Rank {item.get('rank', '?')}"
    )
    video_path = item.get("video_path", "?")
    duration = round(float(item.get("end_time", 0)) - float(item.get("start_time", 0)), 1)

    print("\n" + "=" * 60)
    print("🛡️ MANUAL APPROVAL REQUIRED")
    print("=" * 60)
    print(f"  Judul    : {title}")
    print(f"  File     : {os.path.basename(video_path)}")
    print(f"  Durasi   : ~{duration}s")
    if publish_at_local:
        print(f"  Jadwal   : {publish_at_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("=" * 60)
    print()
    print("⚠️  CHECKLIST SEBELUM APPROVE:")
    print("  □ Sudah review video secara manual?")
    print("  □ Ada narasi/analisis/voice-over kamu sendiri?")
    print("  □ Clip sumber ≤ 10 detik per potongan?")
    print("  □ Judul & thumbnail BUKAN meniru creator asli?")
    print("  □ Description mencantumkan sumber?")
    print()

    while True:
        try:
            answer = input("Upload video ini? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n⏹️ Upload dibatalkan oleh user.")
            return False

        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            print("⏭️ Video dilewati.")
            return False
        print("   Ketik 'y' untuk upload atau 'n' untuk skip.")


# ==============================================================================
# SAFETY SUMMARY PRINTER
# ==============================================================================

def print_safety_summary(safety_config: dict, tz_name: str = "Asia/Makassar") -> None:
    """Print a human-readable summary of active safety rules."""
    is_allowed, uploads_today, max_per_day = check_daily_limit(safety_config, tz_name)

    print("\n" + "=" * 60)
    print("🛡️ UPLOAD SAFETY RULES")
    print("=" * 60)
    print(f"  Max per run        : {safety_config['max_upload_per_run']}")
    print(f"  Max per day        : {max_per_day} (today: {uploads_today})")
    print(f"  Min interval       : {safety_config['interval_hours_min']}h")
    print(f"  Max scheduled queue: {safety_config['max_scheduled_queue']}")
    print(f"  Manual approval    : {'ON ✅' if safety_config['require_manual_approval'] else 'OFF ⚠️'}")
    print(f"  Upload log         : {safety_config['upload_log_file']}")
    print("=" * 60)

    if not is_allowed:
        print(f"\n🚫 DAILY LIMIT REACHED! Sudah {uploads_today}/{max_per_day} upload hari ini.")
        print("   Upload akan di-skip untuk hari ini.")
    print()
