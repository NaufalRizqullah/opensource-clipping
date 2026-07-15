"""
facebook_uploader.uploader — Core Facebook Reels Upload & Scheduling Logic

Uploads Reels to a Facebook Page via the Meta Graph API.
Follows the same manifest-based pattern as youtube_uploader.

API Flow per clip:
  1. Create Reel session  (POST /me/video_reels, upload_phase=start)
  2. Upload binary file   (POST to upload_url)
  3. Poll processing      (GET /{VIDEO_ID}?fields=status)
  4. Finish: publish/schedule (POST /me/video_reels, upload_phase=finish)
"""

import os
import json
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests


# ==============================================================================
# CONFIG & AUTH
# ==============================================================================

def get_meta_config() -> dict:
    """
    Read Meta/Facebook configuration from environment variables.
    Expected env vars: META_PAGE_ID, META_PAGE_ACCESS_TOKEN, META_GRAPH_VERSION.
    """
    page_id = os.environ.get("META_PAGE_ID", "").strip()
    token = os.environ.get("META_PAGE_ACCESS_TOKEN", "").strip()
    version = os.environ.get("META_GRAPH_VERSION", "v25.0").strip()

    if not page_id:
        raise RuntimeError(
            "META_PAGE_ID belum di-set. "
            "Tambahkan ke .env atau environment variable."
        )
    if not token:
        raise RuntimeError(
            "META_PAGE_ACCESS_TOKEN belum di-set. "
            "Tambahkan ke .env atau environment variable."
        )

    return {
        "page_id": page_id,
        "access_token": token,
        "graph_version": version,
        "base_url": f"https://graph.facebook.com/{version}",
    }


def _auth_headers(config: dict) -> dict:
    """Return Authorization header for Graph API requests."""
    return {"Authorization": f"Bearer {config['access_token']}"}


def validate_page_token(config: dict) -> dict:
    """
    GET /me?fields=id,name — validate that the Page Access Token is still valid.
    Returns {"id": "PAGE_ID", "name": "Page Name"}.
    """
    url = f"{config['base_url']}/me"
    params = {"fields": "id,name"}

    resp = requests.get(url, headers=_auth_headers(config), params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"Token validation failed: {data['error']}")

    return data


# ==============================================================================
# SCHEDULE LOGIC
# ==============================================================================

def get_latest_future_schedule(config: dict, tz_name: str = "Asia/Makassar") -> datetime | None:
    """
    GET /{PAGE_ID}/scheduled_posts — find the latest scheduled_publish_time
    that is still in the future. Follows pagination automatically.
    """
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    latest_dt = None

    url = f"{config['base_url']}/{config['page_id']}/scheduled_posts"
    params = {
        "fields": "id,scheduled_publish_time",
        "limit": "100",
    }

    print("🔎 Mengecek scheduled posts di Facebook Page...")

    while url:
        resp = requests.get(url, headers=_auth_headers(config), params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            print(f"⚠️ Error reading scheduled posts: {data['error'].get('message', data['error'])}")
            break

        for post in data.get("data", []):
            ts = post.get("scheduled_publish_time")
            if ts is None:
                continue

            # scheduled_publish_time is a Unix timestamp (int or string)
            try:
                dt = datetime.fromtimestamp(int(ts), tz=tz)
            except (ValueError, TypeError):
                continue

            if dt <= now:
                continue  # Abaikan jadwal yang sudah lewat

            if latest_dt is None or dt > latest_dt:
                latest_dt = dt

        # Follow pagination
        paging = data.get("paging", {})
        next_url = paging.get("next")
        if next_url:
            url = next_url
            params = {}  # params sudah di-encode di next_url
        else:
            break

    if latest_dt:
        print(f"✅ Scheduled terakhir ditemukan: {latest_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        print("ℹ️ Belum ada post terjadwal di masa depan.")

    return latest_dt


# ==============================================================================
# REEL UPLOAD FLOW (4-step Meta API)
# ==============================================================================

def create_reel_session(config: dict) -> dict:
    """
    POST /me/video_reels (upload_phase=start)
    Returns {"video_id": "...", "upload_url": "..."}.
    """
    url = f"{config['base_url']}/me/video_reels"
    payload = {"upload_phase": "start"}

    resp = requests.post(url, headers=_auth_headers(config), data=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"Create reel session failed: {data['error']}")

    video_id = data.get("video_id")
    upload_url = data.get("upload_url")

    if not video_id or not upload_url:
        raise RuntimeError(f"Unexpected response from create session: {data}")

    return {"video_id": video_id, "upload_url": upload_url}


def upload_reel_binary(upload_url: str, file_path: str, token: str) -> bool:
    """
    POST binary file to the upload_url provided by create_reel_session.
    Uses OAuth token in header (not Bearer — Meta uses "OAuth" for upload endpoint).
    """
    file_size = os.path.getsize(file_path)

    headers = {
        "Authorization": f"OAuth {token}",
        "offset": "0",
        "file_size": str(file_size),
        "Content-Type": "application/octet-stream",
    }

    with open(file_path, "rb") as f:
        resp = requests.post(upload_url, headers=headers, data=f, timeout=600)

    resp.raise_for_status()
    data = resp.json()

    if not data.get("success"):
        raise RuntimeError(f"Upload binary failed: {data}")

    return True


def poll_reel_status(
    config: dict,
    video_id: str,
    timeout_seconds: int = 60,
    poll_interval: int = 10,
) -> bool:
    """
    GET /{VIDEO_ID}?fields=status — poll until processing is complete, or progress to next if still processing.
    Returns True if completed/ready, False if still processing when timeout reached.
    """
    url = f"{config['base_url']}/{video_id}"
    params = {"fields": "status"}

    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        try:
            resp = requests.get(url, headers=_auth_headers(config), params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"   ⚠️ Gagal mengecek status video: {e}")
            if elapsed > timeout_seconds:
                return False
            time.sleep(poll_interval)
            continue

        status = data.get("status", {})
        video_status = status.get("video_status", "")
        processing_phase = status.get("processing_phase", {}).get("status", "")

        progress = status.get("processing_progress", 0)
        print(f"   ... processing: {progress}% (status={video_status})")

        # Processing complete
        if video_status == "ready" or processing_phase == "complete":
            return True

        # Error
        if video_status == "error":
            error_info = status.get("processing_phase", {}).get("error", {})
            raise RuntimeError(f"Video processing error: {error_info}")

        if elapsed > timeout_seconds:
            print(f"   ℹ️ Video masih diproses di background oleh Meta (status={video_status}). Melanjutkan batch...")
            return False

        time.sleep(poll_interval)


def finish_reel(
    config: dict,
    video_id: str,
    description: str,
    title: str,
    video_state: str = "PUBLISHED",
    scheduled_timestamp: int | None = None,
) -> dict:
    """
    POST /me/video_reels (upload_phase=finish)
    Publish immediately or schedule for later.
    """
    url = f"{config['base_url']}/me/video_reels"
    payload = {
        "video_id": video_id,
        "upload_phase": "finish",
        "video_state": video_state,
        "description": description,
        "title": title,
    }

    if video_state == "SCHEDULED":
        if scheduled_timestamp is None:
            raise ValueError("scheduled_timestamp wajib diisi untuk video_state=SCHEDULED")
        payload["scheduled_publish_time"] = str(scheduled_timestamp)

    resp = requests.post(url, headers=_auth_headers(config), data=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"Finish reel failed: {data['error']}")

    if not data.get("success"):
        raise RuntimeError(f"Unexpected finish response: {data}")

    return data


# ==============================================================================
# HELPER FILE / MANIFEST
# ==============================================================================

def load_json_file(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_nonempty_file(path):
    return bool(path) and os.path.exists(path) and os.path.isfile(path) and os.path.getsize(path) > 0


def normalize_text(text):
    return " ".join(str(text or "").split()).strip()


def get_upload_candidates(render_manifest):
    """Return items that are status=success and have a valid video file."""
    candidates = []
    for item in render_manifest:
        if item.get("status") != "success":
            continue
        if not is_nonempty_file(item.get("video_path")):
            continue
        candidates.append(item)
    return candidates


def get_manifest_row_by_rank(manifest_rows, rank):
    for row in manifest_rows:
        if row.get("rank") == rank:
            return row
    return None


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def _get_clip_metadata(item: dict) -> tuple[str, str]:
    """
    Extract title and description for Facebook Reels from manifest item.
    Uses YouTube fields (English) as primary source.
    Falls back to Indonesian title if YouTube fields are empty.
    """
    title = (
        item.get("youtube_title_final")
        or item.get("title_inggris")
        or item.get("title_indonesia")
        or f"Clip Rank {item.get('rank', '?')}"
    )
    title = normalize_text(title)[:100]

    description = (
        item.get("youtube_description_final")
        or item.get("tiktok_caption_final")
        or ""
    )
    description = normalize_text(description)

    return title, description


def upload_manifest_to_facebook(
    manifest_file: str = "outputs/render_manifest.json",
    result_file: str = "outputs/fb_upload_results.json",
    updated_manifest_file: str = "outputs/render_manifest_fb_uploaded.json",
    tz_name: str = "Asia/Makassar",
    interval_hours: int = 5,
    test_mode: bool = False,
) -> list:
    """
    Main pipeline: read manifest → determine schedule → upload Reels one by one.

    Scheduling logic (from plan reference §6):
      - Read latest future schedule from Meta
      - Maintain last_assigned_time cursor
      - First clip: publish now if no queue, else schedule
      - Subsequent clips: schedule at last_assigned_time + interval
      - If SCHEDULED fails: STOP batch (no fallback to PUBLISHED)
    """
    # Load .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # --- Load Config & Validate Token ---
    config = get_meta_config()

    print("🔑 Validasi Page Access Token...")
    page_info = validate_page_token(config)
    print(f"✅ Token valid untuk Page: {page_info.get('name')} (ID: {page_info.get('id')})")

    # --- Load Manifest ---
    render_manifest = load_json_file(manifest_file, default=[])
    if not render_manifest:
        print(f"⚠️ {manifest_file} kosong / tidak ditemukan.")
        return []

    candidates = get_upload_candidates(render_manifest)
    if not candidates:
        print("⚠️ Tidak ada item yang siap diupload.")
        return []

    # Filter out already-uploaded items
    pending_items = []
    for item in candidates:
        if item.get("fb_video_id") and item.get("fb_upload_status") == "uploaded":
            print(f"⏭️ Skip Rank {item.get('rank')} karena sudah pernah diupload ke Facebook.")
            continue
        pending_items.append(item)

    if test_mode and pending_items:
        pending_items = pending_items[:1]
        print("🧪 Mode test aktif: hanya upload 1 item pertama.")

    if not pending_items:
        print("⚠️ Semua item success sudah pernah diupload ke Facebook.")
        return []

    # --- Determine Schedule ---
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    interval = timedelta(hours=interval_hours)

    # Get latest future schedule from Meta
    latest_meta_schedule = get_latest_future_schedule(config, tz_name)
    last_assigned_time = latest_meta_schedule  # Could be None

    upload_results = []
    updated_manifest = deepcopy(render_manifest)

    print(f"\n🚀 Mulai upload {len(pending_items)} clip ke Facebook Page...")
    print(f"   Interval antar video: {interval_hours} jam")
    print(f"   Timezone: {tz_name}")

    for idx, item in enumerate(pending_items):
        rank = item.get("rank")
        manifest_row = get_manifest_row_by_rank(updated_manifest, rank)
        title, description = _get_clip_metadata(item)
        video_path = item.get("video_path", "")

        # --- Refresh schedule from Meta (untuk menangkap jadwal dari luar) ---
        if idx > 0:
            latest_meta_schedule = get_latest_future_schedule(config, tz_name)
            if latest_meta_schedule is not None:
                if last_assigned_time is None:
                    last_assigned_time = latest_meta_schedule
                else:
                    last_assigned_time = max(last_assigned_time, latest_meta_schedule)

        # --- Determine publish mode ---
        if last_assigned_time is None:
            # No queue — publish immediately
            video_state = "PUBLISHED"
            scheduled_at = None
            scheduled_timestamp = None
            last_assigned_time = datetime.now(tz)
            mode_label = "PUBLISH NOW"
        else:
            # Schedule at last_assigned_time + interval
            scheduled_at = last_assigned_time + interval
            scheduled_timestamp = int(scheduled_at.timestamp())
            video_state = "SCHEDULED"
            last_assigned_time = scheduled_at
            mode_label = f"SCHEDULED → {scheduled_at.strftime('%Y-%m-%d %H:%M %Z')}"

        print(f"\n{'=' * 60}")
        print(f"=== Clip {idx + 1}/{len(pending_items)} — Rank {rank} ===")
        print(f"Judul  : {title}")
        print(f"Video  : {os.path.basename(video_path)}")
        print(f"Mode   : {mode_label}")
        print(f"{'=' * 60}")

        try:
            # Step 1: Create Reel session
            print("   📝 Membuat sesi upload Reel...")
            session = create_reel_session(config)
            video_id = session["video_id"]
            upload_url = session["upload_url"]
            print(f"   ✅ Session created. Video ID: {video_id}")

            # Step 2: Upload binary
            print(f"   ⬆️ Uploading: {os.path.basename(video_path)} ({os.path.getsize(video_path) / 1024 / 1024:.1f} MB)...")
            upload_reel_binary(upload_url, video_path, config["access_token"])
            print("   ✅ Upload binary berhasil.")

            # Step 3: Finish — publish or schedule (Meta needs finish_reel before starting video processing)
            print(f"   🎬 Finishing reel ({video_state})...")
            finish_result = finish_reel(
                config=config,
                video_id=video_id,
                description=description,
                title=title,
                video_state=video_state,
                scheduled_timestamp=scheduled_timestamp,
            )
            print(f"   ✅ Reel {video_state} berhasil didaftarkan!")

            # Step 4: Poll processing status (Non-blocking: max 60s, for logging status only)
            print("   ⏳ Mengecek status processing awal...")
            poll_reel_status(config, video_id, timeout_seconds=60, poll_interval=10)

            # --- Update manifest row ---
            if manifest_row is not None:
                manifest_row["fb_upload_status"] = "uploaded"
                manifest_row["fb_video_id"] = video_id
                manifest_row["fb_video_state"] = video_state
                manifest_row["fb_scheduled_at"] = (
                    scheduled_at.strftime("%Y-%m-%d %H:%M:%S %Z") if scheduled_at else None
                )
                manifest_row["fb_uploaded_at_utc"] = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                manifest_row["fb_upload_error"] = None

            row_result = {
                "rank": rank,
                "status": "uploaded",
                "filename": os.path.basename(video_path),
                "video_id": video_id,
                "mode": video_state,
                "scheduled_at": (
                    scheduled_at.isoformat() if scheduled_at else None
                ),
                "api_response": finish_result,
            }
            upload_results.append(row_result)

            print(f"   ✅ Upload sukses. Video ID: {video_id}")

        except Exception as e:
            err = str(e)

            if manifest_row is not None:
                manifest_row["fb_upload_status"] = "failed"
                manifest_row["fb_upload_error"] = err

            upload_results.append({
                "rank": rank,
                "status": "failed",
                "filename": os.path.basename(video_path),
                "mode": video_state,
                "error": err,
            })
            print(f"   ❌ Upload gagal untuk Rank {rank}: {err}")

            # STOP batch on failure — do not continue to next clip
            if video_state == "SCHEDULED":
                print("   🛑 Batch dihentikan karena SCHEDULED gagal (tidak fallback ke PUBLISHED).")
            else:
                print("   🛑 Batch dihentikan karena upload gagal.")
            break

        # Incremental save after each clip
        save_json_file(result_file, upload_results)
        save_json_file(updated_manifest_file, updated_manifest)

    # Final save
    save_json_file(result_file, upload_results)
    save_json_file(updated_manifest_file, updated_manifest)

    print(f"\n💾 Hasil upload disimpan ke: {result_file}")
    print(f"💾 Manifest terupdate disimpan ke: {updated_manifest_file}")

    return upload_results
