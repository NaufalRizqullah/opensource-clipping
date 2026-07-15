"""
facebook_uploader.uploader — Core Facebook Reels Upload & Scheduling Logic

Uploads Reels to a Facebook Page via the Meta Graph API.
Follows the same manifest-based pattern as youtube_uploader.

API Flow per clip:
  1. Create Reel session      (POST /{PAGE_ID}/video_reels, upload_phase=start)
  2. Upload binary file       (POST to upload_url)
  3. Finish: publish/schedule (POST /{PAGE_ID}/video_reels, upload_phase=finish)
  4. Poll status              (GET /{VIDEO_ID}?fields=status)
"""

import os
import json
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests


# ==============================================================================
# META PLATFORM LIMITS
# ==============================================================================

# Maximum Reels via API in a rolling 24-hour window
META_REEL_RATE_LIMIT_24H = 30

# scheduled_publish_time must be at least this many minutes from now
META_SCHEDULE_MIN_MINUTES = 10

# scheduled_publish_time must be at most this many days from now
META_SCHEDULE_MAX_DAYS = 29


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
    POST /{PAGE_ID}/video_reels (upload_phase=start)
    Returns {"video_id": "...", "upload_url": "..."}.
    """
    url = f"{config['base_url']}/{config['page_id']}/video_reels"
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
    timeout_seconds: int = 300,
    poll_interval: int = 10,
) -> dict:
    """
    Poll status Reel sampai publishing_phase.status == complete.
    Berlaku untuk PUBLISHED maupun SCHEDULED reels:
      - PUBLISHED → publish_status == "published"
      - SCHEDULED → publish_status == "scheduled"
    """

    url = f"{config['base_url']}/{video_id}"
    params = {"fields": "status"}

    start_time = time.time()
    last_data = {}

    processing_complete = False
    publishing_complete = False

    while True:
        elapsed = time.time() - start_time

        try:
            resp = requests.get(
                url,
                headers=_auth_headers(config),
                params=params,
                timeout=30,
            )

            if not resp.ok:
                try:
                    error_body = resp.json()
                except ValueError:
                    error_body = resp.text

                raise RuntimeError(
                    f"Status request gagal HTTP {resp.status_code}: {error_body}"
                )

            data = resp.json()
            last_data = data

        except Exception as exc:
            print(f"   ⚠️ Gagal mengecek status video: {exc}")

            if elapsed >= timeout_seconds:
                return {
                    "complete": False,
                    "publish_status": None,
                    "processing_complete": processing_complete,
                    "publishing_complete": publishing_complete,
                    "status": {},
                    "raw": last_data,
                }

            time.sleep(poll_interval)
            continue

        status = data.get("status", {})

        video_status = status.get("video_status", "")
        progress = status.get("processing_progress", 0)

        uploading_phase = status.get("uploading_phase", {})
        processing_phase = status.get("processing_phase", {})
        publishing_phase = status.get("publishing_phase", {})

        uploading_status = uploading_phase.get("status", "")
        processing_status = processing_phase.get("status", "")
        publishing_status = publishing_phase.get("status", "")

        processing_complete = (
            processing_status in {"complete", "completed"}
            or video_status == "ready"
        )
        publishing_complete = publishing_status in {"complete", "completed"}

        print(
            "   ... "
            f"video={video_status}, "
            f"uploading={uploading_status}, "
            f"processing={processing_status}, "
            f"publishing={publishing_status}, "
            f"publish_status={publishing_phase.get('publish_status', '-')}, "
            f"progress={progress}%"
        )

        for phase_name, phase_data in (
            ("uploading_phase", uploading_phase),
            ("processing_phase", processing_phase),
            ("publishing_phase", publishing_phase),
        ):
            if phase_data.get("status") == "error":
                error_info = phase_data.get("error", phase_data)
                raise RuntimeError(
                    f"Facebook Reel gagal pada {phase_name}: {error_info}"
                )

        if video_status == "error":
            raise RuntimeError(
                f"Facebook Reel video_status=error: {status}"
            )

        if publishing_complete:
            publish_status = publishing_phase.get("publish_status", "unknown")
            return {
                "complete": True,
                "publish_status": publish_status,
                "processing_complete": processing_complete,
                "publishing_complete": True,
                "status": status,
                "raw": data,
            }

        if elapsed >= timeout_seconds:
            print(
                f"   ℹ️ Publishing belum selesai setelah {timeout_seconds} detik."
            )

            return {
                "complete": False,
                "publish_status": publishing_phase.get("publish_status"),
                "processing_complete": processing_complete,
                "publishing_complete": False,
                "status": status,
                "raw": data,
            }

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
    POST /{PAGE_ID}/video_reels (upload_phase=finish)
    Publish immediately or schedule for later.
    """
    url = f"{config['base_url']}/{config['page_id']}/video_reels"
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


def count_recent_uploads(manifest_rows: list, hours: int = 24) -> int:
    """
    Count how many items in the manifest were uploaded to Facebook
    within the last `hours` hours (based on fb_uploaded_at_utc).
    Used to enforce Meta's rolling 24-hour rate limit.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    count = 0
    for row in manifest_rows:
        uploaded_at_str = row.get("fb_uploaded_at_utc")
        if not uploaded_at_str:
            continue
        try:
            uploaded_at = datetime.fromisoformat(
                uploaded_at_str.replace("Z", "+00:00")
            )
            if uploaded_at >= cutoff:
                count += 1
        except (ValueError, TypeError):
            continue
    return count


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


def refresh_existing_facebook_statuses(
    config: dict,
    manifest_rows: list,
    updated_manifest_file: str,
) -> None:
    """
    Check Facebook Graph API for existing pending/uploaded/scheduled Reels
    and update their statuses in-place in the manifest.

    Uses publishing_phase.publish_status to accurately determine:
      - "published" → already live
      - "scheduled" → scheduled for future
    """
    modified = False
    # Statuses that still need synchronization with Meta
    target_statuses = {"pending", "uploaded", "scheduled_processing", "scheduled"}

    for row in manifest_rows:
        video_id = row.get("fb_video_id")
        status = row.get("fb_upload_status")

        if video_id and status in target_statuses:
            print(f"🔄 Mensinkronisasi status Facebook untuk Video ID {video_id} (status saat ini: {status})...")
            url = f"{config['base_url']}/{video_id}"
            params = {"fields": "status"}

            try:
                resp = requests.get(url, headers=_auth_headers(config), params=params, timeout=15)
                if resp.ok:
                    data = resp.json()
                    fb_status = data.get("status", {})

                    video_status = fb_status.get("video_status", "")
                    processing_status = fb_status.get("processing_phase", {}).get("status", "")
                    publishing_phase = fb_status.get("publishing_phase", {})
                    publishing_status = publishing_phase.get("status", "")
                    publish_status = publishing_phase.get("publish_status", "")

                    new_status = None

                    # Jika phase publishing sudah complete, gunakan publish_status untuk
                    # menentukan status akhir secara akurat
                    if publishing_status in {"complete", "completed"}:
                        if publish_status == "published":
                            new_status = "published"
                        elif publish_status == "scheduled":
                            new_status = "scheduled"
                        else:
                            # Fallback: publishing complete tapi publish_status tidak dikenali
                            new_status = "published"

                    # Jika processing sudah complete tapi publishing belum, tandai pending
                    elif processing_status in {"complete", "completed"} or video_status == "ready":
                        if status in {"uploaded", "scheduled_processing"}:
                            new_status = "pending"

                    # Selalu catat kapan status terakhir kali dicheck
                    row["fb_status_checked_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

                    if new_status and new_status != status:
                        row["fb_upload_status"] = new_status
                        row["fb_publish_status_raw"] = fb_status
                        print(f"   ✅ Status diperbarui: {status} ➔ {new_status}")
                        modified = True
                    else:
                        print(f"   ℹ️ Status masih sama (video={video_status}, processing={processing_status}, publishing={publishing_status}, publish_status={publish_status})")
                        # Timestamp check ditambahkan, tandai manifest sebagai modified
                        modified = True
                else:
                    print(f"   ⚠️ Request status gagal HTTP {resp.status_code}")
            except Exception as e:
                print(f"   ⚠️ Gagal mensinkronisasi status untuk {video_id}: {e}")

    if modified:
        save_json_file(updated_manifest_file, manifest_rows)
        print("💾 Manifest terupdate disimpan setelah sinkronisasi status.")


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
    
    if str(page_info.get("id")) != str(config["page_id"]):
        raise RuntimeError(
            "META_PAGE_ACCESS_TOKEN tidak cocok dengan META_PAGE_ID. "
            f"Token mengarah ke ID {page_info.get('id')}, "
            f"sedangkan konfigurasi menggunakan {config['page_id']}."
        )
    print(f"✅ Token valid untuk Page: {page_info.get('name')} (ID: {page_info.get('id')})")

    # --- Load Manifest ---
    source_manifest_file = manifest_file
    if (
        updated_manifest_file
        and os.path.exists(updated_manifest_file)
        and os.path.getsize(updated_manifest_file) > 0
    ):
        source_manifest_file = updated_manifest_file
        print(f"📂 Menggunakan manifest Facebook sebelumnya: {source_manifest_file}")
    else:
        print(f"📂 Menggunakan manifest awal: {source_manifest_file}")

    render_manifest = load_json_file(source_manifest_file, default=[])
    if not render_manifest:
        print(f"⚠️ {source_manifest_file} kosong / tidak ditemukan.")
        return []

    # Sync existing pending / scheduled statuses before processing
    refresh_existing_facebook_statuses(config, render_manifest, updated_manifest_file)

    candidates = get_upload_candidates(render_manifest)
    if not candidates:
        print("⚠️ Tidak ada item yang siap diupload.")
        return []

    # Filter out already-uploaded items
    pending_items = []
    for item in candidates:
        fb_video_id = item.get("fb_video_id")
        fb_status = item.get("fb_upload_status")

        if fb_video_id:
            print(
                f"⏭️ Skip Rank {item.get('rank')} karena sudah memiliki "
                f"Facebook Video ID: {fb_video_id} (status={fb_status or 'unknown'})"
            )
            continue
        pending_items.append(item)

    # --- Rate Limit Check (30 Reels / 24 jam rolling window) ---
    recent_count = count_recent_uploads(render_manifest, hours=24)
    remaining_quota = max(0, META_REEL_RATE_LIMIT_24H - recent_count)
    print(f"\n📊 Rate Limit: {recent_count}/{META_REEL_RATE_LIMIT_24H} Reels sudah diupload dalam 24 jam terakhir.")

    if remaining_quota == 0:
        print("🛑 Rate limit tercapai! Tidak bisa upload Reel lagi dalam 24 jam ini.")
        return []

    if len(pending_items) > remaining_quota:
        print(
            f"⚠️ Hanya {remaining_quota} dari {len(pending_items)} clip yang akan diupload "
            f"(sisa kuota 24 jam)."
        )
        pending_items = pending_items[:remaining_quota]

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

            # --- Validasi batas jadwal Meta ---
            now_check = datetime.now(tz)
            min_schedule = now_check + timedelta(minutes=META_SCHEDULE_MIN_MINUTES)
            max_schedule = now_check + timedelta(days=META_SCHEDULE_MAX_DAYS)

            # Jika terlalu dekat (< 10 menit), bump ke minimum
            if scheduled_at < min_schedule:
                print(
                    f"   ⚠️ Jadwal {scheduled_at.strftime('%H:%M:%S')} terlalu dekat "
                    f"(min {META_SCHEDULE_MIN_MINUTES} menit). Di-bump ke {min_schedule.strftime('%H:%M:%S')}."
                )
                scheduled_at = min_schedule

            # Jika melebihi 29 hari, stop batch
            if scheduled_at > max_schedule:
                print(
                    f"   🛑 Jadwal {scheduled_at.strftime('%Y-%m-%d %H:%M')} melebihi batas "
                    f"maksimal Meta ({META_SCHEDULE_MAX_DAYS} hari). Batch dihentikan."
                )
                break

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

        video_id = None
        post_id = None
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
            post_id = finish_result.get("post_id")
            print(f"   ✅ Reel {video_state} berhasil didaftarkan!")
            if post_id:
                print(f"   📌 Post ID: {post_id}")

            # Step 4: Poll status — unified for PUBLISHED & SCHEDULED
            #   Meta returns publishing_phase.status == "complete" for both.
            #   publish_status differentiates: "published" vs "scheduled".
            print("   ⏳ Menunggu publishing phase selesai...")
            status_result = poll_reel_status(
                config=config,
                video_id=video_id,
                timeout_seconds=300,
                poll_interval=10,
            )

            if status_result["complete"]:
                publish_status = status_result.get("publish_status", "unknown")
                # Map publish_status to internal status
                if publish_status == "published":
                    final_status = "published"
                elif publish_status == "scheduled":
                    final_status = "scheduled"
                else:
                    final_status = publish_status  # e.g. "draft", "error"
            else:
                # Timeout — tentukan berdasarkan apa yang sudah tercapai
                if status_result.get("processing_complete"):
                    final_status = "pending"  # processing ok, publishing timeout
                else:
                    final_status = "uploaded"  # masih processing

            # --- Update manifest row ---
            if manifest_row is not None:
                manifest_row["fb_upload_status"] = final_status
                manifest_row["fb_publish_status_raw"] = status_result.get("status", {})
                manifest_row["fb_video_id"] = video_id
                manifest_row["fb_post_id"] = post_id
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
                "status": final_status,
                "filename": os.path.basename(video_path),
                "video_id": video_id,
                "post_id": post_id,
                "mode": video_state,
                "scheduled_at": (
                    scheduled_at.isoformat() if scheduled_at else None
                ),
                "api_response": finish_result,
                "facebook_status": status_result.get("status", {}),
            }
            upload_results.append(row_result)

            if final_status == "published":
                print(f"   ✅ Reel benar-benar published. Video ID: {video_id}")
            elif final_status == "scheduled":
                sched_str = scheduled_at.strftime('%Y-%m-%d %H:%M:%S %Z') if scheduled_at else '-'
                print(
                    f"   ✅ Reel berhasil dijadwalkan untuk {sched_str}. Video ID: {video_id}"
                )
            elif final_status == "pending":
                print(
                    f"   ⏳ Reel diterima Meta, menunggu publishing selesai. Video ID: {video_id}"
                )
            else:
                print(
                    f"   ⚠️ Reel diterima Meta dengan status '{final_status}'. Video ID: {video_id}"
                )

        except Exception as e:
            err = str(e)

            if manifest_row is not None:
                manifest_row["fb_upload_status"] = "failed"
                manifest_row["fb_upload_error"] = err
                if video_id:
                    manifest_row["fb_video_id"] = video_id

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
