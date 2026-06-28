"""
youtube_uploader.uploader — Core YouTube Upload & Scheduling Logic
"""

import os
import json
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]


# ------------------------------------------------------------------------------
# HELPER AUTH
# ------------------------------------------------------------------------------
def get_youtube_service(token_file: str):
    if not os.path.exists(token_file):
        raise FileNotFoundError(
            f"{token_file} tidak ditemukan. "
            "Buat/generate dulu token_file di dalam folder credentials."
        )

    creds = Credentials.from_authorized_user_file(token_file, YOUTUBE_SCOPES)

    if not creds.valid:
        if creds.expired and creds.refresh_token:
            print("🔄 Access token expired, mencoba refresh token...")
            creds.refresh(Request())
            with open(token_file, "w", encoding="utf-8") as tf:
                tf.write(creds.to_json())
            print("✅ Token berhasil di-refresh.")
        else:
            raise RuntimeError(
                "Token tidak valid dan tidak punya refresh_token. "
                "Buat ulang token file dari login laptop/PC."
            )

    return build("youtube", "v3", credentials=creds)


# ------------------------------------------------------------------------------
# HELPER WAKTU / SCHEDULER
# ------------------------------------------------------------------------------
def parse_local_datetime(dt_text, tz_name):
    tz = ZoneInfo(tz_name)
    dt = datetime.strptime(dt_text, "%Y-%m-%d %H:%M")
    return dt.replace(tzinfo=tz)


def parse_rfc3339_to_local(dt_text, tz_name):
    if not dt_text:
        return None
    try:
        dt_utc = datetime.fromisoformat(dt_text.replace("Z", "+00:00"))
        return dt_utc.astimezone(ZoneInfo(tz_name))
    except Exception:
        return None


def to_rfc3339_utc(dt_local):
    dt_utc = dt_local.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_latest_scheduled_publish_time(youtube, tz_name="Asia/Makassar", max_pages=10):
    tz = ZoneInfo(tz_name)
    now_local = datetime.now(tz)

    print("🔎 Mengecek scheduled terakhir di channel YouTube...")

    channel_resp = youtube.channels().list(part="contentDetails", mine=True).execute()
    channel_items = channel_resp.get("items", [])
    if not channel_items:
        print("⚠️ Tidak bisa menemukan channel milik akun ini.")
        return None

    uploads_playlist_id = (
        channel_items[0]
        .get("contentDetails", {})
        .get("relatedPlaylists", {})
        .get("uploads")
    )

    if not uploads_playlist_id:
        print("⚠️ Uploads playlist tidak ditemukan.")
        return None

    latest_dt = None
    page_token = None

    for _ in range(max_pages):
        playlist_resp = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=page_token
        ).execute()

        playlist_items = playlist_resp.get("items", [])
        if not playlist_items:
            break

        video_ids = []
        for row in playlist_items:
            video_id = row.get("contentDetails", {}).get("videoId")
            if video_id:
                video_ids.append(video_id)

        if video_ids:
            videos_resp = youtube.videos().list(
                part="status,snippet",
                id=",".join(video_ids),
                maxResults=50
            ).execute()

            for video in videos_resp.get("items", []):
                status = video.get("status", {})
                publish_at = status.get("publishAt")
                privacy_status = status.get("privacyStatus")

                if not publish_at:
                    continue

                dt_local = parse_rfc3339_to_local(publish_at, tz_name)
                if dt_local is None:
                    continue

                if dt_local <= now_local:
                    continue

                if privacy_status != "private":
                    continue

                if latest_dt is None or dt_local > latest_dt:
                    latest_dt = dt_local

        page_token = playlist_resp.get("nextPageToken")
        if not page_token:
            break

    if latest_dt:
        print(f"✅ Scheduled terakhir ditemukan: {latest_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        print("ℹ️ Belum ada video scheduled di masa depan. Pakai fallback schedule default.")

    return latest_dt


def get_first_publish_time(
    youtube=None,
    tz_name="Asia/Makassar",
    start_local=None,
    interval_hours=8
):
    tz = ZoneInfo(tz_name)

    if start_local:
        first_dt = parse_local_datetime(start_local, tz_name)
        print(f"🗓️ Pakai start_local manual: {first_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        return first_dt

    if youtube is not None:
        latest_scheduled = get_latest_scheduled_publish_time(youtube, tz_name)
        if latest_scheduled is not None:
            first_dt = latest_scheduled + timedelta(hours=interval_hours)
            print(f"🗓️ Jadwal pertama baru dari YouTube: {first_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            return first_dt

    now_local = datetime.now(tz)
    first_dt = now_local + timedelta(minutes=30)
    first_dt = first_dt.replace(minute=0, second=0, microsecond=0)
    if first_dt <= now_local:
        first_dt = first_dt + timedelta(hours=1)

    print(f"🗓️ Fallback jadwal pertama: {first_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    return first_dt


def build_schedule_times(
    count,
    youtube=None,
    tz_name="Asia/Makassar",
    interval_hours=8,
    start_local=None
):
    if count <= 0:
        return []
    first_dt = get_first_publish_time(youtube, tz_name, start_local, interval_hours)
    return [first_dt + timedelta(hours=i * interval_hours) for i in range(count)]


# ------------------------------------------------------------------------------
# HELPER FILE / MANIFEST
# ------------------------------------------------------------------------------
def load_json_file(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_nonempty_file(path):
    return bool(path) and os.path.exists(path) and os.path.isfile(path) and os.path.getsize(path) > 0


def normalize_text(text):
    return " ".join(str(text or "").split()).strip()


def normalize_tags(tags, max_items=8):
    if not isinstance(tags, list):
        return []

    out = []
    seen = set()

    for tag in tags:
        t = normalize_text(tag)
        if not t:
            continue

        key = t.lower()
        if key in seen:
            continue

        seen.add(key)
        out.append(t)
        if len(out) >= max_items:
            break

    return out


def get_upload_candidates(render_manifest):
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


def format_http_error(e):
    status = getattr(getattr(e, "resp", None), "status", None)
    try:
        content = e.content.decode("utf-8") if getattr(e, "content", None) else str(e)
    except Exception:
        content = str(e)
    return f"HTTP {status or 'ERR'}: {content}"


# ------------------------------------------------------------------------------
# HELPER YOUTUBE API
# ------------------------------------------------------------------------------
def set_custom_thumbnail(youtube, video_id, thumbnail_path, max_retries=3):
    if not is_nonempty_file(thumbnail_path):
        return {"thumbnail_set": False, "thumbnail_error": "thumbnail file not found"}

    for attempt in range(1, max_retries + 1):
        try:
            media = MediaFileUpload(thumbnail_path, resumable=False)
            youtube.thumbnails().set(videoId=video_id, media_body=media).execute()
            return {"thumbnail_set": True, "thumbnail_error": None}
        except Exception as e:
            err = format_http_error(e) if isinstance(e, HttpError) else str(e)
            if attempt < max_retries:
                print(f"   ⚠️ Gagal set thumbnail (attempt {attempt}/{max_retries}), retry...")
                time.sleep(5)
            else:
                return {"thumbnail_set": False, "thumbnail_error": err}


def upload_video_to_youtube(
    youtube,
    item,
    publish_at_local=None,
    category_id="22",
    default_language="en",
    privacy_status="private",
    chunk_size=8 * 1024 * 1024
):
    video_path = item["video_path"]
    thumbnail_path = item.get("thumbnail_path")

    title = (
        item.get("youtube_title_final")
        or item.get("title_inggris")
        or item.get("title_indonesia")
        or f"Clip Rank {item.get('rank', '?')}"
    )
    title = normalize_text(title)[:100]

    description = normalize_text(item.get("youtube_description_final", ""))
    tags = normalize_tags(item.get("youtube_tags_final", []))

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id,
            "defaultLanguage": default_language
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False
        }
    }

    publish_at_rfc3339 = None
    if privacy_status == "private" and publish_at_local is not None:
        publish_at_rfc3339 = to_rfc3339_utc(publish_at_local)
        body["status"]["publishAt"] = publish_at_rfc3339

    media = MediaFileUpload(video_path, chunksize=chunk_size, resumable=True)

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
        notifySubscribers=False
    )

    response = None
    print(f"   ⬆️ Uploading: {os.path.basename(video_path)}")

    while response is None:
        status, response = request.next_chunk()
        if status:
            progress = int(status.progress() * 100)
            print(f"   ... {progress}%")

    video_id = response["id"]
    thumb_result = {"thumbnail_set": False, "thumbnail_error": None}

    # TODO: uncomment this after we have a good thumbnail system
    # Right now dont upload a thumbnail. let youtube handle it
    # if is_nonempty_file(thumbnail_path):
    #     print("   🖼️ Setting thumbnail...")
    #     thumb_result = set_custom_thumbnail(youtube, video_id, thumbnail_path)

    return {
        "video_id": video_id,
        "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
        "scheduled_publish_local": publish_at_local.strftime("%Y-%m-%d %H:%M:%S %Z") if publish_at_local else None,
        "scheduled_publish_utc": publish_at_rfc3339,
        "thumbnail_set": thumb_result["thumbnail_set"],
        "thumbnail_error": thumb_result["thumbnail_error"],
        "uploaded_title": title,
        "uploaded_description": description,
        "uploaded_tags": tags
    }


# ------------------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------------------
def upload_manifest_to_youtube(
    token_file: str,
    manifest_file: str = "render_manifest.json",
    result_file: str = "youtube_upload_results.json",
    updated_manifest_file: str = "render_manifest_uploaded.json",
    tz_name: str = "Asia/Makassar",
    interval_hours: int = 2,
    start_local: str = None,
    test_mode: bool = False,
    safety_config: dict = None,
    skip_approval: bool = False,
):
    # ── Load Safety Config ────────────────────────────────────────
    from . import safety as safety_mod

    if safety_config is None:
        safety_config = safety_mod.load_safety_config()

    safety_mod.print_safety_summary(safety_config, tz_name)

    # ── Daily Limit Check ────────────────────────────────────────
    is_allowed, uploads_today, max_per_day = safety_mod.check_daily_limit(
        safety_config, tz_name
    )
    if not is_allowed:
        print(
            f"🚫 Upload dibatalkan: sudah {uploads_today}/{max_per_day} upload hari ini. "
            f"Coba lagi besok atau ubah max_upload_per_day di upload_safety.json."
        )
        return []

    # ── Load Manifest ────────────────────────────────────────────
    render_manifest = load_json_file(manifest_file, default=[])
    if not render_manifest:
        print(f"⚠️ {manifest_file} kosong / tidak ditemukan.")
        return []

    candidates = get_upload_candidates(render_manifest)
    if not candidates:
        print("⚠️ Tidak ada item yang siap diupload.")
        return []

    pending_items = []
    for item in candidates:
        if item.get("youtube_video_id") and item.get("youtube_upload_status") == "uploaded":
            print(f"⏭️ Skip Rank {item.get('rank')} karena sudah pernah diupload.")
            continue
        pending_items.append(item)

    if test_mode and pending_items:
        pending_items = pending_items[:1]
        print("🧪 Mode test aktif: hanya upload 1 item pertama.")

    # ── Safety: Per-Run Limit ────────────────────────────────────
    if not test_mode:
        pending_items = safety_mod.limit_pending_items(pending_items, safety_config)

    if not pending_items:
        print("⚠️ Semua item success sudah pernah diupload.")
        return []

    # ── Safety: Enforce Minimum Interval ─────────────────────────
    interval_hours = safety_mod.enforce_min_interval(interval_hours, safety_config)

    youtube = get_youtube_service(token_file)

    # ── Safety: Queue Limit Check ────────────────────────────────
    queue_ok, queue_count, queue_max = safety_mod.check_queue_limit(
        youtube, safety_config, tz_name
    )
    if not queue_ok:
        print(
            f"🚫 Upload dibatalkan: scheduled queue penuh ({queue_count}/{queue_max}). "
            f"Tunggu video terjadwal dipublish dulu, atau ubah max_scheduled_queue "
            f"di upload_safety.json."
        )
        return []
    if queue_count > 0:
        print(f"📋 Scheduled queue saat ini: {queue_count}/{queue_max}")

    schedule_times = build_schedule_times(
        count=len(pending_items),
        youtube=youtube,
        tz_name=tz_name,
        interval_hours=interval_hours,
        start_local=start_local
    )

    upload_results = []
    updated_manifest = deepcopy(render_manifest)

    print(f"🚀 Mulai upload {len(pending_items)} clip ke YouTube...")

    require_approval = safety_config.get("require_manual_approval", True) and not skip_approval

    for item, publish_at_local in zip(pending_items, schedule_times):
        rank = item.get("rank")
        manifest_row = get_manifest_row_by_rank(updated_manifest, rank)

        print(f"\\n=== Upload Rank {rank} ===")
        print(f"Judul  : {item.get('youtube_title_final')}")
        print(f"Video  : {item.get('video_path')}")
        print(f"Jadwal : {publish_at_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        # ── Safety: Manual Approval Gate ─────────────────────────
        if require_approval:
            if not safety_mod.prompt_manual_approval(item, publish_at_local):
                print(f"⏭️ Rank {rank} dilewati (tidak di-approve).")
                if manifest_row is not None:
                    manifest_row["youtube_upload_status"] = "skipped_approval"
                upload_results.append({"rank": rank, "status": "skipped_approval"})
                continue

        # ── Re-check daily limit before each upload ──────────────
        is_still_ok, up_today, max_day = safety_mod.check_daily_limit(
            safety_config, tz_name
        )
        if not is_still_ok:
            print(
                f"🚫 Daily limit tercapai ({up_today}/{max_day}). "
                f"Sisa item di-skip."
            )
            break

        try:
            result = upload_video_to_youtube(youtube, item, publish_at_local)

            if manifest_row is not None:
                manifest_row["youtube_upload_status"] = "uploaded"
                manifest_row["youtube_video_id"] = result["video_id"]
                manifest_row["youtube_url"] = result["youtube_url"]
                manifest_row["youtube_scheduled_publish_local"] = result["scheduled_publish_local"]
                manifest_row["youtube_scheduled_publish_utc"] = result["scheduled_publish_utc"]
                manifest_row["youtube_thumbnail_set"] = result["thumbnail_set"]
                manifest_row["youtube_thumbnail_error"] = result["thumbnail_error"]
                manifest_row["youtube_uploaded_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                manifest_row["youtube_upload_error"] = None

            row_result = {"rank": rank, "status": "uploaded", **result}
            upload_results.append(row_result)

            # ── Safety: Record Upload ────────────────────────────
            safety_mod.record_upload(
                safety_config["upload_log_file"],
                result["video_id"],
                result.get("uploaded_title", ""),
                tz_name,
            )

            print(f"✅ Upload sukses. Video ID: {result['video_id']}")
            print(f"🔗 {result['youtube_url']}")

        except Exception as e:
            err = format_http_error(e) if isinstance(e, HttpError) else str(e)

            if manifest_row is not None:
                manifest_row["youtube_upload_status"] = "failed"
                manifest_row["youtube_upload_error"] = err

            upload_results.append({"rank": rank, "status": "failed", "error": err})
            print(f"❌ Upload gagal untuk Rank {rank}: {err}")

        save_json_file(result_file, upload_results)
        save_json_file(updated_manifest_file, updated_manifest)

    print(f"\\n💾 Hasil upload disimpan ke: {result_file}")
    print(f"💾 Manifest terupdate disimpan ke: {updated_manifest_file}")

    return upload_results
