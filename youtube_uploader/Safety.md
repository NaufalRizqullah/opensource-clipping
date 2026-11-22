# YouTube Uploader Safety Features

Dokumen ini menyimpan kode dan konteks untuk fitur **Safety Checklist** dan **Daily Limits** (pembatasan harian) yang pernah diterapkan pada `youtube_uploader`. 

Fitur ini sebelumnya ditambahkan untuk meminimalisir risiko *channel banned* atau *spam* saat melakukan unggahan otomatis secara massal ke YouTube. Fitur ini sengaja dicabut (di-*revert*) agar *uploader* berjalan 100% secara otomatis, namun kode di bawah ini disimpan jika suatu saat ingin diaktifkan kembali.

File konfigurasi lama `upload_safety.json` telah dihapus karena tidak digunakan.
Contoh kode di bawah hanya dokumentasi historis, bukan fitur aktif uploader.
Launcher berada di `scripts/run_upload.py`; opsi lama `--safety-config` dan
`--no-approval` akan berhenti dengan pesan error sebelum upload.

## 1. File `safety.py`
Kode di bawah ini sebelumnya diletakkan di `youtube_uploader/safety.py`. File ini berisi modul logika yang menangani batas harian (daily limit), batas antrean tayang (queue limit), jeda antar unggahan (interval minimum), dan *prompt* konfirmasi manual.

```python
import os
import json
from datetime import datetime, timezone, timedelta
from typing import Optional

def _get_tz(tz_name: str):
    import pytz
    return pytz.timezone(tz_name)

def load_safety_config(config_path: str) -> dict:
    default_config = {
        "max_upload_per_day": 3,
        "max_upload_per_run": 2,
        "interval_hours_min": 2,
        "max_scheduled_queue": 15,
        "require_manual_approval": True,
        "upload_log_file": "upload_history.json"
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
                default_config.update(user_cfg)
        except Exception as e:
            print(f"⚠️ Gagal membaca {config_path}: {e}")
            
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=4)
        
    return default_config

def load_upload_history(log_file: str) -> list:
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_upload_history(log_file: str, history: list):
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

def record_upload(log_file: str, video_id: str, title: str, tz_name: str = "Asia/Makassar"):
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

def count_uploads_today(log_file: str, tz_name: str = "Asia/Makassar") -> int:
    history = load_upload_history(log_file)
    tz = _get_tz(tz_name)
    today_str = datetime.now(tz).strftime("%Y-%m-%d")
    return sum(1 for entry in history if entry.get("date") == today_str)

def check_daily_limit(safety_config: dict, tz_name: str = "Asia/Makassar") -> tuple[bool, int, int]:
    log_file = safety_config["upload_log_file"]
    max_per_day = safety_config["max_upload_per_day"]
    uploads_today = count_uploads_today(log_file, tz_name)
    return uploads_today < max_per_day, uploads_today, max_per_day

def enforce_min_interval(requested_hours: int, safety_config: dict) -> int:
    min_hours = safety_config["interval_hours_min"]
    effective = max(requested_hours, min_hours)
    if effective != requested_hours:
        print(f"🛡️ Interval diubah dari {requested_hours}h → {effective}h")
    return effective

def limit_pending_items(items: list, safety_config: dict) -> list:
    max_per_run = safety_config["max_upload_per_run"]
    if len(items) > max_per_run:
        print(f"🛡️ Membatasi upload: {len(items)} item → {max_per_run} item per run")
        return items[:max_per_run]
    return items

def check_queue_limit(youtube, safety_config: dict, tz_name: str = "Asia/Makassar") -> tuple[bool, int, int]:
    from .uploader import get_latest_scheduled_publish_time, parse_rfc3339_to_local
    max_queue = safety_config["max_scheduled_queue"]
    try:
        tz = _get_tz(tz_name)
        now_local = datetime.now(tz)
        channel_resp = youtube.channels().list(part="contentDetails", mine=True).execute()
        channel_items = channel_resp.get("items", [])
        if not channel_items:
            return True, 0, max_queue
        
        uploads_playlist_id = channel_items[0].get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
        if not uploads_playlist_id:
            return True, 0, max_queue
            
        scheduled_count = 0
        page_token = None
        for _ in range(5):
            playlist_resp = youtube.playlistItems().list(part="contentDetails", playlistId=uploads_playlist_id, maxResults=50, pageToken=page_token).execute()
            playlist_items = playlist_resp.get("items", [])
            if not playlist_items:
                break
                
            video_ids = [row.get("contentDetails", {}).get("videoId") for row in playlist_items if row.get("contentDetails", {}).get("videoId")]
            if video_ids:
                videos_resp = youtube.videos().list(part="status", id=",".join(video_ids), maxResults=50).execute()
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
        return True, 0, max_queue

def prompt_manual_approval(item: dict, publish_at_local=None) -> bool:
    title = item.get("youtube_title_final") or item.get("title_inggris") or f"Clip Rank {item.get('rank', '?')}"
    video_path = item.get("video_path", "?")
    
    print("\n" + "=" * 60)
    print("🛡️ MANUAL APPROVAL REQUIRED")
    print("=" * 60)
    print(f"  Judul    : {title}")
    print(f"  File     : {os.path.basename(video_path)}")
    if publish_at_local:
        print(f"  Jadwal   : {publish_at_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("=" * 60)
    print()
    print("⚠️  CHECKLIST SEBELUM APPROVE:")
    print("  □ Sudah review video secara manual?")
    print("  □ Ada narasi/analisis/voice-over kamu sendiri?")
    print("  □ Judul & thumbnail BUKAN meniru creator asli?")
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

def print_safety_summary(safety_config: dict, tz_name: str = "Asia/Makassar") -> None:
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
```

## 2. Modifikasi pada `uploader.py`
Agar fitur keamanan di atas bekerja, fungsi `upload_manifest_to_youtube` di dalam `uploader.py` dimodifikasi sebagai berikut:

```python
def upload_manifest_to_youtube(
    token_file: str,
    manifest_file: str = "render_manifest.json",
    result_file: str = "youtube_upload_results.json",
    updated_manifest_file: str = "render_manifest_uploaded.json",
    tz_name: str = "Asia/Makassar",
    interval_hours: int = 2,
    start_local: str = None,
    test_mode: bool = False,
    safety_config: dict = None,      # TAMBAHAN
    skip_approval: bool = False,     # TAMBAHAN
):
    from . import safety as safety_mod

    if safety_config is None:
        safety_config = safety_mod.load_safety_config()

    safety_mod.print_safety_summary(safety_config, tz_name)

    # Cek Limit Harian
    is_allowed, uploads_today, max_per_day = safety_mod.check_daily_limit(safety_config, tz_name)
    if not is_allowed:
        print("🚫 Upload dibatalkan (Daily Limit).")
        return []

    # ... (Muat Manifest) ...

    # Batasi Item per Run
    if not test_mode:
        pending_items = safety_mod.limit_pending_items(pending_items, safety_config)

    # Batasi Interval Minimum
    interval_hours = safety_mod.enforce_min_interval(interval_hours, safety_config)

    youtube = get_youtube_service(token_file)

    # Cek Queue YouTube
    queue_ok, queue_count, queue_max = safety_mod.check_queue_limit(youtube, safety_config, tz_name)
    if not queue_ok:
        print("🚫 Upload dibatalkan: scheduled queue penuh.")
        return []

    # ... (Buat Jadwal) ...

    require_approval = safety_config.get("require_manual_approval", True) and not skip_approval

    for item, publish_at_local in zip(pending_items, schedule_times):
        rank = item.get("rank")
        
        # Minta Konfirmasi Manual sebelum upload (Tampilkan Checklist)
        if require_approval:
            if not safety_mod.prompt_manual_approval(item, publish_at_local):
                print(f"⏭️ Rank {rank} dilewati (tidak di-approve).")
                continue

        # ... (Proses Upload) ...
        
        # Catat History Upload
        if success:
            safety_mod.record_upload(
                safety_config["upload_log_file"],
                result["video_id"],
                result.get("uploaded_title", ""),
                tz_name,
            )
```

**Instruksi:** Jika di masa depan ingin menerapkan sistem keamanan ini kembali, cukup buat file `safety.py` dari _snippet_ pertama dan sisipkan _hook_ pada _snippet_ kedua ke dalam `upload_manifest_to_youtube` di `uploader.py`.
