"""
clipping.metadata — QA Metadata Preview & Normalization

Maps to the QA Metadata Preview cell in the notebook.
"""

import json


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _normalize_spaces(text):
    return " ".join(str(text or "").split()).strip()


def _trim_title(text, max_len=100):
    text = _normalize_spaces(text)
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0].strip()
    return cut if cut else text[:max_len].strip()


def _normalize_hashtags(text, min_tags=2, max_tags=3):
    parts = _normalize_spaces(text).split()
    clean = []
    seen = set()

    for p in parts:
        if not p:
            continue
        if not p.startswith("#"):
            p = "#" + p.lstrip("#")
        key = p.lower()
        if key not in seen:
            seen.add(key)
            clean.append(p)
        if len(clean) >= max_tags:
            break

    return " ".join(clean), len(clean)


def _normalize_keyword_tags(tags, min_items=5, max_items=8):
    if not isinstance(tags, list):
        tags = []
    out = []
    seen = set()

    for t in tags:
        x = _normalize_spaces(t)
        if not x:
            continue
        key = x.lower()
        if key not in seen:
            seen.add(key)
            out.append(x)
        if len(out) >= max_items:
            break

    return out


def _build_youtube_description(hook, context, hashtags):
    parts = [
        _normalize_spaces(hook),
        _normalize_spaces(context),
        _normalize_spaces(hashtags),
    ]
    return "\n\n".join([p for p in parts if p]).strip()


def _build_tiktok_caption(caption, hashtags):
    caption = _normalize_spaces(caption)
    hashtags = _normalize_spaces(hashtags)
    if caption and hashtags:
        return f"{caption}\n{hashtags}"
    return caption or hashtags


def _looks_indonesian(text):
    text = f" {_normalize_spaces(text).lower()} "
    indikator = [
        " yang ", " dan ", " untuk ", " dengan ", " karena ", " adalah ",
        " bisa ", " tidak ", " lebih ", " dalam ", " pada ", " agar ",
        " dari ", " ini ", " itu ", " juga ", " kalau ", " saat ",
        " tentang ", " bikin ", " banget ", " jadi ", " sudah ",
    ]
    return any(w in text for w in indikator)


# ==============================================================================
# MAIN API
# ==============================================================================

def normalize_and_validate(hasil_json: list[dict]) -> list[dict]:
    """
    Normalize and enrich metadata fields, add *_final fields.

    Mutates items in-place and returns sorted list.
    """
    valid_items = []
    laporan = []
    semua_warning = []
    for item in hasil_json:
        if not isinstance(item, dict):
            print(f"⚠️ Melewati item metadata yang tidak valid (bukan dict): {type(item)}")
            continue

        rank = item.get("rank", "?")

        item["title_indonesia"] = _trim_title(item.get("title_indonesia", ""))
        item["title_inggris"] = _trim_title(item.get("title_inggris", ""))
        item["description_hook"] = _normalize_spaces(item.get("description_hook", ""))
        item["description_context"] = _normalize_spaces(item.get("description_context", ""))
        item["tiktok_title_id"] = _normalize_spaces(item.get("tiktok_title_id", ""))
        item["tiktok_caption_id"] = _normalize_spaces(item.get("tiktok_caption_id", ""))
        item["tiktok_caption"] = _normalize_spaces(item.get("tiktok_caption", ""))
        item["keyword_tags"] = _normalize_keyword_tags(item.get("keyword_tags", []))

        hastag_clean, hashtag_count = _normalize_hashtags(item.get("hastag", ""))
        item["hastag"] = hastag_clean

        # Enriched fields
        item["youtube_title_final"] = item["title_inggris"]
        item["youtube_description_final"] = _build_youtube_description(
            item.get("description_hook", ""),
            item.get("description_context", ""),
            item.get("hastag", ""),
        )
        item["youtube_tags_final"] = item.get("keyword_tags", [])

        # TikTok EN (existing behavior)
        item["tiktok_caption_final"] = _build_tiktok_caption(
            item.get("tiktok_caption", ""),
            item.get("hastag", ""),
        )

        # TikTok ID (new fields)
        item["tiktok_title_id_final"] = item.get("tiktok_title_id", "") or item.get("title_indonesia", "")
        item["tiktok_caption_id_final"] = _build_tiktok_caption(
            item.get("tiktok_caption_id", ""),
            item.get("hastag", ""),
        )

        # --- Warnings ---
        warning = []
        if not item["title_indonesia"]:
            warning.append("title_indonesia kosong")
        if len(item["title_indonesia"]) > 100:
            warning.append("title_indonesia > 100 karakter")

        if not item["title_inggris"]:
            warning.append("title_inggris kosong")
        if len(item["title_inggris"]) > 100:
            warning.append("title_inggris > 100 karakter")

        if hashtag_count < 2 or hashtag_count > 3:
            warning.append("jumlah hashtag bukan 2-3")

        if not item["description_hook"]:
            warning.append("description_hook kosong")
        if not item["description_context"]:
            warning.append("description_context kosong")
        if len(item["keyword_tags"]) < 5:
            warning.append("keyword_tags terlalu sedikit")

        if not item["tiktok_title_id"]:
            warning.append("tiktok_title_id kosong")
        if not item["tiktok_caption_id"]:
            warning.append("tiktok_caption_id kosong")
        if not item["tiktok_caption"]:
            warning.append("tiktok_caption kosong")

        if _looks_indonesian(item["title_inggris"]):
            warning.append("title_inggris terdeteksi bukan English penuh")
        if _looks_indonesian(item["description_hook"]):
            warning.append("description_hook terdeteksi bukan English penuh")
        if _looks_indonesian(item["description_context"]):
            warning.append("description_context terdeteksi bukan English penuh")
        if _looks_indonesian(item["tiktok_caption"]):
            warning.append("tiktok_caption terdeteksi bukan English penuh")

        if item["tiktok_title_id"] and not _looks_indonesian(item["tiktok_title_id"]):
            warning.append("tiktok_title_id terdeteksi bukan Bahasa Indonesia")
        if item["tiktok_caption_id"] and not _looks_indonesian(item["tiktok_caption_id"]):
            warning.append("tiktok_caption_id terdeteksi bukan Bahasa Indonesia")

        laporan.append({
            "rank": rank,
            "durasi": round(float(item.get("end_time", 0)) - float(item.get("start_time", 0)), 2),
            "title_indonesia": item["title_indonesia"],
            "title_inggris": item["title_inggris"],
            "tiktok_title_id": item["tiktok_title_id"],
            "hashtags": item["hastag"],
            "warnings": " | ".join(warning) if warning else "OK",
        })

        if warning:
            semua_warning.append((rank, warning))
            
        valid_items.append(item)

    laporan = sorted(laporan, key=lambda x: x["rank"])
    return sorted(valid_items, key=lambda x: x.get("rank", 9999))


def print_preview(hasil_json: list[dict]) -> None:
    """Print a human-readable metadata preview to stdout."""
    print("✅ Preview metadata siap.")
    print("Field tambahan yang dibuat:")
    print("- youtube_title_final")
    print("- youtube_description_final")
    print("- youtube_tags_final")
    print("- tiktok_caption_final")
    print("- tiktok_title_id_final")
    print("- tiktok_caption_id_final")
    print()

    print("===== PREVIEW DETAIL PER KLIP =====")
    for item in hasil_json:
        print(f"\n--- Rank {item['rank']} ---")
        print(f"Title ID          : {item['title_indonesia']}")
        print(f"Title EN          : {item['title_inggris']}")
        print(f"TikTok Title ID   : {item.get('tiktok_title_id_final', '')}")
        print(f"Hashtag           : {item['hastag']}")
        print(f"Hook Desc         : {item['description_hook']}")
        print(f"Ctx Desc          : {item['description_context']}")
        print(f"YT Desc           : {item['youtube_description_final']}")
        print(f"YT Tags           : {item['youtube_tags_final']}")
        print(f"TikTok EN         : {item['tiktok_caption_final']}")
        print(f"TikTok Caption ID : {item.get('tiktok_caption_id_final', '')}")


def save_metadata_preview(hasil_json: list[dict], path: str = "metadata_preview.json") -> None:
    """Save normalized metadata to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hasil_json, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Disimpan ke {path}")