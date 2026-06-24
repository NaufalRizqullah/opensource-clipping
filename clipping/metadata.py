"""
clipping.metadata - QA Metadata Preview & Normalization

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


def _looks_non_english(text):
    """Rough guard: flag text that still contains common non-English filler words."""
    text = f" {_normalize_spaces(text).lower()} "
    indicators = [
        " yang ", " dan ", " untuk ", " dengan ", " karena ", " adalah ",
        " bisa ", " tidak ", " lebih ", " dalam ", " pada ", " agar ",
        " dari ", " ini ", " itu ", " juga ", " kalau ", " saat ",
        " tentang ", " bikin ", " banget ", " jadi ", " sudah ",
    ]
    return any(w in text for w in indicators)


# ==============================================================================
# MAIN API
# ==============================================================================

def normalize_and_validate(hasil_json: list[dict]) -> list[dict]:
    """
    Normalize and enrich metadata fields, add *_final fields.

    Mutates items in-place and returns the sorted list.
    """
    valid_items = []
    for item in hasil_json:
        if not isinstance(item, dict):
            print(f"⚠️ Skipping invalid metadata item (not a dict): {type(item)}")
            continue

        # 1. FLATTENING: if the model put everything under a "metadata" key, bring it up
        if isinstance(item.get("metadata"), dict):
            for k, v in item["metadata"].items():
                if k not in item:
                    item[k] = v

        # 2. ALIASING: handle field-name variations (especially for non-Gemini providers)
        rank = item.get("rank") or item.get("peringkat") or item.get("no") or "?"
        item["rank"] = rank
        item["viral_score"] = item.get("viral_score", 0)

        # Timing aliases
        it_st = item.get("start_time") or item.get("timing_klip_start") or item.get("clip_start") or item.get("start")
        it_en = item.get("end_time") or item.get("timing_klip_end") or item.get("clip_end") or item.get("end")
        item["start_time"] = float(it_st) if it_st is not None else 0.0
        item["end_time"] = float(it_en) if it_en is not None else 0.0

        # Hook aliases (some providers return hook as a string + hook_start_time at root)
        if isinstance(item.get("hook"), str) and "hook_start_time" in item:
            item["hook"] = {
                "text": item["hook"],
                "start_time": item.get("hook_start_time", item["start_time"]),
                "end_time": item.get("hook_end_time", item["end_time"]),
            }

        # Ensure hook exists as a dict for later code
        if not isinstance(item.get("hook"), dict):
            item["hook"] = {"text": str(item.get("hook", "")), "start_time": item["start_time"], "end_time": item["start_time"] + 3.0}

        item["title_inggris"] = _trim_title(item.get("title_inggris", ""))
        item["description_hook"] = _normalize_spaces(item.get("description_hook", ""))
        item["description_context"] = _normalize_spaces(item.get("description_context", ""))
        item["hastag"] = _normalize_spaces(item.get("hastag") or item.get("hashtag") or "")
        item["tiktok_caption"] = _normalize_spaces(item.get("tiktok_caption", ""))
        item["keyword_tags"] = _normalize_keyword_tags(item.get("keyword_tags", []))

        hastag_clean, hashtag_count = _normalize_hashtags(item["hastag"])
        item["hastag"] = hastag_clean

        # Enriched fields
        item["youtube_title_final"] = item["title_inggris"]
        item["youtube_description_final"] = _build_youtube_description(
            item.get("description_hook", ""),
            item.get("description_context", ""),
            item.get("hastag", ""),
        )
        item["youtube_tags_final"] = item.get("keyword_tags", [])
        item["tiktok_caption_final"] = _build_tiktok_caption(
            item.get("tiktok_caption", ""),
            item.get("hastag", ""),
        )

        # --- Warnings ---
        warning = []
        if not item["title_inggris"]:
            warning.append("title_inggris empty")
        if len(item["title_inggris"]) > 100:
            warning.append("title_inggris > 100 characters")

        if hashtag_count < 2 or hashtag_count > 3:
            warning.append("hashtag count is not 2-3")

        if not item["description_hook"]:
            warning.append("description_hook empty")
        if not item["description_context"]:
            warning.append("description_context empty")
        if len(item["keyword_tags"]) < 5:
            warning.append("too few keyword_tags")
        if not item["tiktok_caption"]:
            warning.append("tiktok_caption empty")

        if _looks_non_english(item["title_inggris"]):
            warning.append("title_inggris detected as not fully English")
        if _looks_non_english(item["description_hook"]):
            warning.append("description_hook detected as not fully English")
        if _looks_non_english(item["description_context"]):
            warning.append("description_context detected as not fully English")
        if _looks_non_english(item["tiktok_caption"]):
            warning.append("tiktok_caption detected as not fully English")

        item["_warnings_temp"] = " | ".join(warning) if warning else "OK"
        valid_items.append(item)

    # Sort by viral_score (descending)
    valid_items = sorted(valid_items, key=lambda x: x.get("viral_score", 0), reverse=True)

    # Re-assign rank to be purely sequential
    for idx, item in enumerate(valid_items):
        item["rank"] = idx + 1
        item.pop("_warnings_temp", None)

    return valid_items


def print_preview(hasil_json: list[dict]) -> None:
    """Print a human-readable metadata preview to stdout."""
    print("✅ Metadata preview ready.")
    print("Additional fields created:")
    print("- youtube_title_final")
    print("- youtube_description_final")
    print("- youtube_tags_final")
    print("- tiktok_caption_final")
    print()

    print("===== DETAILED PREVIEW PER CLIP =====")
    for item in hasil_json:
        print(f"\n--- Rank {item['rank']} (Viral Score: {item.get('viral_score', '?')}) ---")
        print(f"Title             : {item['title_inggris']}")
        print(f"Hashtag           : {item['hastag']}")
        print(f"Hook Desc         : {item['description_hook']}")
        print(f"Ctx Desc          : {item['description_context']}")
        print(f"YT Desc           : {item['youtube_description_final']}")
        print(f"YT Tags           : {item['youtube_tags_final']}")
        print(f"TikTok Caption    : {item['tiktok_caption_final']}")


def save_metadata_preview(hasil_json: list[dict], path: str = "metadata_preview.json") -> None:
    """Save normalized metadata to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hasil_json, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Saved to {path}")
