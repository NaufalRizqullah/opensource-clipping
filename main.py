#!/usr/bin/env python3
"""
OpenSource Clipping - AI Auto-Clipper & Teaser Generator

Usage:
    python main.py --url "https://..."      # run with required URL
    python main.py --url "https://..." --clips 5 --ratio 16:9
    python main.py --help                   # show all available options
"""

import sys

from clipping.config import build_config


def main():
    cfg = build_config(sys.argv[1:])

    version = "1.0.7"

    # ── Story Clip Mode ──────────────────────────────────────────────
    if getattr(cfg, "story_mode", False):
        from clipping.story_runner import run_story_pipeline

        print("=" * 70)
        print(f"🎬 OpenSource Clipping v{version} - Story Clip Mode")
        print("=" * 70)
        print(f"   Recipe      : {cfg.story_recipe_path}")
        print(f"   Sources     : {cfg.sources_json_path}")
        print(f"   Ratio       : {cfg.pilihan_rasio}")
        print(f"   Output Dir  : {cfg.story_output_dir}")
        print(f"   Skip DL     : {'YES' if cfg.skip_download else 'NO'}")
        print("=" * 70)

        run_story_pipeline(cfg)

        print("\n✅ Done! All story clips have been rendered.")
        return

    # ── Normal Auto-Clip Mode ────────────────────────────────────────
    # Lazy import so --help works without heavy deps
    from clipping.runner import run_pipeline

    if not cfg.api_key_gemini:
        print("❌ ERROR: GOOGLE_API_KEY environment variable not found.")
        print("   Set via: export GOOGLE_API_KEY='your-key' or create a .env file")
        sys.exit(1)

    _PLATFORM_LABELS = {
        "youtube": "YouTube",
        "tiktok": "TikTok",
        "instagram": "Instagram",
        "gdrive": "Google Drive",
    }
    platform_key = getattr(cfg, "source_platform", "youtube")
    platform_label = _PLATFORM_LABELS.get(platform_key, platform_key)
    
    print("=" * 70)
    print(f"🎬 OpenSource Clipping v{version}")
    print("=" * 70)
    print(f"   Source      : {platform_label}")
    print(f"   URL         : {cfg.url_youtube}")
    print(f"   Clip Count  : {cfg.jumlah_clip}")
    print(f"   Ratio       : {cfg.pilihan_rasio}")
    print(f"   Font Style  : {cfg.gaya_font_aktif}")
    print(f"   Subtitles   : {'OFF' if cfg.no_subs else 'ON'}")
    print(f"   B-Roll      : {'ON' if cfg.use_broll else 'OFF'}")
    print(f"   Hook Glitch : {'ON' if cfg.use_hook_glitch else 'OFF'}")
    print(f"   BGM         : {'ON' if cfg.use_auto_bgm else 'OFF'}")
    print(f"   Karaoke     : {'ON' if cfg.use_karaoke_effect else 'OFF'}")
    print(f"   Split-Screen: {'ON' if cfg.use_split_screen else 'OFF'}")
    if cfg.use_split_screen:
        print(f"   Dynamic Split: {'ON' if cfg.use_dynamic_split else 'OFF'}")
        print(f"   Split Trigger: {cfg.split_trigger}")
    print(f"   Whisper     : {cfg.whisper_model} ({cfg.whisper_device})")
    print(f"   Gemini      : {cfg.gemini_model}")
    print("=" * 70)

    run_pipeline(cfg)

    print("\n✅ Done! All clips have been rendered.")


if __name__ == "__main__":
    main()
