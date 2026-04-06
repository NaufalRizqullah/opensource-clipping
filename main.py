#!/usr/bin/env python3
"""
OpenSource Clipping — AI Auto-Clipper & Teaser Generator

Usage:
    python main.py                          # run with all defaults
    python main.py --url "https://..." --clips 5 --ratio 16:9
    python main.py --help                   # show all available options
"""

import sys

from clipping.config import build_config


def main():
    cfg = build_config(sys.argv[1:])

    # Lazy import so --help works without heavy deps
    from clipping.runner import run_pipeline

    if not cfg.api_key_gemini:
        print("❌ ERROR: GOOGLE_API_KEY environment variable tidak ditemukan.")
        print("   Set via: export GOOGLE_API_KEY='your-key' atau buat file .env")
        sys.exit(1)

    print("=" * 70)
    print("🎬 OpenSource Clipping v0.6.3")
    print("=" * 70)
    print(f"   URL         : {cfg.url_youtube}")
    print(f"   Jumlah Clip : {cfg.jumlah_clip}")
    print(f"   Rasio       : {cfg.pilihan_rasio}")
    print(f"   Font Style  : {cfg.gaya_font_aktif}")
    print(f"   B-Roll      : {'ON' if cfg.use_broll else 'OFF'}")
    print(f"   Hook Glitch : {'ON' if cfg.use_hook_glitch else 'OFF'}")
    print(f"   BGM         : {'ON' if cfg.use_auto_bgm else 'OFF'}")
    print(f"   Karaoke     : {'ON' if cfg.use_karaoke_effect else 'OFF'}")
    print(f"   Split-Screen: {'ON' if cfg.use_split_screen else 'OFF'}")
    print(f"   Whisper     : {cfg.whisper_model} ({cfg.whisper_device})")
    print(f"   Gemini      : {cfg.gemini_model}")
    print("=" * 70)

    run_pipeline(cfg)

    print("\n✅ Selesai! Semua klip telah dirender.")


if __name__ == "__main__":
    main()
