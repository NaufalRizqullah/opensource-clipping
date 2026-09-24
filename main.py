#!/usr/bin/env python3
"""
OpenSource Clipping — AI Auto-Clipper & Teaser Generator

Usage:
    python main.py --url "https://..."                    # single URL
    python main.py --url "https://..." "https://..." ...  # batch multi-URL
    python main.py --url "https://..." --clips 5 --ratio 16:9
    python main.py --help                                 # show all available options
"""

import copy
import json
import os
import sys
import time

from clipping.config import build_config, _make_url_slug


def _print_banner(cfg, version, url, idx=None, total=None):
    """Print configuration banner for a single URL run."""
    _PLATFORM_LABELS = {
        "youtube": "YouTube",
        "tiktok": "TikTok",
        "instagram": "Instagram",
        "gdrive": "Google Drive",
    }
    platform_key = getattr(cfg, "source_platform", "youtube")
    platform_label = _PLATFORM_LABELS.get(platform_key, platform_key)

    print("=" * 70)
    if idx is not None and total is not None and total > 1:
        print(f"🎬 OpenSource Clipping v{version} — [{idx}/{total}]")
    else:
        print(f"🎬 OpenSource Clipping v{version}")
    print("=" * 70)
    print(f"   Source      : {platform_label}")
    print(f"   URL         : {url}")
    print(f"   Jumlah Clip : {cfg.jumlah_clip}")
    print(f"   Rasio       : {cfg.pilihan_rasio}")
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
    if getattr(cfg, "watermark_enabled", False):
        wm_type = "Text" if cfg.watermark_text else "Image"
        wm_content = cfg.watermark_text or cfg.watermark_image or "-"
        print(f"   Watermark   : ON ({wm_type}: {wm_content})")
        print(f"   WM Opacity  : {cfg.watermark_opacity}%")
        print(f"   WM Position : {cfg.watermark_position}")
        print(f"   WM Padding  : {cfg.watermark_padding}px")
        if cfg.watermark_image:
            print(f"   WM Scale    : {getattr(cfg, 'watermark_scale', 15)}% of frame height")
    if getattr(cfg, "cleanup_source", False):
        print(f"   Cleanup Src : ON (auto-delete source video after render)")
    print("=" * 70)


def _cleanup_source_video(cfg):
    """Remove downloaded source video and associated subtitle files to free disk."""
    video_path = cfg.file_video_asli
    if os.path.exists(video_path):
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        os.remove(video_path)
        print(f"   🗑️ Deleted source video: {os.path.basename(video_path)} ({size_mb:.1f} MB freed)")

    # Also clean up subtitle files (*.json3)
    import glob
    json3_pattern = video_path.replace(".mp4", ".*.json3")
    for f in glob.glob(json3_pattern):
        os.remove(f)
        print(f"   🗑️ Deleted subtitle: {os.path.basename(f)}")

    # Clean up temp audio if exists
    audio_path = video_path.replace(".mp4", "_audio.wav")
    if os.path.exists(audio_path):
        os.remove(audio_path)


def _run_single_url(cfg, url, version, idx=None, total=None):
    """
    Run the full pipeline for a single URL.

    Returns the render manifest list on success, or None on failure.
    """
    from clipping.runner import run_pipeline

    _print_banner(cfg, version, url, idx, total)

    manifest = run_pipeline(cfg)

    # Cleanup source video if requested
    if getattr(cfg, "cleanup_source", False):
        print("\n🧹 Cleaning up source video...")
        _cleanup_source_video(cfg)

    return manifest


def main():
    cfg = build_config(sys.argv[1:])

    version = "1.12.0"

    # ── Story Clip Mode ──────────────────────────────────────────────
    if getattr(cfg, "story_mode", False):
        from clipping.story_runner import run_story_pipeline

        print("=" * 70)
        print(f"🎬 OpenSource Clipping v{version} — Story Clip Mode")
        print("=" * 70)
        print(f"   Recipe      : {cfg.story_recipe_path}")
        print(f"   Sources     : {cfg.sources_json_path}")
        print(f"   Rasio       : {cfg.pilihan_rasio}")
        print(f"   Output Dir  : {cfg.story_output_dir}")
        print(f"   Skip DL     : {'YES' if cfg.skip_download else 'NO'}")
        print("=" * 70)

        run_story_pipeline(cfg)

        print("\n✅ Selesai! Semua story clips telah dirender.")
        return

    # ── Normal Auto-Clip Mode ────────────────────────────────────────
    if not cfg.api_key_gemini:
        print("❌ ERROR: GOOGLE_API_KEY environment variable tidak ditemukan.")
        print("   Set via: export GOOGLE_API_KEY='your-key' atau buat file .env")
        sys.exit(1)

    url_list = getattr(cfg, "url_list", [cfg.url_youtube] if isinstance(cfg.url_youtube, str) else cfg.url_youtube)
    is_batch = len(url_list) > 1

    # ── Single URL Mode (backward-compatible) ────────────────────────
    if not is_batch:
        _run_single_url(cfg, url_list[0], version)
        print("\n✅ Selesai! Semua klip telah dirender.")
        return

    # ── Batch Multi-URL Mode ─────────────────────────────────────────
    base_outputs_dir = cfg.outputs_dir
    total = len(url_list)

    print("=" * 70)
    print(f"🎬 OpenSource Clipping v{version} — Batch Mode")
    print("=" * 70)
    print(f"   Total URLs  : {total}")
    for i, url in enumerate(url_list):
        print(f"   [{i + 1}] {url}")
    if getattr(cfg, "cleanup_source", False):
        print(f"   Cleanup Src : ON (auto-delete after each render)")
    print("=" * 70)

    all_manifests = []
    errors = []
    t_batch_start = time.time()
    
    # Pre-load Whisper model for batch mode to avoid reloading per URL
    global_whisper_model = None
    try:
        from faster_whisper import WhisperModel
        print(f"⏳ Pre-loading Whisper model '{cfg.whisper_model}' for batch processing...")
        global_whisper_model = WhisperModel(
            cfg.whisper_model, 
            device=cfg.whisper_device, 
            compute_type=cfg.whisper_compute_type
        )
    except Exception as e:
        print(f"⚠️ Failed to pre-load Whisper model: {e}")

    for idx, url in enumerate(url_list):
        t_start = time.time()

        # Clone config for this URL
        url_cfg = copy.copy(cfg)
        url_cfg.url_youtube = url
        url_cfg.url_list = [url]  # single-item for runner compatibility

        # Isolate output directory per URL
        slug = _make_url_slug(url, idx)
        url_cfg.outputs_dir = os.path.join(base_outputs_dir, slug)
        os.makedirs(url_cfg.outputs_dir, exist_ok=True)

        # Isolate downloaded video file per URL
        url_cfg.file_video_asli = os.path.join(url_cfg.outputs_dir, "video_asli.mp4")

        print(f"\n{'━' * 70}")
        print(f"📹 [{idx + 1}/{total}] Processing: {url}")
        print(f"   Output Dir  : {url_cfg.outputs_dir}")
        print(f"{'━' * 70}")

        try:
            from clipping.runner import run_pipeline
            _print_banner(url_cfg, version, url, idx + 1, total)
            
            manifest = run_pipeline(url_cfg, whisper_model_instance=global_whisper_model)
            
            # Cleanup source video if requested
            if getattr(url_cfg, "cleanup_source", False):
                print("\n🧹 Cleaning up source video...")
                _cleanup_source_video(url_cfg)

            elapsed = time.time() - t_start
            clip_count = len(manifest) if manifest else 0
            all_manifests.append({
                "index": idx + 1,
                "url": url,
                "slug": slug,
                "output_dir": url_cfg.outputs_dir,
                "status": "success",
                "clips_rendered": clip_count,
                "duration_seconds": round(elapsed, 1),
                "manifest": manifest,
            })
            print(f"\n✅ [{idx + 1}/{total}] Selesai — {clip_count} clip(s) rendered in {elapsed:.0f}s")

        except Exception as e:
            elapsed = time.time() - t_start
            error_msg = str(e)
            errors.append({"index": idx + 1, "url": url, "error": error_msg})
            all_manifests.append({
                "index": idx + 1,
                "url": url,
                "slug": slug,
                "output_dir": url_cfg.outputs_dir,
                "status": "failed",
                "error": error_msg,
                "duration_seconds": round(elapsed, 1),
                "manifest": None,
            })
            print(f"\n❌ [{idx + 1}/{total}] GAGAL — {error_msg}")
            print(f"   ⏭️ Melanjutkan ke URL berikutnya...")

    # ── Batch Summary ────────────────────────────────────────────────
    total_elapsed = time.time() - t_batch_start
    success_count = sum(1 for m in all_manifests if m["status"] == "success")
    fail_count = sum(1 for m in all_manifests if m["status"] == "failed")

    # Save combined batch manifest
    batch_manifest_path = os.path.join(base_outputs_dir, "batch_manifest.json")
    batch_summary = {
        "batch_mode": True,
        "total_urls": total,
        "success": success_count,
        "failed": fail_count,
        "total_duration_seconds": round(total_elapsed, 1),
        "results": all_manifests,
    }
    with open(batch_manifest_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f"📦 Batch Processing Complete")
    print(f"{'=' * 70}")
    print(f"   Total       : {total} video(s)")
    print(f"   ✅ Success   : {success_count}")
    if fail_count:
        print(f"   ❌ Failed    : {fail_count}")
        for err in errors:
            print(f"      [{err['index']}] {err['url']}")
            print(f"          Error: {err['error']}")
    print(f"   ⏱️ Total Time: {total_elapsed:.0f}s")
    print(f"   📄 Manifest  : {batch_manifest_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

