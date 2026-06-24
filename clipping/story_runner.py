"""
clipping.story_runner - Story Clip Pipeline Orchestrator

Orchestrates the full Story Clip pipeline:
  1. Load & validate sources.json
  2. Download & cache all source videos
  3. Transcribe each source with Whisper
  4. Load & validate story_recipe.json
  5. Assemble each clip (hook + highlight) - clean, no subs
  6. Save story_manifest.json
"""

import json
import os

from .story import loader, source_manager, assembler


# ==============================================================================
# WHISPER TRANSCRIPTION FOR STORY SOURCES
# ==============================================================================

def _transcribe_sources(
    cached_paths: dict[str, str],
    cache_dir: str,
    cfg,
) -> dict[str, dict]:
    """
    Transcribe all cached source videos using Faster-Whisper.

    Parameters
    ----------
    cached_paths : dict[str, str]
        Mapping of source_id → cached video file path.
    cache_dir : str
        Directory where transcript JSON files will be saved.
    cfg : SimpleNamespace
        Config with Whisper settings (model, device, compute_type).

    Returns
    -------
    dict[str, dict]
        Mapping of source_id → {"transkrip": str, "segmen": list, "path": str}.
    """
    # Lazy import to avoid pulling in heavy deps
    try:
        from . import engine
    except ImportError as e:
        print(f"   ⚠️ Whisper not available ({e}). Skipping transcription.")
        print(f"   💡 Install faster-whisper to enable transcription.")
        return {}

    whisper_model = getattr(cfg, "whisper_model", "large-v3")
    whisper_device = getattr(cfg, "whisper_device", "cuda")
    whisper_compute = getattr(cfg, "whisper_compute_type", "float16")
    max_words = getattr(cfg, "max_kata_per_subtitle", 5)

    transcripts: dict[str, dict] = {}
    total = len(cached_paths)

    for idx, (sid, video_path) in enumerate(cached_paths.items(), 1):
        transcript_path = os.path.join(cache_dir, f"{sid}_transcript.json")

        # Skip if already transcribed
        if os.path.exists(transcript_path):
            print(f"   ⏩ [{idx}/{total}] '{sid}' already has a transcript, skip.")
            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    transcripts[sid] = json.load(f)
                continue
            except Exception:
                pass  # Re-transcribe if JSON is corrupted

        if not os.path.exists(video_path):
            print(f"   ⚠️ [{idx}/{total}] '{sid}' file not found, skip transcript.")
            continue

        print(f"   🎤 [{idx}/{total}] Transcribing '{sid}'...")
        try:
            transkrip, segmen = engine.transcribe_video(
                video_path,
                max_words_per_subtitle=max_words,
                model_size=whisper_model,
                device=whisper_device,
                compute_type=whisper_compute,
            )

            result = {
                "source_id": sid,
                "transkrip": transkrip,
                "segmen": segmen,
                "path": transcript_path,
            }

            # Save transcript JSON
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            transcripts[sid] = result
            print(f"   ✅ '{sid}' successfully transcribed ({len(segmen)} segment(s)).")

        except Exception as e:
            print(f"   ⚠️ '{sid}' failed to transcribe: {e}")

    return transcripts


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_story_pipeline(cfg) -> list[dict]:
    """
    Run the full Story Clip pipeline.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object from ``config.build_config()``.
        Must include ``story_recipe_path``, ``sources_json_path``,
        and standard config fields.

    Returns
    -------
    list[dict]
        Story manifest (one dict per clip with output paths).
    """

    print("=" * 70)
    print("🎬 Story Clip - Multi-Source Narrative Assembly")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1 - Load sources.json
    # ------------------------------------------------------------------
    sources_path = getattr(cfg, "sources_json_path", "sources.json")
    print(f"\n[1/6] Loading sources: {sources_path}")
    source_registry = loader.load_sources(sources_path)

    # ------------------------------------------------------------------
    # Step 2 - Download & cache all sources
    # ------------------------------------------------------------------
    skip_download = getattr(cfg, "skip_download", False)
    cache_dir = source_manager.get_cache_dir(cfg.outputs_dir)

    if skip_download:
        print("\n[2/6] ⏩ Skip download (--skip-download active)")
        # Build paths from existing cache
        cached_paths = {}
        for sid, src in source_registry.items():
            if src["platform"] == "local":
                cached_paths[sid] = src["local_path"]
            else:
                cached = os.path.join(cache_dir, f"{sid}.mp4")
                if os.path.exists(cached):
                    cached_paths[sid] = cached
                else:
                    print(f"   ⚠️ Cache not found for '{sid}': {cached}")
    else:
        print(f"\n[2/6] Downloading sources → {cache_dir}")
        download_height = getattr(cfg, "download_source_height", "max")
        cached_paths = source_manager.download_all_sources(
            source_registry, cache_dir, download_height
        )

    # Save download status
    source_manager.save_sources_status(source_registry, cached_paths, cfg.outputs_dir)

    # ------------------------------------------------------------------
    # Step 3 - Transcribe each source with Whisper
    # ------------------------------------------------------------------
    print(f"\n[3/6] Transcribing sources with Whisper...")
    transcripts = _transcribe_sources(cached_paths, cache_dir, cfg)
    print(f"   📝 {len(transcripts)}/{len(cached_paths)} source(s) successfully transcribed.")

    # ------------------------------------------------------------------
    # Step 4 - Load & validate recipe
    # ------------------------------------------------------------------
    recipe_path = getattr(cfg, "story_recipe_path", "story_recipe.json")
    print(f"\n[4/6] Loading recipe: {recipe_path}")
    recipe = loader.load_recipe(recipe_path, source_registry)

    # ------------------------------------------------------------------
    # Step 5 - Assemble each clip (clean, no subs, no text overlay)
    # ------------------------------------------------------------------
    clips = recipe.get("clips", [])
    defaults = recipe.get("_defaults", None)
    ratio = getattr(cfg, "pilihan_rasio", None) or (defaults.ratio if defaults else "9:16")

    story_output_dir = getattr(
        cfg, "story_output_dir",
        os.path.join(cfg.outputs_dir, "story_clips")
    )
    os.makedirs(story_output_dir, exist_ok=True)

    print(f"\n[5/6] Assembling {len(clips)} clip(s)...")
    print(f"   Output dir: {story_output_dir}")
    print(f"   Ratio: {ratio}")
    print(f"   Mode: clean (no subs, no text overlay)")

    manifest: list[dict] = []

    for clip_config in sorted(clips, key=lambda c: c["clip_id"]):
        cid = clip_config["clip_id"]
        title = clip_config.get("title", f"Clip {cid}")

        print(f"\n{'─'*50}")
        print(f"📎 Clip #{cid}: {title}")
        print(f"{'─'*50}")

        clip_dir = os.path.join(story_output_dir, f"clip_{cid}")
        os.makedirs(clip_dir, exist_ok=True)

        # --- Assemble Hook (clean) ---
        hook_path = assembler.assemble_hook(
            clip_config=clip_config,
            source_registry=source_registry,
            cache_dir=cache_dir,
            output_dir=clip_dir,
            ratio=ratio,
        )

        # --- Assemble Highlight (clean) ---
        highlight_path = assembler.assemble_highlight(
            clip_config=clip_config,
            source_registry=source_registry,
            cache_dir=cache_dir,
            output_dir=clip_dir,
            ratio=ratio,
        )

        entry = {
            "clip_id": cid,
            "title": title,
            "hook_path": hook_path,
            "highlight_path": highlight_path,
            "status": "ok" if (hook_path and highlight_path) else "partial",
            "metadata": clip_config.get("metadata", {}),
        }
        manifest.append(entry)

    # ------------------------------------------------------------------
    # Step 6 - Save manifest
    # ------------------------------------------------------------------
    manifest_path = os.path.join(cfg.outputs_dir, "story_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Save transcripts index
    transcripts_index_path = os.path.join(cfg.outputs_dir, "story_transcripts.json")
    transcripts_summary = {}
    for sid, t in transcripts.items():
        transcripts_summary[sid] = {
            "path": t.get("path", ""),
            "segmen_count": len(t.get("segmen", [])),
        }
    with open(transcripts_index_path, "w", encoding="utf-8") as f:
        json.dump(transcripts_summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ Story Clip done! {len(manifest)} clip(s) rendered.")
    print(f"💾 Manifest: {manifest_path}")
    print(f"📝 Transcripts: {transcripts_index_path}")
    print(f"📁 Output: {story_output_dir}")
    print(f"{'='*70}")

    # Print summary table
    print(f"\n{'Clip':>6} | {'Title':<35} | {'Hook':>6} | {'Highlight':>10} | Status")
    print(f"{'─'*6} | {'─'*35} | {'─'*6} | {'─'*10} | {'─'*8}")
    for entry in manifest:
        hook_ok = "✅" if entry["hook_path"] else "❌"
        hl_ok = "✅" if entry["highlight_path"] else "❌"
        title_short = entry["title"][:35]
        print(f"  {entry['clip_id']:>4} | {title_short:<35} | {hook_ok:>6} | {hl_ok:>10} | {entry['status']}")

    return manifest

