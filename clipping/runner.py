"""
clipping.runner — Pipeline Orchestrator

Maps to Cell 4 (Execute) of the notebook.
Orchestrates the full clip generation pipeline.
"""

import json

from . import engine, metadata, studio


def run_pipeline(cfg) -> list[dict]:
    """
    Run the full clipping pipeline:
      1. Download YouTube video
      2. Transcribe with Whisper
      3. Analyse with Gemini AI
      4. Normalize metadata
      5. Prepare glitch transition
      6. Render each clip
      7. Save render_manifest.json

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object from ``config.build_config()``.

    Returns
    -------
    list[dict]
        Render manifest (one dict per clip).
    """

    # Step 1 — Download
    engine.download_video(cfg.url_youtube, cfg.file_video_asli)

    # Step 2 — Transcribe
    transkrip_lengkap, data_segmen = engine.transcribe_video(
        cfg.file_video_asli,
        max_words_per_subtitle=cfg.max_kata_per_subtitle,
        model_size=cfg.whisper_model,
        device=cfg.whisper_device,
        compute_type=cfg.whisper_compute_type,
    )

    # Step 3 — Gemini AI analysis
    hasil_json = engine.analyze_with_gemini(transkrip_lengkap, cfg)

    # Step 4 — Metadata normalisation
    hasil_json = metadata.normalize_and_validate(hasil_json)
    metadata.print_preview(hasil_json)
    metadata.save_metadata_preview(hasil_json)

    # Step 5 — Video encoder & glitch
    video_encoder = studio.detect_video_encoder()

    file_glitch_ts = None
    if cfg.use_hook_glitch:
        print("⚙️ Menyiapkan Video Glitch Transisi...")
        file_glitch_ts = studio.siapkan_glitch_video(cfg.pilihan_rasio, cfg, video_encoder)

    # Step 6 — Render each clip
    render_manifest: list[dict] = []

    for klip in sorted(hasil_json, key=lambda x: x["rank"]):
        hasil_render = studio.proses_klip(
            klip["rank"],
            klip,
            cfg.pilihan_rasio,
            file_glitch_ts,
            data_segmen,
            cfg,
            video_encoder,
        )
        if hasil_render:
            render_manifest.append(hasil_render)

    # Step 7 — Save manifest
    with open("render_manifest.json", "w", encoding="utf-8") as f:
        json.dump(render_manifest, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Render manifest disimpan ke render_manifest.json ({len(render_manifest)} item)")
    return render_manifest
