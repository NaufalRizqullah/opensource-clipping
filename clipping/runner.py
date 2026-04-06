"""
clipping.runner — Pipeline Orchestrator

Maps to Cell 4 (Execute) of the notebook.
Orchestrates the full clip generation pipeline.
"""

import os
import json

from . import engine, metadata, studio
from . import diarization as diarization_mod


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
    engine.download_video(cfg.url_youtube, cfg.file_video_asli, getattr(cfg, "use_dlp_subs", False))

    # Step 2 — Transcribe
    transkrip_lengkap = ""
    data_segmen = []
    
    import glob
    # Mencari file json3 apapun (karena bahasanya bisa .id.json3 atau .en.json3)
    json3_files = glob.glob(cfg.file_video_asli.replace(".mp4", ".*.json3"))
    file_json3 = json3_files[0] if json3_files else None

    if getattr(cfg, "use_dlp_subs", False) and file_json3 and os.path.exists(file_json3):
        transkrip_lengkap, data_segmen = engine.parse_youtube_json3_subs(
            file_json3,
            max_words_per_subtitle=cfg.max_kata_per_subtitle
        )
        if transkrip_lengkap and data_segmen:
            print(f"✅ Berhasil memparsing subtitle dari YouTube ({os.path.basename(file_json3)}), melewati proses Whisper.")

    if not transkrip_lengkap or not data_segmen:
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
    
    metadata_path = os.path.join(cfg.outputs_dir, "metadata_preview.json")
    metadata.save_metadata_preview(hasil_json, path=metadata_path)

    # Step 5 — Diarization (split-screen)
    diarization_data = None
    if getattr(cfg, "use_split_screen", False) and cfg.pilihan_rasio == "9:16":
        try:
            print("\n🎙️ [Split-Screen] Menjalankan speaker diarization...")
            audio_path = cfg.file_video_asli.replace(".mp4", "_audio.wav")
            diarization_mod.extract_audio(cfg.file_video_asli, audio_path)
            diarization_data = diarization_mod.run_diarization(
                audio_path,
                hf_token=cfg.hf_token,
                num_speakers=getattr(cfg, "diarization_num_speakers", 2),
            )
            # Clean up temp audio
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            print(f"⚠️ Diarization gagal: {e}")
            print("   Fallback ke mode render biasa (tanpa split-screen).")
            diarization_data = None

    # Step 6 — Video encoder & glitch
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
            diarization_data=diarization_data,
        )
        if hasil_render:
            render_manifest.append(hasil_render)

    # Step 7 — Save manifest
    manifest_path = os.path.join(cfg.outputs_dir, "render_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(render_manifest, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Render manifest disimpan ke {manifest_path} ({len(render_manifest)} item)")
    return render_manifest
