"""
clipping.runner — Pipeline Orchestrator

Maps to Cell 4 (Execute) of the notebook.
Orchestrates the full clip generation pipeline.
"""

import json
import os

from . import diarization as diarization_mod
from . import engine, metadata, studio, hook_manager


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
    engine.download_video(
        cfg.url_youtube,
        cfg.file_video_asli,
        getattr(cfg, "use_dlp_subs", False),
        getattr(cfg, "download_source_height", "max"),
    )

    # Step 2 — Transcribe
    transkrip_lengkap = ""
    data_segmen = []

    import glob

    # Mencari file json3 apapun (karena bahasanya bisa .id.json3 atau .en.json3)
    json3_files = glob.glob(cfg.file_video_asli.replace(".mp4", ".*.json3"))
    file_json3 = json3_files[0] if json3_files else None

    if (
        getattr(cfg, "use_dlp_subs", False)
        and file_json3
        and os.path.exists(file_json3)
    ):
        transkrip_lengkap, data_segmen = engine.parse_youtube_json3_subs(
            file_json3, max_words_per_subtitle=cfg.max_kata_per_subtitle
        )
        if transkrip_lengkap and data_segmen:
            print(
                f"✅ Berhasil memparsing subtitle dari YouTube ({os.path.basename(file_json3)}), melewati proses Whisper."
            )

    if not transkrip_lengkap or not data_segmen:
        transkrip_lengkap, data_segmen = engine.transcribe_video(
            cfg.file_video_asli,
            max_words_per_subtitle=cfg.max_kata_per_subtitle,
            model_size=cfg.whisper_model,
            device=cfg.whisper_device,
            compute_type=cfg.whisper_compute_type,
        )

    # Step 3 — Gemini AI analysis
    gemini_output_path = os.path.join(cfg.outputs_dir, "gemini_response.json")
    
    if getattr(cfg, "load_gemini_json", False) and os.path.exists(gemini_output_path):
        print(f"\n🔄 [3/3] Memuat data Gemini (" + cfg.gemini_model + f") dari file lokal: {gemini_output_path}")
        with open(gemini_output_path, "r", encoding="utf-8") as f:
            hasil_json = json.load(f)
    else:
        hasil_json = engine.analyze_with_gemini(transkrip_lengkap, cfg)
        
        # Save raw gemini json for future loading/reproduction
        with open(gemini_output_path, "w", encoding="utf-8") as f:
            json.dump(hasil_json, f, indent=4, ensure_ascii=False)
        print(f"💾 Raw Gemini response tersimpan di: {gemini_output_path}")

    # Step 4 — Metadata normalisation
    hasil_json = metadata.normalize_and_validate(hasil_json)
    metadata.print_preview(hasil_json)

    metadata_path = os.path.join(cfg.outputs_dir, "metadata_preview.json")
    metadata.save_metadata_preview(hasil_json, path=metadata_path)

    # Step 5 — Diarization (split-screen / camera-switch)
    diarization_data = None
    if (
        (getattr(cfg, "use_split_screen", False) and cfg.split_trigger == "diarization")
        or getattr(cfg, "use_camera_switch", False)
    ) and cfg.pilihan_rasio == "9:16":
        try:
            mode_label = (
                "Split-Screen"
                if getattr(cfg, "use_split_screen", False)
                else "Camera-Switch"
            )
            print(f"\n🎙️ [{mode_label}] Menjalankan speaker diarization...")
            audio_path = cfg.file_video_asli.replace(".mp4", "_audio.wav")
            diarization_mod.extract_audio(cfg.file_video_asli, audio_path)
            num_speakers_arg = getattr(cfg, "diarization_num_speakers", 2)
            min_spk = None
            max_spk = None

            if str(num_speakers_arg).lower() == "auto":
                max_faces = studio.estimate_speaker_count_from_video(
                    cfg.file_video_asli, cfg
                )
                num_speakers_arg = "auto"
                min_spk = max(1, max_faces)
                max_spk = min_spk + 2
                print(f"   ℹ️ Instruksi Pyannote: {min_spk} hingga {max_spk} speaker.")

            diarization_data = diarization_mod.run_diarization(
                audio_path,
                hf_token=cfg.hf_token,
                num_speakers=num_speakers_arg,
                min_speakers=min_spk,
                max_speakers=max_spk,
            )
            # Clean up temp audio
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            print(f"⚠️ Diarization gagal: {e}")
            print("   Fallback ke mode render biasa (tanpa split-screen).")
            diarization_data = None

    # Step 6 — Video encoder & glitch
    os.environ["OSC_VIDEO_SCALE_ALGO"] = str(
        getattr(cfg, "video_scale_algo", "lanczos")
    )
    video_encoder = studio.detect_video_encoder(cfg)

    file_glitch_ts = None
    if cfg.use_hook_glitch:
        print("⚙️ Menyiapkan Video Glitch Transisi...")
        
        # Get source dimensions for proper glitch scaling
        import cv2
        cap_g = cv2.VideoCapture(cfg.file_video_asli)
        source_h_g = int(cap_g.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_g.release()

        file_glitch_ts = studio.siapkan_glitch_video(
            cfg.pilihan_rasio, cfg, video_encoder, source_h=source_h_g
        )

    # Step 6 — Render each clip
    render_manifest: list[dict] = []

    custom_hook_path = None
    if getattr(cfg, "hook_source", None):
        print("\n🎣 Mengunduh sumber klip Hook kustom...")
        custom_hook_path = hook_manager.download_custom_hook(cfg)

    for klip in sorted(hasil_json, key=lambda x: x["rank"]):
        
        if custom_hook_path:
            klip["custom_hook_info"] = {"file_path": custom_hook_path}

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

    print(
        f"\n💾 Render manifest disimpan ke {manifest_path} ({len(render_manifest)} item)"
    )
    return render_manifest
