"""
Core Studio pipeline implementation extracted from clipping/studio.py.

This module acts as an orchestrator, importing from modularized subcomponents
to maintain original behavior while allowing clipping/studio.py to remain
a thin orchestration and compatibility entry point.
"""

import html
import importlib.util
import json
import math
import os
import random
import re
import shutil
import string
import subprocess
import textwrap
import time
import urllib.parse
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import requests
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw, ImageFont
from yt_dlp import YoutubeDL

def _load_studio_internal_module(file_name: str, module_alias: str):
    module_path = os.path.join(os.path.dirname(__file__), file_name)
    spec = importlib.util.spec_from_file_location(module_alias, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import modules to re-export for studio.py
utils = _load_studio_internal_module("utils.py", "clipping_studio_utils")
_get_cv2_interpolation = utils._get_cv2_interpolation
_resize_frame = utils._resize_frame
_get_render_dims = utils._get_render_dims
face_detection = _load_studio_internal_module("face_detection.py", "clipping_studio_face_detection")
get_face_detector = face_detection.get_face_detector
estimate_speaker_count_from_video = face_detection.estimate_speaker_count_from_video
typography = _load_studio_internal_module("typography.py", "clipping_studio_typography")
download_google_font = typography.download_google_font
register_fonts_for_libass = typography.register_fonts_for_libass
siapkan_font_tipografi = typography.siapkan_font_tipografi
audio_bgm = _load_studio_internal_module("audio_bgm.py", "clipping_studio_audio_bgm")
resolve_pixabay_audio_url = audio_bgm.resolve_pixabay_audio_url
download_bgm_from_pixabay_page = audio_bgm.download_bgm_from_pixabay_page
broll = _load_studio_internal_module("broll.py", "clipping_studio_broll")
download_pexels_broll = broll.download_pexels_broll
crop_center_broll = broll.crop_center_broll
subtitles = _load_studio_internal_module("subtitles.py", "clipping_studio_subtitles")
buat_file_ass = subtitles.buat_file_ass
effects = _load_studio_internal_module("effects.py", "clipping_studio_effects")
siapkan_glitch_video = effects.siapkan_glitch_video
thumbnail = _load_studio_internal_module("thumbnail.py", "clipping_studio_thumbnail")
buat_thumbnail = thumbnail.buat_thumbnail
render_hybrid = _load_studio_internal_module("render_hybrid.py", "clipping_studio_render_hybrid")
buat_video_hybrid = render_hybrid.buat_video_hybrid
render_split_screen = _load_studio_internal_module("render_split_screen.py", "clipping_studio_render_split_screen")
buat_video_split_screen = render_split_screen.buat_video_split_screen
render_camera_switch = _load_studio_internal_module("render_camera_switch.py", "clipping_studio_render_camera_switch")
buat_video_camera_switch = render_camera_switch.buat_video_camera_switch

# Helpers and ffmpeg_utils
_helpers = _load_studio_internal_module("helpers.py", "clipping_studio_helpers")
_ffmpeg_utils = _load_studio_internal_module("ffmpeg_utils.py", "clipping_studio_ffmpeg_utils")

format_seconds = _helpers.format_seconds
escape_ffmpeg_filter_value = _helpers.escape_ffmpeg_filter_value
detect_video_encoder = _ffmpeg_utils.detect_video_encoder
get_ts_encode_args = _ffmpeg_utils.get_ts_encode_args
get_mp4_encode_args = _ffmpeg_utils.get_mp4_encode_args
open_ffmpeg_video_writer = _ffmpeg_utils.open_ffmpeg_video_writer
build_ffmpeg_progress_cmd = _ffmpeg_utils.build_ffmpeg_progress_cmd
run_ffmpeg_with_progress = _ffmpeg_utils.run_ffmpeg_with_progress

def proses_klip(
    rank, clip, rasio, glitch_ts, data_segmen, cfg, video_encoder, diarization_data=None
):
    """
    Run full clip processing pipeline from render to final output files.

    Args:
        rank: Clip rank/index.
        clip: Clip metadata object.
        rasio: Target output ratio.
        glitch_ts: Optional prepared glitch transition path.
        data_segmen: Transcript segments.
        cfg: Runtime config object.
        video_encoder: Encoder descriptor dict.
        diarization_data: Optional speaker diarization metadata.

    Returns:
        Manifest dictionary describing processing result and output paths.
    """
    get_x_h = None
    get_x_main = None
    h_start = float(clip.get("hook_start_time", clip["start_time"]))
    h_end = float(
        clip.get(
            "hook_end_time",
            clip.get("hook_start_time", clip["start_time"]) + cfg.durasi_hook,
        )
    )
    
    # Custom Hook Override
    file_hook_src = cfg.file_video_asli
    custom_hook = clip.get("custom_hook_info")
    if custom_hook:
        file_hook_src = custom_hook["file_path"]
        h_start = getattr(cfg, "hook_source_start", 0.0)
        
        try:
            cap_h = cv2.VideoCapture(file_hook_src)
            fps = cap_h.get(cv2.CAP_PROP_FPS)
            frames = cap_h.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0:
                vid_duration = frames / fps
            else:
                vid_duration = float('inf')
            cap_h.release()
        except:
            vid_duration = float('inf')

        h_end = h_start + cfg.durasi_hook
        if h_end > vid_duration:
            h_end = vid_duration
    m_start = float(clip["start_time"])
    m_end = float(clip["end_time"])
    judul = clip.get("title_indonesia")
    judul_en = clip.get("title_inggris")
    
    out_vid = os.path.join(cfg.outputs_dir, f"highlight_rank_{rank}_ready.mp4")
    if getattr(cfg, "dev_mode_with_output_merge", False):
        out_vid = os.path.join(cfg.outputs_dir, f"highlight_rank_{rank}_dev_mode_merge_ready.mp4")
        
    out_thm = os.path.join(cfg.outputs_dir, f"thumbnail_rank_{rank}.jpg")

    # Ambil resolusi video asli untuk perhitungan posisi subtitle di dev-mode
    cap_asli = cv2.VideoCapture(cfg.file_video_asli)
    sw = int(cap_asli.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap_asli.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_asli.release()
    source_dim = (sw, sh)

    manifest_item = {
        "rank": rank,
        "status": "pending",
        "ratio": rasio,
        "video_path": out_vid,
        "thumbnail_path": out_thm,
        "thumbnail_text": judul_en or judul or f"Highlight {rank}",
        "youtube_title_final": clip.get(
            "youtube_title_final", clip.get("title_inggris", "")
        ),
        "youtube_description_final": clip.get("youtube_description_final", ""),
        "youtube_tags_final": clip.get("youtube_tags_final", []),
        "tiktok_caption_final": clip.get(
            "tiktok_caption_final", clip.get("hastag", "")
        ),
        "title_indonesia": clip.get("title_indonesia", ""),
        "title_inggris": clip.get("title_inggris", ""),
        "hastag": clip.get("hastag", ""),
        "start_time": m_start,
        "end_time": m_end,
        "hook_start_time": h_start,
        "hook_end_time": h_end,
        "duration": round(m_end - m_start, 2),
        "alasan": clip.get("alasan", ""),
        "broll_list": clip.get("broll_list", []),
        "typography_plan": clip.get("typography_plan", []),
    }

    print(f"\n{'=' * 70}")
    print(f"🔥 [Rank {rank}] Memproses clip")
    print(f"📝 [Judul Indo]   : '{clip.get('title_indonesia', '-')}'")
    print(f"📝 [Judul Inggris]: '{clip.get('title_inggris', '-')}'")
    print(f"#️⃣ [Hastag]      : '{clip.get('hastag', '-')}'")
    print(f"🧠 Encoder aktif  : {video_encoder['name']}")
    print(f"{'=' * 70}")

    typography_plan = clip.get("typography_plan", [])
    siapkan_font_tipografi(cfg)

    h_ts, m_ts, a_hook, a_main = (
        f"h_{rank}.ts",
        f"m_{rank}.ts",
        f"ah_{rank}.ass",
        f"am_{rank}.ass",
    )
    h_silent, m_silent = f"h_silent_{rank}.mp4", f"m_silent_{rank}.mp4"
    
    dev_dual = getattr(cfg, "dev_mode_with_output", False)
    h_ts_dev = f"h_{rank}_dev.ts"
    m_ts_dev = f"m_{rank}_dev.ts"
    
    aktif_hook = cfg.use_hook_glitch


    # Determine if we should use split-screen mode
    if getattr(cfg, "use_split_screen", False) and rasio == "9:16":
        if cfg.split_trigger == "face":
            use_split = True
        else:
            use_split = (
                diarization_data is not None
                and len(set(s["speaker"] for s in diarization_data)) >= 2
            )
    else:
        use_split = False

    # Camera-switch mode (mutually exclusive: split-screen takes precedence)
    use_camera_switch = (
        not use_split
        and getattr(cfg, "use_camera_switch", False)
        and rasio == "9:16"
        and diarization_data
        and len(set(s["speaker"] for s in diarization_data)) >= 2
    )

    broll_list = clip.get("broll_list", [])
    broll_aktif = []
    if cfg.use_broll and broll_list:
        print(f"   🎥 Mendownload {len(broll_list)} video B-Roll dari Pexels...")
        for i, br in enumerate(broll_list):
            q = br.get("search_query", "nature")
            file_broll = f"temp_broll_{rank}_{i}.mp4"
            if download_pexels_broll(q, rasio, file_broll, cfg.pexels_api_key):
                br_copy = dict(br)
                br_copy["filepath"] = file_broll
                broll_aktif.append(br_copy)

    std_p = get_ts_encode_args(video_encoder, fps=30)

    try:
        # HOOK
        if aktif_hook:
            get_x_h = None
            if use_split:
                print("   📸 [Hook] Split-screen render (Custom Hook diabaikan untuk format ini saat ini atau digabung)...")
                get_x_h = buat_video_split_screen(
                    file_hook_src,
                    h_silent,
                    h_start,
                    h_end,
                    diarization_data if not custom_hook else None,
                    cfg,
                    label=f"Rank {rank} Hook SplitScreen",
                )
            elif use_camera_switch:
                print("   📸 [Hook] Camera switch render...")
                get_x_h = buat_video_camera_switch(
                    file_hook_src,
                    h_silent,
                    h_start,
                    h_end,
                    diarization_data if not custom_hook else None,
                    cfg,
                    label=f"Rank {rank} Hook CameraSwitch",
                )
            else:
                print("   📸 [Hook] Hybrid render...")
                get_x_h = buat_video_hybrid(
                    file_hook_src,
                    h_silent,
                    h_start,
                    h_end,
                    rasio,
                    cfg,
                    label=f"Rank {rank} Hook",
                )
            
            aktif_advanced_hook = cfg.use_advanced_text_on_hook
            if not cfg.no_subs and not custom_hook:
                buat_file_ass(
                    data_segmen,
                    h_start,
                    h_end,
                    a_hook,
                    rasio,
                    cfg,
                    typography_plan=typography_plan,
                    gunakan_advanced=aktif_advanced_hook,
                    get_x_func=get_x_h,
                    source_dim=source_dim,
                )

                print("   🎬 [Hook] FFmpeg burn subtitle + audio...")
                esc_ass_hook = escape_ffmpeg_filter_value(os.path.abspath(a_hook))
                esc_fontsdir = escape_ffmpeg_filter_value(os.path.abspath(cfg.font_dir))
                vf_hook_list = [f"subtitles={esc_ass_hook}:fontsdir={esc_fontsdir}"]
            else:
                print(f"   🎬 [Hook] Skip subtitle rendering {'(Custom Hook)' if custom_hook else ''}...")
                vf_hook_list = []
            
            if cfg.video_sharpen:
                vf_hook_list.append("unsharp=5:5:0.5:5:5:0.0")
            
            vf_hook = ",".join(vf_hook_list) if vf_hook_list else None

            cmd_h_base = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "verbose",
                "-y",
                "-i",
                h_silent,
                "-ss",
                str(h_start),
                "-to",
                str(h_end),
                "-i",
                file_hook_src,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
            ]
            if vf_hook:
                cmd_h_base += ["-vf", vf_hook]
            cmd_h_base += std_p

            cmd_h = build_ffmpeg_progress_cmd(cmd_h_base, h_ts)
            rc_h, err_h = run_ffmpeg_with_progress(
                cmd_h, h_end - h_start, label=f"Rank {rank} Hook FFmpeg"
            )
            if rc_h != 0:
                raise RuntimeError("FFmpeg hook gagal:\n" + "\n".join(err_h))

        # MAIN
        if use_split:
            print("   📸 [Main] Split-screen render (Visual)...")
            get_x_main = buat_video_split_screen(
                cfg.file_video_asli,
                m_silent,
                m_start,
                m_end,
                diarization_data,
                cfg,
                label=f"Rank {rank} Main SplitScreen",
            )
        elif use_camera_switch:
            # Note: Camera Switch doesn't currently support dev_mode frames but we pass it anyway
            print("   📸 [Main] Camera switch render (Visual)...")
            get_x_main = buat_video_camera_switch(
                cfg.file_video_asli,
                m_silent,
                m_start,
                m_end,
                diarization_data,
                cfg,
                label=f"Rank {rank} Main CameraSwitch",
            )
        else:
            print("   📸 [Main] Hybrid render (Visual)...")
            get_x_main = buat_video_hybrid(
                cfg.file_video_asli,
                m_silent,
                m_start,
                m_end,
                rasio,
                cfg,
                broll_aktif,
                label=f"Rank {rank} Main",
            )

        if not cfg.no_subs:
            buat_file_ass(
                data_segmen,
                m_start,
                m_end,
                a_main,
                rasio,
                cfg,
                typography_plan=typography_plan,
                gunakan_advanced=True,
                get_x_func=get_x_main,
                source_dim=source_dim,
            )

        print(f"   🎬 [Main] FFmpeg {'skip subtitle' if cfg.no_subs else 'burn subtitle'} + audio ducking...")
        esc_ass_main = escape_ffmpeg_filter_value(os.path.abspath(a_main)) if not cfg.no_subs else ""
        esc_fontsdir = escape_ffmpeg_filter_value(os.path.abspath(cfg.font_dir))

        # SMART BGM
        aktif_bgm = cfg.use_auto_bgm
        bgm_mood = clip.get("bgm_mood", "chill")
        if bgm_mood not in cfg.bgm_pool:
            bgm_mood = "chill"
        bgm_page = cfg.bgm_pool[bgm_mood]
        file_bgm = os.path.abspath(os.path.join(cfg.base_dir, f"bgm_{bgm_mood}.mp3"))

        if aktif_bgm and not os.path.exists(file_bgm):
            print(f"   🎵 Mendownload Background Music (Mood: {bgm_mood})...")
            ok_bgm = download_bgm_from_pixabay_page(bgm_page, file_bgm)
            if not ok_bgm and bgm_mood != "chill":
                print("   🔄 Fallback ke BGM chill...")
                chill_page = cfg.bgm_pool["chill"]
                file_bgm = os.path.abspath(os.path.join(cfg.base_dir, "bgm_chill.mp3"))
                ok_bgm = download_bgm_from_pixabay_page(chill_page, file_bgm)

            if ok_bgm:
                print(f"   ✅ BGM siap: {file_bgm}")
            else:
                print("   ⚠️ Semua fallback gagal. Render lanjut tanpa BGM.")

        # --- Subtitle & BGM Encoding Loop (Handles dual output files if needed) ---
        runs = [m_silent] if not dev_dual else [m_silent, m_silent.replace(".ts", "_dev.ts")]
        out_targets = [m_ts] if not dev_dual else [m_ts, m_ts_dev]
        
        for input_silent_ts, output_final_ts in zip(runs, out_targets):
            lbl_suffix = "" if input_silent_ts == m_silent else " (DEV)"
            
            if aktif_bgm and os.path.exists(file_bgm):
                v_filter_parts = [f"subtitles={esc_ass_main}:fontsdir={esc_fontsdir}"] if not cfg.no_subs else ["null"]
                if cfg.video_sharpen:
                    v_filter_parts.append("unsharp=5:5:0.5:5:5:0.0")
                v_filter = ",".join(v_filter_parts)
                
                filter_complex = (
                    f"[0:v]{v_filter}[v_out]; "
                    f"[1:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,volume=1.2[voc]; "
                    f"[2:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,volume={cfg.bgm_base_volume}[bgm]; "
                    f"[bgm][voc]sidechaincompress=threshold=0.08:ratio=5.0:attack=100:release=1000[bgm_ducked]; "
                    f"[voc][bgm_ducked]amix=inputs=2:duration=first:weights=1 1:dropout_transition=2[a_out]"
                )

                cmd_m_base = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "verbose",
                    "-y",
                    "-i",
                    input_silent_ts,
                    "-ss",
                    str(m_start),
                    "-to",
                    str(m_end),
                    "-i",
                    cfg.file_video_asli,
                    "-stream_loop",
                    "-1",
                    "-i",
                    file_bgm,
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[v_out]",
                    "-map",
                    "[a_out]",
                    "-shortest",
                ] + std_p
            else:
                cmd_m_base = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "verbose",
                    "-y",
                    "-i",
                    input_silent_ts,
                    "-ss",
                    str(m_start),
                    "-to",
                    str(m_end),
                    "-i",
                    cfg.file_video_asli,
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                ]
                if not cfg.no_subs:
                    vf_main_parts = [f"subtitles={esc_ass_main}:fontsdir={esc_fontsdir}"]
                else:
                    vf_main_parts = []
                
                if cfg.video_sharpen:
                    vf_main_parts.append("unsharp=5:5:0.5:5:5:0.0")
                
                if vf_main_parts:
                    cmd_m_base += ["-vf", ",".join(vf_main_parts)]
                cmd_m_base += std_p

            cmd_m = build_ffmpeg_progress_cmd(cmd_m_base, output_final_ts)
            rc_m, err_m = run_ffmpeg_with_progress(
                cmd_m, m_end - m_start, label=f"Rank {rank} Main FFmpeg{lbl_suffix}"
            )
            if rc_m != 0:
                raise RuntimeError(f"FFmpeg main{lbl_suffix} gagal:\n" + "\n".join(err_m))

        # FINAL CONCAT
        print("   🔗 [Final] Menyelesaikan clip akhir...")
        
        # Calculate target dimensions for each run
        out_w_std, out_h_std = _get_render_dims(cfg, rasio, source_h=sh)
        if getattr(cfg, "dev_mode_with_output_merge", False):
            out_w_std, out_h_std = 2648, 1220
        elif getattr(cfg, "dev_mode", False) and not dev_dual:
            # Single stream pure dev mode
            out_w_std, out_h_std = 1920, 1080
            
        concat_runs = [(out_vid, m_ts, h_ts, (out_w_std, out_h_std))]
        if dev_dual:
            out_vid_dev = os.path.join(cfg.outputs_dir, f"highlight_rank_{rank}_dev_mode_ready.mp4")
            # Usually hook doesn't generate dual, so we fallback to standard hook for dev if missing
            h_dev_target = h_ts_dev if os.path.exists(h_ts_dev) else h_ts 
            concat_runs.append((out_vid_dev, m_ts_dev, h_dev_target, (1920, 1080)))
            
        for final_path, main_vid_ts, hook_vid_ts, dims in concat_runs:
            # Dynamically prepare glitch for THIS run's dimensions
            cur_glitch = None
            if aktif_hook:
                cur_glitch = siapkan_glitch_video(rasio, cfg, video_encoder, source_h=sh, custom_dims=dims)
            
            if aktif_hook and cur_glitch and os.path.exists(cur_glitch) and os.path.exists(hook_vid_ts):
                concat_str = f"concat:{hook_vid_ts}|{cur_glitch}|{main_vid_ts}"
            else:
                concat_str = f"concat:{main_vid_ts}"

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    concat_str,
                    "-c",
                    "copy",
                    "-bsf:a",
                    "aac_adtstoasc",
                    final_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        judul_thumbnail = judul_en or judul or f"Highlight {rank}"
        buat_thumbnail(out_vid, out_thm, judul_thumbnail, cfg)

        manifest_item["status"] = "success"
        manifest_item["video_exists"] = os.path.exists(out_vid)
        manifest_item["thumbnail_exists"] = os.path.exists(out_thm)

        print(f"✅ [Rank {rank}] Selesai.")
        return manifest_item

    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: FFmpeg gagal. Error: {e}")
        manifest_item["status"] = "failed"
        manifest_item["error"] = str(e)
        manifest_item["video_exists"] = os.path.exists(out_vid)
        manifest_item["thumbnail_exists"] = os.path.exists(out_thm)
        return manifest_item

    except Exception as e:
        print(f"\n❌ ERROR: Kegagalan tak terduga. Error: {e}")
        manifest_item["status"] = "failed"
        manifest_item["error"] = str(e)
        manifest_item["video_exists"] = os.path.exists(out_vid)
        manifest_item["thumbnail_exists"] = os.path.exists(out_thm)
        return manifest_item

    finally:
        files_to_remove = [h_ts, m_ts, a_hook, a_main, h_silent, m_silent]
        if dev_dual:
            files_to_remove.extend([h_ts_dev, m_ts_dev, m_silent.replace(".ts", "_dev.ts")])
            
        for br in broll_aktif:
            files_to_remove.append(br["filepath"])

        for f_path in files_to_remove:
            if os.path.exists(f_path):
                os.remove(f_path)
