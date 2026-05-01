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

FIREFOX_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0"

def _load_studio_internal_module(file_name: str, module_alias: str):
    module_path = os.path.join(os.path.dirname(__file__), file_name)
    spec = importlib.util.spec_from_file_location(module_alias, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

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
utils = _load_studio_internal_module("utils.py", "clipping_studio_utils")
_resize_frame = utils._resize_frame
_get_render_dims = utils._get_render_dims
face_detection = _load_studio_internal_module("face_detection.py", "clipping_studio_face_detection")
get_face_detector = face_detection.get_face_detector

def buat_video_hybrid(
    input_video,
    output_video,
    start_clip,
    end_clip,
    rasio,
    cfg,
    broll_data=None,
    label="Hybrid",
):
    """
    Render a hybrid video combining main footage and b-roll with dynamic panning based on face tracking.

    Args:
        input_video (str): Source video file path.
        output_video (str): Output video file path.
        start_clip (float): Start timestamp in seconds.
        end_clip (float): End timestamp in seconds.
        rasio (str): Output ratio string ('9:16' or '16:9').
        cfg: Configuration object for parameters like deadzones and smoothing factors.
        broll_data (list, optional): Metadata dicts of B-roll timing to overlay.
        label (str, optional): The UI label used for rendering progress output.

    Returns:
        callable: A lambda `get_x_final(t)` which returns the dynamic X crop position given a timestamp `t`.

    Side Effects:
        Reads frames, runs face detection (YOLO or Mediapipe), and writes processed frames via FFMPEG.
        Loads B-Roll clips and composites them automatically.

    Raises:
        Exceptions are typically handled and may cause a hard exit if critical video rendering fails.
    """
    if broll_data is None:
        broll_data = []

    # =======================================================
    # 🎛️ PARAMETER TUNING KAMERA
    # =======================================================
    STEP_DETEKSI     = cfg.track_step if cfg.track_step is not None else 0.25   # AI mengecek wajah tiap 0.25 detik
    # STEP_DETEKSI     = 0.5   # AI mengecek wajah tiap 0.5 detik
    # STEP_DETEKSI     = max(0.5, (end_clip - start_clip) / 60.0)   # [OLD] AI mengecek wajah tiap max 0.5 atau sepanjang durasi (end_clip - start_clip) detik per menit

    DEADZONE_RATIO   = cfg.track_deadzone if cfg.track_deadzone is not None else 0.15  # 15% area tengah adalah zona aman (kamera tidak ikut gerak)
    # DEADZONE_RATIO   = 0.25  # 25% area tengah adalah zona aman (kamera tidak ikut gerak)
    # DEADZONE_RATIO   = 0.20  # [OLD] 20% area tengah adalah zona aman (kamera tidak ikut gerak)

    SMOOTH_FACTOR    = cfg.track_smooth if cfg.track_smooth is not None else 0.30  # Kecepatan kamera menyusul (30% jarak). Bikin pergerakan sangat mulus.
    # SMOOTH_FACTOR    = 0.15  # Kecepatan kamera menyusul (15% jarak). Bikin pergerakan sangat mulus.
    # SMOOTH_FACTOR    = 0.10  # [NEW; NOT USED]Kecepatan kamera menyusul (10% jarak). Bikin pergerakan sangat mulus.

    JITTER_THRESHOLD = cfg.track_jitter if cfg.track_jitter is not None else 5     # Abaikan pergeseran di bawah 5 pixel (Anti-getar/Micro-jitter)
    # JITTER_THRESHOLD = 4     # [OLD] Abaikan pergeseran di bawah 4 pixel (Anti-getar/Micro-jitter)

    SNAP_THRESHOLD   = cfg.track_snap if cfg.track_snap is not None else 0.25  # Jika wajah lompat > 25% lebar layar, anggap ganti orang (Hard Cut)
    # SNAP_THRESHOLD   = 0.30  # [NEW; NOT USED] Jika wajah lompat > 30% lebar layar, anggap ganti orang (Hard Cut)
    # =======================================================

    video_encoder = detect_video_encoder(cfg)

    yolo_model = None
    detector = None
    if cfg.face_detector == "yolo":
        if not os.path.exists(cfg.file_yolo_model):
            print(f"   📥 Mendownload YOLOv8 Face Model ({cfg.yolo_size})...")
            import urllib.request

            urllib.request.urlretrieve(cfg.url_yolo_model, cfg.file_yolo_model)
        from ultralytics import YOLO

        yolo_model = YOLO(cfg.file_yolo_model)
    else:
        detector = get_face_detector(cfg)

    cap = cv2.VideoCapture(input_video)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if math.isnan(orig_fps) or orig_fps == 0:
        orig_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop_w = int(height * 9 / 16)
    crop_h = height
    default_cx = width // 2
    default_cy = height // 2
    duration = end_clip - start_clip

    broll_caps = []
    for br in broll_data:
        if "filepath" in br and os.path.exists(br["filepath"]):
            broll_caps.append(
                {
                    "start": br["start_time"],
                    "end": br["end_time"],
                    "cap": cv2.VideoCapture(br["filepath"]),
                }
            )

    # FASE 1: DETEKSI WAJAH
    raw_data = []
    current_time = 0.0
    last_detect_percent = -1
    print(f"🧠 {label} - Analisa wajah dimulai...", flush=True)

    while current_time <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, (start_clip + current_time) * 1000)
        ret, frame = cap.read()
        if not ret:
            break


        face_box = None

        if cfg.face_detector == "yolo":
            yolo_results = yolo_model(frame, verbose=False)
            if yolo_results and len(yolo_results[0].boxes) > 0:
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                largest_idx = areas.argmax()
                x1, y1, x2, y2 = boxes[largest_idx]
                center_x = x1 + (x2 - x1) / 2
                center_y = y1 + (y2 - y1) / 2
                face_box = (x1, y1, x2, y2)
        else:
            results = detector.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
            )

            if results.detections:
                largest_face = max(
                    results.detections,
                    key=lambda d: d.bounding_box.width * d.bounding_box.height,
                ).bounding_box
                center_x = largest_face.origin_x + (largest_face.width / 2)
                center_y = largest_face.origin_y + (largest_face.height / 2)
                face_box = (
                    largest_face.origin_x,
                    largest_face.origin_y,
                    largest_face.origin_x + largest_face.width,
                    largest_face.origin_y + largest_face.height,
                )

        raw_data.append(
            {
                "time": current_time,
                "cx": center_x if face_box else default_cx,
                "cy": center_y if face_box else default_cy,
                "box": face_box,
            }
        )

        detect_percent = (
            min(100, int((current_time / duration) * 100)) if duration > 0 else 100
        )
        if detect_percent != last_detect_percent:
            print(f"⏳ {label} - Analisa wajah: {detect_percent:3d}%", flush=True)
            last_detect_percent = detect_percent

        current_time += STEP_DETEKSI

    # FASE 2: SMOOTH CAMERA
    smooth_data = []
    if raw_data:
        import statistics as _st
        initial_cxs = [d["cx"] for d in raw_data[:5]]
        initial_cys = [d["cy"] for d in raw_data[:5]]
        cam_cx = _st.median(initial_cxs) if initial_cxs else raw_data[0]["cx"]
        cam_cy = _st.median(initial_cys) if initial_cys else raw_data[0]["cy"]
        
        deadzone_px = crop_w * DEADZONE_RATIO
        
        # Consistent aggressive snapping for wide-to-tight camera cuts in standard clips
        temp_snap = SNAP_THRESHOLD if SNAP_THRESHOLD < 0.1 else 0.08
        snap_px = width * temp_snap

        for d in raw_data:
            face_cx = d["cx"]
            face_cy = d["cy"]

            if abs(face_cx - cam_cx) > snap_px:
                cam_cx = face_cx
            else:
                if face_cx > cam_cx + deadzone_px:
                    cam_cx += (face_cx - (cam_cx + deadzone_px)) * SMOOTH_FACTOR
                elif face_cx < cam_cx - deadzone_px:
                    cam_cx += (face_cx - (cam_cx - deadzone_px)) * SMOOTH_FACTOR

            # Vertical smoothing
            cam_cy += (face_cy - cam_cy) * SMOOTH_FACTOR

            smooth_data.append({"time": d["time"], "cx": cam_cx, "cy": cam_cy})

    def get_x(t):
        if not smooth_data:
            return default_cx
        if t <= smooth_data[0]["time"]:
            return smooth_data[0]["cx"]
        if t >= smooth_data[-1]["time"]:
            return smooth_data[-1]["cx"]
        for i in range(len(smooth_data) - 1):
            if smooth_data[i]["time"] <= t <= smooth_data[i + 1]["time"]:
                t1, t2 = smooth_data[i]["time"], smooth_data[i + 1]["time"]
                cx1, cx2 = smooth_data[i]["cx"], smooth_data[i + 1]["cx"]
                if t1 == t2: return cx1
                return cx1 + (cx2 - cx1) * (t - t1) / (t2 - t1)
        return default_cx

    def get_box(t):
        if not raw_data:
            return None
        if t <= raw_data[0]["time"]:
            return raw_data[0]["box"]
        if t >= raw_data[-1]["time"]:
            return raw_data[-1]["box"]
        for i in range(len(raw_data) - 1):
            if raw_data[i]["time"] <= t <= raw_data[i + 1]["time"]:
                return raw_data[i]["box"]
        return None

    def _get_pos(t):
        if not smooth_data:
            return default_cx, default_cy
        if t <= smooth_data[0]["time"]:
            return smooth_data[0]["cx"], smooth_data[0]["cy"]
        if t >= smooth_data[-1]["time"]:
            return smooth_data[-1]["cx"], smooth_data[-1]["cy"]

        for i in range(len(smooth_data) - 1):
            if smooth_data[i]["time"] <= t <= smooth_data[i + 1]["time"]:
                t1, t2 = smooth_data[i]["time"], smooth_data[i + 1]["time"]
                cx1, cx2 = smooth_data[i]["cx"], smooth_data[i + 1]["cx"]
                cy1, cy2 = smooth_data[i]["cy"], smooth_data[i + 1]["cy"]
                if t1 == t2:
                    return cx1, cy1
                frac = (t - t1) / (t2 - t1)
                return (
                    cx1 + (cx2 - cx1) * frac,
                    cy1 + (cy2 - cy1) * frac
                )
        return default_cx, default_cy

    def format_seconds(s):
        mins = int(s) // 60
        secs = int(s % 60)
        return f"{mins:02d}:{secs:02d}"

    # FASE 3: RENDER FRAME
    base_out_w, base_out_h = _get_render_dims(cfg, rasio, source_h=height)
    
    # DEV MODE: Force 16:9 to show context or 2648 ultrawide for merge
    dev_visualize = cfg.dev_mode and rasio == "9:16"
    merge_output = dev_visualize and getattr(cfg, "dev_mode_with_output_merge", False)
    
    if merge_output:
        writer_w, writer_h = 2648, 1220
    elif dev_visualize:
        writer_w, writer_h = 1920, 1080
    else:
        writer_w, writer_h = base_out_w, base_out_h

    writer = open_ffmpeg_video_writer(
        output_video, writer_w, writer_h, orig_fps, video_encoder
    )

    TRANSITION_DUR = 0.3
    MAX_ZOOM = 1.10

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_clip * 1000)
        frame_count = 0
        last_render_percent = -1

        print(f"🎬 {label} - Render frame dimulai...", flush=True)

        while True:
            ret, frame_utama = cap.read()
            if not ret:
                break

            t = frame_count / orig_fps
            if t > duration:
                break

            waktu_absolut = start_clip + t

            # --- 1. ALWAYS CREATE NORMAL 9:16 OUTPUT ---
            if rasio == "9:16":
                cx_base, cy_base = _get_pos(t)
                x1_crop = int(max(0, min(cx_base - crop_w // 2, width - crop_w)))
                y1_crop = int(max(0, min(cy_base - crop_h // 2, height - crop_h)))
                cropped = frame_utama[y1_crop : y1_crop + crop_h, x1_crop : x1_crop + crop_w]
                frame_normal = _resize_frame(cropped, (base_out_w, base_out_h))
            else:
                cx_base, cy_base = default_cx, default_cy
                frame_normal = _resize_frame(frame_utama, (base_out_w, base_out_h))

            # Base target for filtering (e.g. B-Roll applies to the normal output)
            frame_terpilih = frame_normal

            # --- 2. CREATE DEV CONTEXT FRAME IF ACTIVE ---
            frame_dev = None
            if dev_visualize:
                frame_base = _resize_frame(frame_utama, (1920, 1080))
                frame_dev = (frame_base * 0.35).astype(np.uint8)
                
                scale_x = 1920 / width
                scale_y = 1080 / height
                
                cx_dev = int(cx_base * scale_x)
                cy_dev = int(cy_base * scale_y)
                cw_dev = int(crop_w * scale_x)
                ch_dev = int(crop_h * scale_y)
                
                x1 = int(max(0, min(cx_dev - cw_dev // 2, 1920 - cw_dev)))
                y1_dev = int(max(0, min(cy_dev - ch_dev // 2, 1080 - ch_dev)))
                
                # Bright focal crop
                frame_dev[y1_dev : y1_dev + ch_dev, x1 : x1 + cw_dev] = frame_base[y1_dev : y1_dev + ch_dev, x1 : x1 + cw_dev]
                
                # Frame borders
                cv2.rectangle(frame_dev, (x1, y1_dev), (x1+cw_dev, y1_dev+ch_dev), (255, 255, 255), 2)
                
                # Face tracking box & target lines
                if cfg.box_face_detection or cfg.track_lines or True:
                    box = get_box(t)
                    if box:
                        bx1, by1 = int(box[0] * scale_x), int(box[1] * scale_y)
                        bx2, by2 = int(box[2] * scale_x), int(box[3] * scale_y)
                        cv2.rectangle(frame_dev, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                        
                        if cfg.track_lines or cfg.dev_mode:
                            mid_x = (bx1 + bx2) // 2
                            mid_y = (by1 + by2) // 2
                            cv2.line(frame_dev, (x1, mid_y), (bx1, mid_y), (0, 255, 255), 2)
                            cv2.line(frame_dev, (bx2, mid_y), (x1 + cw_dev, mid_y), (0, 255, 255), 2)
                            cv2.line(frame_dev, (mid_x, y1_dev), (mid_x, by1), (0, 255, 255), 2)
                            cv2.line(frame_dev, (mid_x, by2), (mid_x, y1_dev + ch_dev), (0, 255, 255), 2)
                
                # Dev UI HUD Text

                hud_lines = [
                    f"MODE: HYBRID STANDARD (DEV)",
                    f"TIME: {format_seconds(t)}",
                    f"LAYOUT: FULL 9:16",
                    f"ANCHOR CX: {int(cx_base)}"
                ]
                for i, line in enumerate(hud_lines):
                    cv2.putText(frame_dev, line, (40, 60 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # --- 3. B-ROLL OVERLAY OVER NORMAL FRAME ---
            for bc in broll_caps:
                if bc["start"] <= waktu_absolut <= bc["end"]:
                    elapsed_broll = waktu_absolut - bc["start"]
                    bc["cap"].set(cv2.CAP_PROP_POS_MSEC, elapsed_broll * 1000)
                    ret_b, frame_b = bc["cap"].read()

                    if ret_b:
                        durasi_total_broll = bc["end"] - bc["start"]
                        progress_broll = elapsed_broll / durasi_total_broll if durasi_total_broll > 0 else 0
                        zoom_factor = 1.0 + ((MAX_ZOOM - 1.0) * progress_broll)

                        frame_b_crop = crop_center_broll(frame_b, base_out_w, base_out_h)
                        M = cv2.getRotationMatrix2D((base_out_w / 2, base_out_h / 2), 0, zoom_factor)
                        frame_b_zoomed = cv2.warpAffine(frame_b_crop, M, (base_out_w, base_out_h))

                        alpha = 1.0
                        if elapsed_broll < TRANSITION_DUR:
                            alpha = elapsed_broll / TRANSITION_DUR
                        elif (bc["end"] - waktu_absolut) < TRANSITION_DUR:
                            alpha = (bc["end"] - waktu_absolut) / TRANSITION_DUR

                        if alpha >= 1.0:
                            frame_terpilih = frame_b_zoomed
                        else:
                            frame_terpilih = cv2.addWeighted(frame_b_zoomed, alpha, frame_terpilih, 1.0 - alpha, 0)
                    break

            # --- 4. OUTPUT WRITING AND MERGING ---
            if merge_output:
                frm_normal_small = _resize_frame(frame_terpilih, (608, 1080))
                frm_merged = np.full((1220, 2648, 3), 30, dtype=np.uint8)
                
                cv2.putText(frm_merged, "DIRECTOR'S CONSOLE (16:9 RAW)", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(frm_merged, "FINAL OUTPUT (9:16 CROP)", (2000, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                cv2.rectangle(frm_merged, (38, 98), (40+1920+2, 100+1080+2), (255, 255, 255), 4)
                cv2.rectangle(frm_merged, (1998, 98), (2000+608+2, 100+1080+2), (255, 255, 255), 4)
                
                frm_merged[100:1180, 40:1960] = frame_dev
                frm_merged[100:1180, 2000:2608] = frm_normal_small
                
                writer.stdin.write(frm_merged.tobytes())
            elif dev_visualize:
                writer.stdin.write(frame_dev.tobytes())
            else:
                writer.stdin.write(frame_terpilih.tobytes())
            frame_count += 1

            render_percent = (
                min(100, int((t / duration) * 100)) if duration > 0 else 100
            )
            if render_percent != last_render_percent:
                print(
                    f"⏳ {label} - Render frame: {render_percent:3d}% | "
                    f"{format_seconds(t)} / {format_seconds(duration)}",
                    flush=True,
                )
                last_render_percent = render_percent

        writer.stdin.close()
        stderr_data = writer.stderr.read().decode("utf-8", errors="ignore")
        return_code = writer.wait()

        if return_code != 0:
            raise RuntimeError(f"FFmpeg writer gagal: {stderr_data[-1000:]}")

        print(f"✅ {label} selesai.", flush=True)

    finally:
        cap.release()
        for bc in broll_caps:
            bc["cap"].release()
            
    return get_x


