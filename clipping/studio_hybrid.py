"""
clipping.studio_hybrid — Hybrid Video Renderer

Renderer utama untuk single-speaker video: face-tracking crop (9:16),
B-Roll overlay dengan crossfade dan slow zoom, serta dev mode visualization.

Fungsi utama: ``buat_video_hybrid()``

Dipindahkan dari studio.py tanpa perubahan logic.
"""

import math
import os

import cv2
import mediapipe as mp
import numpy as np

from .studio_utils import (
    detect_video_encoder,
    format_seconds,
    open_ffmpeg_video_writer,
)
from .studio_face import get_face_detector
from .studio_assets import crop_center_broll


# ==============================================================================
# HYBRID VIDEO RENDERER
# ==============================================================================


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
    """Render video hybrid dengan face-tracking crop dan B-Roll overlay.

    Parameters
    ----------
    input_video : str
        Path video sumber.
    output_video : str
        Path output video (silent, tanpa audio).
    start_clip, end_clip : float
        Rentang waktu klip dalam detik.
    rasio : str
        Rasio video (``"9:16"`` atau ``"16:9"``).
    cfg : SimpleNamespace
        Konfigurasi (face_detector, tracking params, dev_mode, dll.).
    broll_data : list[dict] | None
        Data B-Roll (filepath, start_time, end_time).
    label : str
        Label untuk log progress.

    Returns
    -------
    callable
        Fungsi ``get_x(t)`` yang mengembalikan posisi crop X pada waktu ``t``.
        Digunakan oleh subtitle builder untuk alignment di dev mode.
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

    video_encoder = detect_video_encoder()

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
    default_x = (width - crop_w) // 2
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

        best_x = default_x
        face_box = None

        if cfg.face_detector == "yolo":
            yolo_results = yolo_model(frame, verbose=False)
            if yolo_results and len(yolo_results[0].boxes) > 0:
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                largest_idx = areas.argmax()
                x1, y1, x2, y2 = boxes[largest_idx]
                center_x = x1 + (x2 - x1) / 2
                best_x = center_x - (crop_w / 2)
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
                best_x = (largest_face.origin_x + (largest_face.width / 2)) - (
                    crop_w // 2
                )
                face_box = (
                    largest_face.origin_x,
                    largest_face.origin_y,
                    largest_face.origin_x + largest_face.width,
                    largest_face.origin_y + largest_face.height,
                )

        raw_data.append(
            {
                "time": current_time,
                "x": max(0, min(best_x, width - crop_w)),
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
        cam_x = raw_data[0]["x"]
        deadzone_px = crop_w * DEADZONE_RATIO
        snap_px = width * SNAP_THRESHOLD

        for d in raw_data:
            face_x = d["x"]

            if abs(face_x - cam_x) > snap_px:
                cam_x = face_x
            else:
                if face_x > cam_x + deadzone_px:
                    cam_x += (face_x - (cam_x + deadzone_px)) * SMOOTH_FACTOR
                elif face_x < cam_x - deadzone_px:
                    cam_x += (face_x - (cam_x - deadzone_px)) * SMOOTH_FACTOR

            final_x = int(max(0, min(cam_x, width - crop_w)))
            if smooth_data and abs(final_x - smooth_data[-1]["x"]) <= JITTER_THRESHOLD:
                final_x = smooth_data[-1]["x"]

            smooth_data.append({"time": d["time"], "x": final_x})

    def get_box(t):
        if not raw_data:
            return None
        if t <= raw_data[0]["time"]:
            return raw_data[0]["box"]
        if t >= raw_data[-1]["time"]:
            return raw_data[-1]["box"]

        for i in range(len(raw_data) - 1):
            if raw_data[i]["time"] <= t <= raw_data[i + 1]["time"]:
                b1 = raw_data[i]["box"]
                b2 = raw_data[i + 1]["box"]
                if b1 is None or b2 is None:
                    return b1 if b1 else b2
                t1, t2 = raw_data[i]["time"], raw_data[i + 1]["time"]
                frac = (t - t1) / (t2 - t1)
                return (
                    b1[0] + (b2[0] - b1[0]) * frac,
                    b1[1] + (b2[1] - b1[1]) * frac,
                    b1[2] + (b2[2] - b1[2]) * frac,
                    b1[3] + (b2[3] - b1[3]) * frac,
                )
        return None

    def get_x(t):
        if not smooth_data:
            return default_x
        if t <= smooth_data[0]["time"]:
            return smooth_data[0]["x"]
        if t >= smooth_data[-1]["time"]:
            return smooth_data[-1]["x"]

        for i in range(len(smooth_data) - 1):
            if smooth_data[i]["time"] <= t <= smooth_data[i + 1]["time"]:
                t1, t2 = smooth_data[i]["time"], smooth_data[i + 1]["time"]
                x1, x2 = smooth_data[i]["x"], smooth_data[i + 1]["x"]
                if t1 == t2:
                    return x1
                return int(x1 + (x2 - x1) * (t - t1) / (t2 - t1))

        return default_x

    # FASE 3: RENDER FRAME
    out_w, out_h = (1080, 1920) if rasio == "9:16" else (1920, 1080)
    
    # DEV MODE: Force 16:9 to show context
    dev_visualize = cfg.dev_mode and rasio == "9:16"
    if dev_visualize:
        out_w, out_h = (1920, 1080)

    writer = open_ffmpeg_video_writer(
        output_video, out_w, out_h, orig_fps, video_encoder
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

            if dev_visualize:
                # Render full context for dev mode
                cx = get_x(t)
                # Resize original to 16:9 canvas
                frame_base = cv2.resize(frame_utama, (out_w, out_h))
                
                # Dim background
                frame_dev = (frame_base * 0.35).astype(np.uint8)
                
                # Highlight 9:16 window
                # Current source width/height vs canvas out_w/out_h
                scale_x = out_w / width
                cx_scaled = int(cx * scale_x)
                cw_scaled = int(crop_w * scale_x)
                
                # Paste bright crop
                frame_dev[:, cx_scaled : cx_scaled + cw_scaled] = frame_base[:, cx_scaled : cx_scaled + cw_scaled]
                
                # Draw vertical border lines
                cv2.line(frame_dev, (cx_scaled, 0), (cx_scaled, out_h), (255, 255, 255), 2)
                cv2.line(frame_dev, (cx_scaled + cw_scaled, 0), (cx_scaled + cw_scaled, out_h), (255, 255, 255), 2)
                
                # Draw face box if detected
                if cfg.box_face_detection or cfg.track_lines or True: # Force in dev mode
                    box = get_box(t)
                    if box:
                        scale_y = out_h / height
                        bx1, by1 = int(box[0] * scale_x), int(box[1] * scale_y)
                        bx2, by2 = int(box[2] * scale_x), int(box[3] * scale_y)
                        cv2.rectangle(frame_dev, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                        
                        if cfg.track_lines or cfg.dev_mode:
                            # Center points of sides
                            mid_x = (bx1 + bx2) // 2
                            mid_y = (by1 + by2) // 2
                            
                            # Horizontal lines to 9:16 boundaries
                            cv2.line(frame_dev, (cx_scaled, mid_y), (bx1, mid_y), (0, 255, 255), 2)
                            cv2.line(frame_dev, (bx2, mid_y), (cx_scaled + cw_scaled, mid_y), (0, 255, 255), 2)
                            
                            # Vertical lines to frame boundaries (top/bottom)
                            cv2.line(frame_dev, (mid_x, 0), (mid_x, by1), (0, 255, 255), 2)
                            cv2.line(frame_dev, (mid_x, by2), (mid_x, out_h), (0, 255, 255), 2)
                
                frame_terpilih = frame_dev
                frame_utama_siap = frame_dev # Ensure this is defined for B-roll transitions

            elif rasio == "9:16":
                cx = get_x(t)
                cropped = frame_utama[:, cx : cx + crop_w]
                frame_utama_siap = cv2.resize(cropped, (out_w, out_h))
                frame_terpilih = frame_utama_siap
            else:
                frame_utama_siap = cv2.resize(frame_utama, (out_w, out_h))
                frame_terpilih = frame_utama_siap

            for bc in broll_caps:
                if bc["start"] <= waktu_absolut <= bc["end"]:
                    elapsed_broll = waktu_absolut - bc["start"]
                    bc["cap"].set(cv2.CAP_PROP_POS_MSEC, elapsed_broll * 1000)
                    ret_b, frame_b = bc["cap"].read()

                    if ret_b:
                        durasi_total_broll = bc["end"] - bc["start"]
                        progress_broll = (
                            elapsed_broll / durasi_total_broll
                            if durasi_total_broll > 0
                            else 0
                        )
                        zoom_factor = 1.0 + ((MAX_ZOOM - 1.0) * progress_broll)

                        frame_b_crop = crop_center_broll(frame_b, out_w, out_h)
                        center_x, center_y = out_w / 2, out_h / 2
                        M = cv2.getRotationMatrix2D(
                            (center_x, center_y), 0, zoom_factor
                        )
                        frame_b_zoomed = cv2.warpAffine(frame_b_crop, M, (out_w, out_h))

                        alpha = 1.0
                        if elapsed_broll < TRANSITION_DUR:
                            alpha = elapsed_broll / TRANSITION_DUR
                        elif (bc["end"] - waktu_absolut) < TRANSITION_DUR:
                            alpha = (bc["end"] - waktu_absolut) / TRANSITION_DUR

                        if alpha >= 1.0:
                            frame_terpilih = frame_b_zoomed
                        else:
                            frame_terpilih = cv2.addWeighted(
                                frame_b_zoomed, alpha, frame_utama_siap, 1.0 - alpha, 0
                            )

                    break

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
