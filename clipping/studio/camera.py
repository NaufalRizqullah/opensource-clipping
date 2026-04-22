"""
clipping.studio.camera — Camera Switch Renderer

Renderer otomatis camera switching berdasarkan active speaker.
Behaviour:
  - One speaker active   → full 9:16 crop centred on that speaker's face
  - Both speakers active → blurred pillarbox (16:9 wide-shot + blurred bg)
  - No speaker active    → stay on last known speaker (or pillarbox if none yet)

Fungsi utama: ``buat_video_camera_switch()``

Dipindahkan dari studio.py tanpa perubahan logic.
"""

import math
import os

import cv2
import mediapipe as mp
import numpy as np

from .utils import (
    detect_video_encoder,
    format_seconds,
    open_ffmpeg_video_writer,
)
from .face import get_face_detector


# ==============================================================================
# CAMERA SWITCH RENDERER
# ==============================================================================


def buat_video_camera_switch(
    input_video,
    output_video,
    start_clip,
    end_clip,
    diarization_data,
    cfg,
    label="CameraSwitch",
):
    """
    Render a full 9:16 video with automatic camera switching based on active speaker.

    Behaviour per frame:
      - One speaker active   → full 9:16 crop centred on that speaker's face
      - Both speakers active → blurred pillarbox (16:9 wide-shot + blurred bg)
      - No speaker active    → stay on last known speaker (or pillarbox if none yet)

    A minimum hold duration (cfg.switch_hold_duration, default 2 s) prevents
    rapid flickering when speakers alternate quickly.

    Parameters
    ----------
    input_video : str
        Path to the source video.
    output_video : str
        Output video path (silent, no audio).
    start_clip, end_clip : float
        Clip time range in seconds.
    diarization_data : list[dict]
        Speaker segments from diarization.run_diarization().
    cfg : SimpleNamespace
        Config object (uses cfg.switch_hold_duration, cfg.face_detector, etc.)
    label : str
        Label for progress logging.
    """
    from ..diarization import get_active_speakers

    STEP_DETEKSI     = cfg.track_step if cfg.track_step is not None else 0.25
    DEADZONE_RATIO   = cfg.track_deadzone if cfg.track_deadzone is not None else 0.15
    SMOOTH_FACTOR    = cfg.track_smooth if cfg.track_smooth is not None else 0.30
    JITTER_THRESHOLD = cfg.track_jitter if cfg.track_jitter is not None else 5
    SNAP_THRESHOLD   = cfg.track_snap if cfg.track_snap is not None else 0.25
    MIN_HOLD = float(getattr(cfg, "switch_hold_duration", 2.0))
    BLUR_KERNEL = 99
    BLUR_SIGMA = 30

    video_encoder = detect_video_encoder()

    # ---------------------------------------------------------------- face detector
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
    duration = end_clip - start_clip

    out_w, out_h = 1080, 1920

    # 9:16 crop region from the source frame
    crop_ratio = out_w / out_h  # ≈ 0.5625
    if (width / height) > crop_ratio:  # source is wider  → crop width
        crop_h = height
        crop_w = int(height * crop_ratio)
    else:  # source is taller → crop height
        crop_w = width
        crop_h = int(width / crop_ratio)

    default_x = (width - crop_w) // 2

    # All unique speakers in this clip
    speakers = sorted(set(s["speaker"] for s in diarization_data))
    if not speakers:
        speakers = ["SPEAKER_00"]

    # ================================================================
    # FASE 1 — Per-speaker face profiling (diarization-guided)
    # ================================================================
    # Strategy:
    #   1 active + 1 face  → trivial: face belongs to active speaker
    #   N active + N faces → sort faces by X; sort speakers by label; assign in order
    #   1 active + N faces → use speaker's last-known position to pick nearest face
    #   otherwise          → skip (ambiguous / no data)

    print(f"🧠 {label} - Analisa wajah (camera switch) dimulai...", flush=True)

    all_frame_data: list[dict] = []  # [{time, face_centers, active_now}]
    speaker_solo_cxs: dict[str, list] = {}  # speaker → [cx, ...] from 1:1 frames
    solo_counts: dict[str, int] = {spk: 0 for spk in speakers}
    multi_counts: dict[str, int] = {spk: 0 for spk in speakers}
    current_time = 0.0
    last_detect_percent = -1

    def _clamp_x_cs(cx_center: float) -> int:
        return max(0, min(int(cx_center - crop_w / 2), width - crop_w))

    while current_time <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, (start_clip + current_time) * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        face_centers = []
        face_boxes = []

        if cfg.face_detector == "yolo":
            yolo_results = yolo_model(frame, verbose=False)
            if yolo_results and len(yolo_results[0].boxes) > 0:
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box
                    face_centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
                    face_boxes.append((x1, y1, x2, y2))
        else:
            results = detector.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
            )
            if results.detections:
                for d in results.detections:
                    bb = d.bounding_box
                    face_centers.append(
                        (
                            bb.origin_x + bb.width / 2,
                            bb.origin_y + bb.height / 2,
                        )
                    )
                    face_boxes.append(
                        (
                            bb.origin_x,
                            bb.origin_y,
                            bb.origin_x + bb.width,
                            bb.origin_y + bb.height,
                        )
                    )

        face_centers.sort(key=lambda fc: fc[0])  # left → right
        active_now = get_active_speakers(diarization_data, start_clip + current_time)
        n_faces = len(face_centers)
        n_active = len(active_now)

        # Pass A: collect face data and build canonical solo observations.
        # Full assignment happens in Pass B after canonical_cx is built from entire clip.
        if n_faces == 1 and n_active == 1:
            spk = active_now[0]
            speaker_solo_cxs.setdefault(spk, []).append(face_centers[0][0])
            solo_counts[spk] = solo_counts.get(spk, 0) + 1
        elif n_faces >= 2 and n_active >= 1:
            for spk in active_now:
                multi_counts[spk] = multi_counts.get(spk, 0) + 1

        all_frame_data.append(
            {
                "time": current_time,
                "face_centers": face_centers,
                "face_boxes": face_boxes,
                "active_now": active_now,
            }
        )

        detect_pct = (
            min(100, int(current_time / duration * 100)) if duration > 0 else 100
        )
        if detect_pct != last_detect_percent:
            print(f"⏳ {label} - Analisa wajah: {detect_pct:3d}%", flush=True)
            last_detect_percent = detect_pct

        current_time += STEP_DETEKSI

    # Build canonical center-X per speaker from 1:1 frames
    import statistics as _stats

    speaker_canonical_cx: dict[str, float] = {
        spk: _stats.median(cxs) for spk, cxs in speaker_solo_cxs.items() if cxs
    }

    if not diarization_data:
        # Visual-only: assume L is left and R is right
        if "FACE_L" not in speaker_canonical_cx:
            speaker_canonical_cx["FACE_L"] = width * 0.25
        if "FACE_R" not in speaker_canonical_cx:
            speaker_canonical_cx["FACE_R"] = width * 0.75

    # Determine which speakers appear mostly in solo-scene frames
    speaker_is_solo: dict[str, bool] = {
        spk: (solo_counts.get(spk, 0) > multi_counts.get(spk, 0))
        for spk in speakers
    }

    # ================================================================
    # Pass B — Canonical-guided face-to-speaker assignment
    # ================================================================
    # Re-process all stored frames using canonical_cx built from entire clip.
    # Multi-scene videos: Speaker C's solo canonical eliminates their face from
    # Scene A frames, leaving the correct face for Scene A speakers.
    # ================================================================
    raw_data: dict[str, list] = {spk: [] for spk in speakers}

    for fd in all_frame_data:
        fc_list = fd["face_centers"]
        act_list = fd["active_now"]
        nf = len(fc_list)
        na = len(act_list)

        if nf == 0 or na == 0:
            continue

        if nf == 1 and na == 1:
            spk = act_list[0]
            if spk in raw_data:
                raw_data[spk].append(
                    {"time": fd["time"], "x": _clamp_x_cs(fc_list[0][0])}
                )

        elif nf >= 2 and na >= 2:
            remaining = list(fc_list)
            for spk in sorted(
                act_list, key=lambda s: speaker_canonical_cx.get(s, width / 2)
            ):
                if not remaining or spk not in raw_data:
                    break
                best = min(
                    remaining,
                    key=lambda fc: abs(
                        fc[0] - speaker_canonical_cx.get(spk, width / 2)
                    ),
                )
                remaining.remove(best)
                raw_data[spk].append({"time": fd["time"], "x": _clamp_x_cs(best[0])})

        elif nf >= 2 and na == 1:
            spk = act_list[0]
            if spk not in raw_data:
                continue
            remaining = list(fc_list)
            for other in speakers:
                if other == spk or other not in speaker_canonical_cx or not remaining:
                    continue
                claimed = min(
                    remaining, key=lambda fc: abs(fc[0] - speaker_canonical_cx[other])
                )
                remaining.remove(claimed)
            pool = remaining if remaining else fc_list
            if raw_data[spk]:
                last_cx = raw_data[spk][-1]["x"] + crop_w / 2
                best = min(pool, key=lambda fc: abs(fc[0] - last_cx))
            elif spk in speaker_canonical_cx:
                best = min(pool, key=lambda fc: abs(fc[0] - speaker_canonical_cx[spk]))
            else:
                best = pool[len(pool) // 2]
            raw_data[spk].append({"time": fd["time"], "x": _clamp_x_cs(best[0])})

    # ================================================================
    # FASE 2 — Smooth per-speaker camera positions
    # ================================================================
    def _smooth_positions(raw_list):
        smooth = []
        if not raw_list:
            return smooth
        cam_x = raw_list[0]["x"]
        deadzone_px = crop_w * DEADZONE_RATIO
        snap_px = width * SNAP_THRESHOLD
        for d in raw_list:
            face_x = d["x"]
            if abs(face_x - cam_x) > snap_px:
                cam_x = face_x
            else:
                if face_x > cam_x + deadzone_px:
                    cam_x += (face_x - (cam_x + deadzone_px)) * SMOOTH_FACTOR
                elif face_x < cam_x - deadzone_px:
                    cam_x += (face_x - (cam_x - deadzone_px)) * SMOOTH_FACTOR
            final_x = int(max(0, min(cam_x, width - crop_w)))
            if smooth and abs(final_x - smooth[-1]["x"]) <= JITTER_THRESHOLD:
                final_x = smooth[-1]["x"]
            smooth.append({"time": d["time"], "x": final_x})
        return smooth

    smooth: dict[str, list] = {
        spk: _smooth_positions(raw_data[spk]) for spk in speakers
    }

    def _get_x(speaker, t):
        sd = smooth.get(speaker, [])
        if not sd:
            return default_x
        if t <= sd[0]["time"]:
            return sd[0]["x"]
        if t >= sd[-1]["time"]:
            return sd[-1]["x"]
        for i in range(len(sd) - 1):
            if sd[i]["time"] <= t <= sd[i + 1]["time"]:
                t1, t2 = sd[i]["time"], sd[i + 1]["time"]
                x1, x2 = sd[i]["x"], sd[i + 1]["x"]
                if t1 == t2:
                    return x1
                return int(x1 + (x2 - x1) * (t - t1) / (t2 - t1))
        return default_x

    # ----------------------------------------------------------------
    # Helper: blurred pillarbox for wide-shot / simultaneous speech
    # ----------------------------------------------------------------
    def _make_blurred_pillarbox(frame):
        h, w = frame.shape[:2]
        # Background: scale to cover 1080×1920, crop centre, then blur
        scale = max(out_w / w, out_h / h)
        new_w = max(out_w, int(w * scale))
        new_h = max(out_h, int(h * scale))
        bg = cv2.resize(frame, (new_w, new_h))
        y0 = (new_h - out_h) // 2
        x0 = (new_w - out_w) // 2
        bg = bg[y0 : y0 + out_h, x0 : x0 + out_w]
        ksize = BLUR_KERNEL if BLUR_KERNEL % 2 == 1 else BLUR_KERNEL + 1
        bg = cv2.GaussianBlur(bg, (ksize, ksize), BLUR_SIGMA)
        # Foreground: scale frame to width=out_w, preserve aspect ratio
        fg_w = out_w
        fg_h = min(out_h, int(h * out_w / w))
        fg = cv2.resize(frame, (fg_w, fg_h))
        # Composite: centre foreground vertically on blurred background
        result = bg.copy()
        y_start = (out_h - fg_h) // 2
        result[y_start : y_start + fg_h, 0:fg_w] = fg
        return result

    def _get_all_boxes(t):
        if not all_frame_data:
            return []
        if t <= all_frame_data[0]["time"]:
            return all_frame_data[0]["face_boxes"]
        if t >= all_frame_data[-1]["time"]:
            return all_frame_data[-1]["face_boxes"]

        for i in range(len(all_frame_data) - 1):
            if all_frame_data[i]["time"] <= t <= all_frame_data[i + 1]["time"]:
                b1s = all_frame_data[i]["face_boxes"]
                b2s = all_frame_data[i + 1]["face_boxes"]
                if len(b1s) != len(b2s):
                    return b1s if abs(t - all_frame_data[i]["time"]) < abs(t - all_frame_data[i+1]["time"]) else b2s
                t1, t2 = all_frame_data[i]["time"], all_frame_data[i + 1]["time"]
                frac = (t - t1) / (t2 - t1)
                res = []
                for b1, b2 in zip(b1s, b2s):
                    res.append((
                        b1[0] + (b2[0] - b1[0]) * frac,
                        b1[1] + (b2[1] - b1[1]) * frac,
                        b1[2] + (b2[2] - b1[2]) * frac,
                        b1[3] + (b2[3] - b1[3]) * frac,
                    ))
                return res
        return []

    # FASE 3: RENDER FRAME
    out_w, out_h = (1080, 1920) # Assume 9:16 target
    
    dev_visualize = cfg.dev_mode # Assume only for 9:16 as described
    if dev_visualize:
        out_w, out_h = (1920, 1080)

    writer = open_ffmpeg_video_writer(
        output_video, out_w, out_h, orig_fps, video_encoder
    )

    current_speaker = None
    last_switch_time = 0.0

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_clip * 1000)
        frame_count = 0
        last_render_percent = -1
        tracking_log = [] # Store (t, cx) for each frame

        print(f"🎬 {label} - Render camera switch dimulai...", flush=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t = frame_count / orig_fps
            if t > duration:
                break

            if cfg.box_face_detection:
                boxes = _get_all_boxes(t)
                for b in boxes:
                    cv2.rectangle(
                        frame,
                        (int(b[0]), int(b[1])),
                        (int(b[2]), int(b[3])),
                        (0, 255, 255),
                        3,
                    )

            timestamp_abs = start_clip + t
            active_speakers = get_active_speakers(diarization_data, timestamp_abs)

            if dev_visualize:
                # Dev visualization for camera-switch
                frame_base = cv2.resize(frame, (out_w, out_h))
                frame_dev = (frame_base * 0.35).astype(np.uint8)
                
                scale_x = out_w / width
                
                # Check what state we are in (Wide or Crop)
                is_wide = False
                if len(active_speakers) >= 2:
                    all_multi_scene = all(not speaker_is_solo.get(spk, False) for spk in active_speakers)
                    if all_multi_scene:
                        is_wide = True
                elif len(active_speakers) == 0 and current_speaker is None:
                    is_wide = True
                
                if is_wide:
                    # Show full frame in dev mode (maybe slightly brightened back or with label)
                    frame_dev = (frame_base * 0.8).astype(np.uint8)
                    cv2.putText(frame_dev, "WIDE SHOT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    # It's a crop on current_speaker
                    if current_speaker is not None:
                        cx = _get_x(current_speaker, t)
                        cx_scaled = int(cx * scale_x)
                        cw_scaled = int(crop_w * scale_x)
                        
                        # Paste bright crop
                        frame_dev[:, cx_scaled : cx_scaled + cw_scaled] = frame_base[:, cx_scaled : cx_scaled + cw_scaled]
                        # Vertical lines
                        cv2.line(frame_dev, (cx_scaled, 0), (cx_scaled, out_h), (255, 255, 255), 2)
                        cv2.line(frame_dev, (cx_scaled + cw_scaled, 0), (cx_scaled + cw_scaled, out_h), (255, 255, 255), 2)
                        
                        label_spk = f"TRACKING: {current_speaker}"
                        cv2.putText(frame_dev, label_spk, (cx_scaled + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Force face boxes in dev mode
                f_boxes = _get_all_boxes(t)
                for fb in f_boxes:
                    scale_y = out_h / height
                    fb1, fb2, fb3, fb4 = int(fb[0]*scale_x), int(fb[1]*scale_y), int(fb[2]*scale_x), int(fb[3]*scale_y)
                    cv2.rectangle(frame_dev, (fb1, fb2), (fb3, fb4), (0, 255, 255), 2)
                    
                    if (cfg.track_lines or cfg.dev_mode) and not is_wide:
                        # Current crop boundaries
                        if current_speaker is not None:
                            cx = _get_x(current_speaker, t)
                            cx_scaled = int(cx * scale_x)
                            cw_scaled = int(crop_w * scale_x)
                            
                            mid_x = (fb1 + fb3) // 2
                            mid_y = (fb2 + fb4) // 2
                            
                            # Horizontal lines
                            cv2.line(frame_dev, (cx_scaled, mid_y), (fb1, mid_y), (0, 255, 255), 2)
                            cv2.line(frame_dev, (fb3, mid_y), (cx_scaled + cw_scaled, mid_y), (0, 255, 255), 2)
                            
                            # Vertical lines
                            cv2.line(frame_dev, (mid_x, 0), (mid_x, fb2), (0, 255, 255), 2)
                            cv2.line(frame_dev, (mid_x, fb4), (mid_x, out_h), (0, 255, 255), 2)
                
                out_frame = frame_dev

            elif len(active_speakers) >= 2:
                # ... standard logic ...
                all_multi_scene = all(
                    not speaker_is_solo.get(spk, False) for spk in active_speakers
                )
                if all_multi_scene:
                    cx = (width - crop_w) // 2
                    out_frame = _make_blurred_pillarbox(frame)
                else:
                    if current_speaker is None or current_speaker not in active_speakers:
                        current_speaker = active_speakers[0]
                        last_switch_time = t
                    cx = _get_x(current_speaker, t)
                    crop_fr = frame[0:crop_h, cx : cx + crop_w]
                    out_frame = cv2.resize(crop_fr, (out_w, out_h))

            elif len(active_speakers) == 1:
                new_speaker = active_speakers[0]
                if current_speaker is None:
                    current_speaker = new_speaker
                    last_switch_time = t
                elif (
                    new_speaker != current_speaker
                    and (t - last_switch_time) >= MIN_HOLD
                ):
                    current_speaker = new_speaker
                    last_switch_time = t
                cx = _get_x(current_speaker, t)
                crop_fr = frame[0:crop_h, cx : cx + crop_w]
                out_frame = cv2.resize(crop_fr, (out_w, out_h))

            else:
                if current_speaker is not None:
                    cx = _get_x(current_speaker, t)
                    crop_fr = frame[0:crop_h, cx : cx + crop_w]
                    out_frame = cv2.resize(crop_fr, (out_w, out_h))
                else:
                    cx = (width - crop_w) // 2 # Center for blurred view
                    out_frame = _make_blurred_pillarbox(frame)

            tracking_log.append((t, cx))
            writer.stdin.write(out_frame.tobytes())
            frame_count += 1

            render_pct = min(100, int(t / duration * 100)) if duration > 0 else 100
            if render_pct != last_render_percent:
                print(
                    f"⏳ {label} - Render camera switch: {render_pct:3d}% | "
                    f"{format_seconds(t)} / {format_seconds(duration)}",
                    flush=True,
                )
                last_render_percent = render_pct

        writer.stdin.close()
        stderr_data = writer.stderr.read().decode("utf-8", errors="ignore")
        return_code = writer.wait()

        if return_code != 0:
            raise RuntimeError(f"FFmpeg writer gagal: {stderr_data[-1000:]}")

        print(f"✅ {label} selesai.", flush=True)

    finally:
        cap.release()
