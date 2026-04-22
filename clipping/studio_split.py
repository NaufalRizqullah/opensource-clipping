"""
clipping.studio_split — Split-Screen Renderer (Podcast 2 Speaker)

Renderer untuk format split-screen (atas-bawah) di video 9:16 untuk podcast
dengan 2+ speaker. Mendukung dynamic layout switching (split ↔ full),
scene cut detection, dev mode visualization, dan dual/merge output.

Fungsi utama: ``buat_video_split_screen()``

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


# ==============================================================================
# SPLIT-SCREEN RENDERER (PODCAST 2 SPEAKER)
# ==============================================================================


def buat_video_split_screen(
    input_video,
    output_video,
    start_clip,
    end_clip,
    diarization_data,
    cfg,
    label="SplitScreen",
):
    """
    Render a split-screen (top-bottom) 9:16 video for 2-speaker podcasts.

    Each panel shows one speaker's face-tracked crop. The active speaker's
    panel is brighter; the inactive speaker's panel has a subtle dark overlay.

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
        Configuration object.
    label : str
        Label for progress logging.

    Returns
    -------
    callable
        Fungsi ``get_x_final(t)`` untuk tracking posisi subtitle.
    """
    from .diarization import get_active_speaker, get_active_speakers

    STEP_DETEKSI     = cfg.track_step if cfg.track_step is not None else 0.25
    DEADZONE_RATIO   = cfg.track_deadzone if cfg.track_deadzone is not None else 0.15
    SMOOTH_FACTOR    = cfg.track_smooth if cfg.track_smooth is not None else 0.30
    JITTER_THRESHOLD = cfg.track_jitter if cfg.track_jitter is not None else 5
    SNAP_THRESHOLD   = cfg.track_snap if cfg.track_snap is not None else 0.25
    DIVIDER_HEIGHT = 4  # px, divider between panels
    INACTIVE_ALPHA = 0.15  # darkening for inactive speaker panel
    ACTIVE_BORDER = 3  # px, highlight border for active speaker

    video_encoder = detect_video_encoder()

    # Setup face detector
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

    # Output dimensions: 1080x1920 for 9:16
    out_w, out_h = 1080, 1920
    out_w_final, out_h_final = out_w, out_h
    
    dev_visualize = cfg.dev_mode or cfg.dev_mode_with_output or cfg.dev_mode_with_output_merge
    dual_output = cfg.dev_mode_with_output
    merge_output = cfg.dev_mode_with_output_merge
    
    if merge_output:
        # Merged Full Padded Canvas: 2648 x 1220
        out_w_final, out_h_final = 2648, 1220
    elif dev_visualize and not dual_output:
        # Pure Dev Mode (override output)
        out_w_final, out_h_final = 1920, 1080
    
    # Calculate panel dimensions based on the 1080x1920 orientation
    # regardless of whether dev mode is on, because the internal layout arithmetic
    # must still think in terms of the target 9:16 portrait canvas.
    panel_h = (1920 - DIVIDER_HEIGHT) // 2
    panel_w = 1080

    # --- Panel Crop dimensions (wider aspect for half-height panels) ---
    panel_ratio = panel_w / panel_h
    if (width / height) > panel_ratio:
        crop_h = height
        crop_w = int(height * panel_ratio)
    else:
        crop_w = width
        crop_h = int(width / panel_ratio)

    # --- Full 9:16 Crop dimensions (for solo mode) ---
    full_ratio = out_w / out_h
    if (width / height) > full_ratio:
        crop_h_full = height
        crop_w_full = int(height * full_ratio)
    else:
        crop_w_full = width
        crop_h_full = int(width / full_ratio)

    default_x = (width - crop_w) // 2
    default_y = (height - crop_h) // 2
    default_x_full = (width - crop_w_full) // 2

    # ----------------------------------------------------------------
    # Determine top/bottom panel speakers
    # ----------------------------------------------------------------
    if diarization_data:
        all_speakers_in_clip = sorted(set(s["speaker"] for s in diarization_data))
        speaking_time: dict[str, float] = {spk: 0.0 for spk in all_speakers_in_clip}
        for seg in diarization_data:
            eff_start = max(seg["start"], start_clip)
            eff_end = min(seg["end"], end_clip)
            if eff_end > eff_start:
                speaking_time[seg["speaker"]] += eff_end - eff_start

        ranked = sorted(all_speakers_in_clip, key=lambda s: speaking_time[s], reverse=True)
        speaker_top = ranked[0]
        speaker_bottom = ranked[1] if len(ranked) > 1 else ranked[0]
        extra_speakers = ranked[2:]
    else:
        # Visual-only mode: use generic labels
        all_speakers_in_clip = ["FACE_L", "FACE_R"]
        speaker_top = "FACE_L"
        speaker_bottom = "FACE_R"
        ranked = all_speakers_in_clip
        extra_speakers = []

    # ---- FASE 1: DETECT ALL FACES & ASSIGN TO SPEAKERS (diarization-guided) ----
    # Strategy:
    #   1 active + 1 face  → trivial: face belongs to active speaker
    #   N active + N faces → sort faces by X; sort active speakers by label; assign in order
    #   1 active + N faces → use speaker's last-known position to pick nearest face
    #   otherwise          → skip (ambiguous or no data)

    print(f"🧠 {label} - Analisa wajah (split-screen) dimulai...", flush=True)

    all_frame_data: list[dict] = []  # [{time, face_centers, face_boxes, active_now}]
    speaker_solo_cxs: dict[str, list] = {}  # speaker → [cx, ...] from 1:1 frames
    solo_counts: dict[str, int] = {spk: 0 for spk in all_speakers_in_clip}
    multi_counts: dict[str, int] = {spk: 0 for spk in all_speakers_in_clip}
    current_time = 0.0
    last_detect_percent = -1

    def _clamp_x(cx_center: float) -> int:
        return max(0, min(int(cx_center - crop_w / 2), width - crop_w))

    while current_time <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, (start_clip + current_time) * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        face_centers = []
        face_boxes = []

        if cfg.face_detector == "yolo":
            # Higher confidence to filter background noise (microphones, reflections)
            det_conf = getattr(cfg, "track_conf", 0.55)
            yolo_results = yolo_model(frame, verbose=False, conf=det_conf)
            if yolo_results and len(yolo_results[0].boxes) > 0:
                raw_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                
                # IoU filter to merge overlapping boxes for one person
                def compute_iou(b1, b2):
                    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
                    area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
                    return inter / (area1 + area2 - inter + 1e-6)

                final_boxes = []
                for rb in raw_boxes:
                    merged = False
                    for i, fb in enumerate(final_boxes):
                        iou_thresh = getattr(cfg, "track_iou_threshold", 0.2)
                        if compute_iou(rb, fb) > iou_thresh: # More aggressive merging
                            # Keep the larger box
                            if (rb[2]-rb[0])*(rb[3]-rb[1]) > (fb[2]-fb[0])*(fb[3]-fb[1]):
                                final_boxes[i] = rb
                            merged = True
                            break
                    if not merged:
                        final_boxes.append(rb)

                for box in final_boxes:
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    face_centers.append((cx, cy))
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
                    cx = bb.origin_x + bb.width / 2
                    cy = bb.origin_y + bb.height / 2
                    face_centers.append((cx, cy))
                    face_boxes.append(
                        (
                            bb.origin_x,
                            bb.origin_y,
                            bb.origin_x + bb.width,
                            bb.origin_y + bb.height,
                        )
                    )

        face_centers.sort(key=lambda fc: fc[0])  # left → right
        
        if diarization_data:
            from .diarization import get_active_speakers
            active_now = get_active_speakers(diarization_data, start_clip + current_time)
        else:
            # Visual-only: base active speakers on face count
            if len(face_centers) == 1:
                active_now = ["FACE_L"]
            elif len(face_centers) >= 2:
                active_now = ["FACE_L", "FACE_R"]
            else:
                active_now = []

        n_faces = len(face_centers)
        n_active = len(active_now)

        # Pass A: collect face data and track canonical solo observations.
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

        detect_percent = (
            min(100, int((current_time / duration) * 100)) if duration > 0 else 100
        )
        if detect_percent != last_detect_percent:
            print(f"⏳ {label} - Analisa wajah: {detect_percent:3d}%", flush=True)
            last_detect_percent = detect_percent

        current_time += STEP_DETEKSI

    # Build canonical center-X per speaker from 1:1 frames (median, robust to outliers)
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
        for spk in all_speakers_in_clip
    }

    # ================================================================
    # Pass B — Canonical-guided face-to-speaker assignment
    # ================================================================
    # Re-process all stored frames using canonical_cx built from the entire clip.
    # This handles multi-scene videos correctly: e.g. in a 3-speaker podcast where
    # Speaker C's solo canonical_cx (center of Scene B) can be used to *eliminate*
    # that face from Scene A frames, leaving the correct face for Speaker A/B.
    # ================================================================
    raw_data: dict[str, list] = {spk: [] for spk in all_speakers_in_clip}

    # Helper for visual-only assignment
    def _get_canonical_x(spk):
        if diarization_data:
            return speaker_canonical_cx.get(spk, width / 2)
        else:
            return 0 if spk == "FACE_L" else width

    for fd in all_frame_data:
        fc_list = fd["face_centers"]  # list of (cx, cy), sorted left → right
        act_list = fd["active_now"]
        nf = len(fc_list)
        na = len(act_list)

        if nf == 0 or na == 0:
            continue

        if nf == 1 and na == 1:
            spk = act_list[0]
            if spk in raw_data:
                raw_data[spk].append({"time": fd["time"], "cx": fc_list[0][0]})

        elif nf >= 2 and na >= 2:
            # Greedy canonical assignment: sort active speakers by canonical X,
            # each picks their nearest remaining face.
            remaining = list(fc_list)
            for spk in sorted(
                act_list, key=lambda s: _get_canonical_x(s)
            ):
                if not remaining or spk not in raw_data:
                    break
                best = min(
                    remaining,
                    key=lambda fc: abs(
                        fc[0] - _get_canonical_x(spk)
                    ),
                )
                remaining.remove(best)
                raw_data[spk].append({"time": fd["time"], "cx": best[0]})

        elif nf >= 2 and na == 1:
            spk = act_list[0]
            if spk not in raw_data:
                continue
            # Step 1: eliminate faces claimed by OTHER speakers with known canonical positions
            remaining = list(fc_list)
            for other in all_speakers_in_clip:
                if other == spk or other not in speaker_canonical_cx or not remaining:
                    continue
                claimed = min(
                    remaining, key=lambda fc: abs(fc[0] - speaker_canonical_cx[other])
                )
                remaining.remove(claimed)
            # Step 2: from remaining (or full list if exhausted), pick best for this speaker
            pool = remaining if remaining else fc_list
            if raw_data[spk]:
                last_cx = raw_data[spk][-1]["x"] + crop_w / 2
                best = min(pool, key=lambda fc: abs(fc[0] - last_cx))
            elif spk in speaker_canonical_cx:
                best = min(pool, key=lambda fc: abs(fc[0] - speaker_canonical_cx[spk]))
            else:
                best = pool[len(pool) // 2]
            raw_data[spk].append({"time": fd["time"], "cx": best[0]})

    # ---- FASE 2: SMOOTH CAMERA PER SPEAKER (Centering CX) ----
    def _smooth_positions(raw_data):
        smooth = []
        if not raw_data:
            return smooth
        cam_cx = raw_data[0]["cx"]
        # Use crop_w_full (the 9:16 solo width) for deadzone to match standard behavior
        deadzone_px = crop_w_full * DEADZONE_RATIO
        snap_px = width * SNAP_THRESHOLD

        for d in raw_data:
            face_cx = d["cx"]
            if abs(face_cx - cam_cx) > snap_px:
                cam_cx = face_cx
            else:
                if face_cx > cam_cx + deadzone_px:
                    cam_cx += (face_cx - (cam_cx + deadzone_px)) * SMOOTH_FACTOR
                elif face_cx < cam_cx - deadzone_px:
                    cam_cx += (face_cx - (cam_cx - deadzone_px)) * SMOOTH_FACTOR

            final_cx = cam_cx
            # No jitter clamping here, we'll do it during render-time clamping
            smooth.append({"time": d["time"], "cx": final_cx})
        return smooth

    smooth: dict[str, list] = {
        spk: _smooth_positions(raw_data[spk]) for spk in all_speakers_in_clip
    }

    def _get_cx(speaker: str, t: float) -> float:
        sd = smooth.get(speaker, [])
        if not sd:
            return width / 2
        if t <= sd[0]["time"]:
            return sd[0]["cx"]
        if t >= sd[-1]["time"]:
            return sd[-1]["cx"]
        for i in range(len(sd) - 1):
            if sd[i]["time"] <= t <= sd[i + 1]["time"]:
                t1, t2 = sd[i]["time"], sd[i + 1]["time"]
                cx1, cx2 = sd[i]["cx"], sd[i + 1]["cx"]
                if t1 == t2:
                    return cx1
                return cx1 + (cx2 - cx1) * (t - t1) / (t2 - t1)
        return width / 2

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
                
                # Simple approach: if counts match, interpolate. Else just return nearest.
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

    # ---- FASE 3: RENDER FRAMES ----
    # Determine outputs needed
    writer_main = None
    writer_dev = None
    
    if dual_output:
        # Two separate files need to be written simultaneously
        vid_main = output_video
        vid_dev = output_video.replace(".ts", "_dev.ts").replace(".mp4", "_dev.mp4")
        writer_main = open_ffmpeg_video_writer(vid_main, 1080, 1920, orig_fps, video_encoder)
        writer_dev = open_ffmpeg_video_writer(vid_dev, 1920, 1080, orig_fps, video_encoder)
    else:
        # Only one stream (either standard, pure dev, or merged)
        writer_main = open_ffmpeg_video_writer(output_video, out_w_final, out_h_final, orig_fps, video_encoder)

    # Pre-create overlay for inactive speaker
    dark_overlay = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    # Per-speaker frozen frame cache: stores last valid crop for each speaker
    # so any panel can fallback when its speaker is not in the current scene
    last_valid_crop: dict[str, np.ndarray] = {}

    current_layout = "split"
    current_speaker = None
    MIN_HOLD = float(getattr(cfg, "switch_hold_duration", 2.0))
    # Initialize with a large negative time to allow instant layout decision at t=0
    last_switch_time = -MIN_HOLD
    
    # Stability window for layout decisions (Majority Vote of face counts)
    LAYOUT_SMOOTH_WINDOW = getattr(cfg, "track_smooth_window", 12)
    face_count_history = []
    # (MIN_HOLD is already initialized above)
    is_dynamic = getattr(cfg, "use_dynamic_split", False)
    
    # Scene cut detection state
    prev_small_gray = None
    SCENE_CUT_THRESHOLD = getattr(cfg, "scene_cut_threshold", 18) # Lower = more sensitive to camera cuts

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_clip * 1000)
        frame_count = 0
        last_render_percent = -1

        print(f"🎬 {label} - Render split-screen {'(dynamic)' if is_dynamic else ''} dimulai...", flush=True)
        tracking_log = [] # Store (t, cx) for subtitle tracking

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

            # --- Scene Cut Detection ---
            # Lightweight check: if pixels change drastically, clear stability history to allow instant switch
            curr_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
            if prev_small_gray is not None:
                diff = cv2.absdiff(curr_small, prev_small_gray)
                avg_diff = np.mean(diff)
                if avg_diff > SCENE_CUT_THRESHOLD:
                    face_count_history.clear()
                    # Also temporarily lower MIN_HOLD requirement for this frame to cut instantly
                    last_switch_time = t - MIN_HOLD 
            prev_small_gray = curr_small

            timestamp_abs = start_clip + t
            from .diarization import get_active_speakers
            active_speakers = get_active_speakers(diarization_data, timestamp_abs)
            active_speaker = get_active_speaker(diarization_data, timestamp_abs)

            # --- Layout decision logic ---
            if is_dynamic:
                if cfg.split_trigger == "face":
                    now_boxes = _get_all_boxes(t)
                    now_count = len(now_boxes)
                    face_count_history.append(now_count)
                    if len(face_count_history) > LAYOUT_SMOOTH_WINDOW:
                        face_count_history.pop(0)
                    
                    # Majority vote face count
                    if face_count_history:
                        stable_count = max(set(face_count_history), key=face_count_history.count)
                    else:
                        stable_count = now_count

                    # FAST PATH: split→full is instant (no ghost frames)
                    # When currently in split and this frame sees only 1 face,
                    # force immediate switch without majority vote or MIN_HOLD.
                    # Rationale: a split layout showing 1 person looks broken;
                    # the transition must be imperceptible.
                    if current_layout == "split" and now_count == 1:
                        current_layout = "full"
                        current_speaker = ranked[0]
                        last_switch_time = t
                        face_count_history.clear()  # Reset for clean state
                    else:
                        # Normal path: use stable_count (guarded by MIN_HOLD)
                        if stable_count == 1:
                            target_layout = "full"
                            target_speaker = ranked[0]
                        elif stable_count >= 2:
                            target_layout = "split"
                            target_speaker = ranked[0]
                        else:
                            target_layout = current_layout
                            target_speaker = current_speaker

                        if target_layout != current_layout and (t - last_switch_time) >= MIN_HOLD:
                            current_layout = target_layout
                            current_speaker = target_speaker
                            last_switch_time = t
                else:
                    if len(active_speakers) == 1:
                        target_layout = "full"
                        target_speaker = active_speakers[0]
                    elif len(active_speakers) >= 2:
                        target_layout = "split"
                        target_speaker = active_speakers[0] # Not used in split but for tracking
                    else:
                        target_layout = current_layout # Stay put
                        target_speaker = current_speaker
                
                    if target_layout != current_layout and (t - last_switch_time) >= MIN_HOLD:
                        current_layout = target_layout
                        current_speaker = target_speaker
                        last_switch_time = t
                
                if current_layout == "full":
                    # Switch speaker if audio trigger is active, or stay on tracked
                    if not cfg.split_trigger == "face":
                        if len(active_speakers) == 1 and active_speakers[0] != current_speaker:
                            if (t - last_switch_time) >= MIN_HOLD:
                                current_speaker = active_speakers[0]
                                last_switch_time = t
            else:
                current_layout = "split"

            if current_layout == "full":
                # Solo mode: Render full 9:16 crop
                spk = current_speaker or (speaker_top if speaker_top in ranked else ranked[0])
                smooth_cx = _get_cx(spk, t)
                # Calculate Top-Left X for full 9:16 crop
                x_full = int(max(0, min(smooth_cx - crop_w_full / 2, width - crop_w_full)))
                
                crop = frame[0:crop_h_full, x_full : x_full + crop_w_full]
                final_frame = cv2.resize(crop, (out_w, out_h))
                # Log cx for subtitles to follow
                tracking_log.append((t, smooth_cx))
            else:
                # Split mode: existing logic
                if active_speaker and active_speaker not in (speaker_top, speaker_bottom):
                    panel_top_spk = active_speaker
                    panel_bottom_spk = speaker_bottom
                else:
                    panel_top_spk = speaker_top
                    panel_bottom_spk = speaker_bottom

                # Detect whether the current scene is a solo shot
                in_solo_scene = bool(active_speaker and speaker_is_solo.get(active_speaker, False))

                # ---- Helper: build a panel crop for a given speaker ----
                def _build_panel(spk, is_other_panel_spk=False):
                    spk_not_visible = (
                        active_speaker in all_speakers_in_clip
                        and active_speaker != spk
                        and (speaker_is_solo.get(spk, False) or (in_solo_scene and not speaker_is_solo.get(spk, False)))
                    )
                    if spk_not_visible:
                        if spk in last_valid_crop:
                            return last_valid_crop[spk].copy(), True
                        else:
                            return cv2.resize(frame[0:crop_h, default_x : default_x + crop_w], (panel_w, panel_h)), True
                    else:
                        smooth_cx = _get_cx(spk, t)
                        # Calculate Top-Left X for wide panel crop
                        x_panel = int(max(0, min(smooth_cx - crop_w / 2, width - crop_w)))
                        crop = cv2.resize(frame[0:crop_h, x_panel : x_panel + crop_w], (panel_w, panel_h))
                        if not in_solo_scene or active_speaker == spk:
                            last_valid_crop[spk] = crop.copy()
                        return crop, False

                # ---- Top panel ----
                panel_top, is_fallback_top = _build_panel(panel_top_spk)

                # ---- Bottom panel ----
                panel_bottom, is_fallback_bottom = _build_panel(panel_bottom_spk)

                # ---- Active / inactive highlighting ----
                if active_speaker == panel_top_spk:
                    panel_bottom = cv2.addWeighted(panel_bottom, 1.0 - INACTIVE_ALPHA, dark_overlay, INACTIVE_ALPHA, 0)
                    cv2.rectangle(panel_top, (0, 0), (panel_w - 1, panel_h - 1), (0, 255, 255), ACTIVE_BORDER)
                elif active_speaker == panel_bottom_spk:
                    panel_top = cv2.addWeighted(panel_top, 1.0 - INACTIVE_ALPHA, dark_overlay, INACTIVE_ALPHA, 0)
                    cv2.rectangle(panel_bottom, (0, 0), (panel_w - 1, panel_h - 1), (0, 255, 255), ACTIVE_BORDER)
                else:
                    if is_fallback_top:
                        panel_top = cv2.addWeighted(panel_top, 1.0 - INACTIVE_ALPHA, dark_overlay, INACTIVE_ALPHA, 0)
                    if is_fallback_bottom:
                        panel_bottom = cv2.addWeighted(panel_bottom, 1.0 - INACTIVE_ALPHA, dark_overlay, INACTIVE_ALPHA, 0)

                # Compose the final frame
                divider = np.full((DIVIDER_HEIGHT, panel_w, 3), 80, dtype=np.uint8)
                final_frame = np.vstack([panel_top, divider, panel_bottom])
                # Log neutral center for split mode subtitles
                tracking_log.append((t, width / 2))


            # Ensure exact output dimensions
            if not dev_visualize:
                if final_frame.shape[0] != out_h_final or final_frame.shape[1] != out_w_final:
                    final_frame = cv2.resize(final_frame, (out_w_final, out_h_final))
            else:
                # --- DIRECTOR'S CONSOLE (DEV MODE) ---
                # UI Constants
                HUD_COLOR = (0, 255, 0)
                HUD_X, HUD_Y = 30, 50
                
                # Base frame: 1920x1080 landscape
                frame_res = cv2.resize(frame, (1920, 1080))
                frame_dev = (frame_res * 0.35).astype(np.uint8) # Dim background
                
                scale_x = 1920 / width
                scale_y = 1080 / height
                
                # PREcalculate coordinates for BOTH layouts
                # 1. solo (9:16)
                spk_solo = current_speaker or (speaker_top if speaker_top in ranked else ranked[0])
                cx_solo = _get_cx(spk_solo, t)
                cx_s_scaled = int(cx_solo * scale_x)
                cw_s_scaled = int(crop_w_full * scale_x)
                x1s = max(0, cx_s_scaled - cw_s_scaled // 2)
                x2s = min(1919, x1s + cw_s_scaled)
                
                # 2. split boxes (horizontal)
                cx_split = width / 2
                cx_p_scaled = int(cx_split * scale_x)
                cw_p_scaled = int(crop_w * scale_x)
                x1p = max(0, cx_p_scaled - cw_p_scaled // 2)
                x2p = min(1919, x1p + cw_p_scaled)
                mid_h = (1080 - DIVIDER_HEIGHT) // 2
                
                # --- APPLY CLEAR WINDOW (Active) ---
                if current_layout == "full":
                    # Clear solo window
                    frame_dev[0:1080, x1s:x2s] = frame_res[0:1080, x1s:x2s]
                else:
                    # Clear split windows
                    frame_dev[0:mid_h, x1p:x2p] = frame_res[0:mid_h, x1p:x2p]
                    frame_dev[mid_h + DIVIDER_HEIGHT:1080, x1p:x2p] = frame_res[mid_h + DIVIDER_HEIGHT:1080, x1p:x2p]

                # --- DRAW BOXES (Both) ---
                # Inactive boxes are drawn with lower opacity or thinner lines
                color_solo = (255, 255, 255) if current_layout == "full" else (100, 100, 100)
                color_split = (255, 255, 255) if current_layout == "split" else (100, 100, 100)
                thick_solo = 3 if current_layout == "full" else 1
                thick_split = 2 if current_layout == "split" else 1

                cv2.rectangle(frame_dev, (x1s, 0), (x2s, 1079), color_solo, thick_solo)
                cv2.rectangle(frame_dev, (x1p, 0), (x2p, mid_h), color_split, thick_split)
                cv2.rectangle(frame_dev, (x1p, mid_h + DIVIDER_HEIGHT), (x2p, 1079), color_split, thick_split)

                # Labels
                if current_layout == "full":
                    cv2.putText(frame_dev, f"ACTIVE SOLO: {spk_solo}", (x1s + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    cv2.putText(frame_dev, "ACTIVE SPLIT", (x1p + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Draw all detected faces (on top of clearing)
                all_boxes = _get_all_boxes(t)
                for b in all_boxes:
                    bx1, by1, bx2, by2 = int(b[0]*scale_x), int(b[1]*scale_y), int(b[2]*scale_x), int(b[3]*scale_y)
                    cv2.rectangle(frame_dev, (bx1, by1), (bx2, by2), (0, 255, 255), 2)

                # HUD Info
                now_count = face_count_history[-1] if face_count_history else 0
                diff_val = avg_diff if 'avg_diff' in locals() else 0
                
                hud_lines = [
                    f"MODE: DYNAMIC SPLIT (DEV)",
                    f"TIME: {format_seconds(t)}",
                    f"LAYOUT: {current_layout.upper()}",
                    f"FACES (NOW): {now_count} | STABLE: {stable_count}",
                    f"SCENE DIFF: {diff_val:.1f} (Thr: {SCENE_CUT_THRESHOLD})",
                ]
                
                # Scene cut alert
                if diff_val > SCENE_CUT_THRESHOLD:
                    hud_lines[-1] += " >> RESET! <<"
                
                # Hold status
                hold_rem = max(0, MIN_HOLD - (t - last_switch_time))
                if hold_rem > 0:
                    hud_lines.append(f"SWITCH HOLD: {hold_rem:.1f}s")
                else:
                    hud_lines.append(f"SWITCH HOLD: READY")
                
                for i, line in enumerate(hud_lines):
                    cv2.putText(frame_dev, line, (HUD_X, HUD_Y + i*35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, HUD_COLOR, 2)

            # --- DUAL ROUTING LOGIC ---
            if dual_output:
                # Resize normal output as base logic does not guarantee 1080x1920 if resolution is off
                if final_frame.shape[0] != 1920 or final_frame.shape[1] != 1080:
                    frm_normal = cv2.resize(final_frame, (1080, 1920))
                else:
                    frm_normal = final_frame
                writer_main.stdin.write(frm_normal.tobytes())
                writer_dev.stdin.write(frame_dev.tobytes())
                
            elif merge_output:
                # Resize normal portrait output to fit the 1080 height evenly
                frm_normal_small = cv2.resize(final_frame, (608, 1080))
                
                # Create dark grey large canvas backdrop
                frm_merged = np.full((1220, 2648, 3), 30, dtype=np.uint8)
                
                # Title texts (Legends)
                cv2.putText(frm_merged, "DIRECTOR'S CONSOLE (16:9 RAW)", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(frm_merged, "FINAL OUTPUT (9:16 CROP)", (2000, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                # Bounding Boxes (Drawn slightly outwards to act as neat white borders)
                cv2.rectangle(frm_merged, (38, 98), (40+1920+2, 100+1080+2), (255, 255, 255), 4)
                cv2.rectangle(frm_merged, (1998, 98), (2000+608+2, 100+1080+2), (255, 255, 255), 4)
                
                # Paste the literal video frames onto the exact pixel coordinates
                frm_merged[100:1180, 40:1960] = frame_dev
                frm_merged[100:1180, 2000:2608] = frm_normal_small
                
                writer_main.stdin.write(frm_merged.tobytes())
                
            else:
                # Single Stream Handling
                output_frm = frame_dev if dev_visualize else final_frame
                if output_frm.shape[0] != out_h_final or output_frm.shape[1] != out_w_final:
                    output_frm = cv2.resize(output_frm, (out_w_final, out_h_final))
                writer_main.stdin.write(output_frm.tobytes())

            frame_count += 1

            render_percent = (
                min(100, int((t / duration) * 100)) if duration > 0 else 100
            )
            if render_percent != last_render_percent:
                print(
                    f"⏳ {label} - Render split-screen: {render_percent:3d}% | "
                    f"{format_seconds(t)} / {format_seconds(duration)}",
                    flush=True,
                )
                last_render_percent = render_percent

        if writer_main:
            writer_main.stdin.close()
            stderr_data = writer_main.stderr.read().decode("utf-8", errors="ignore")
            return_code = writer_main.wait()
            if return_code != 0:
                raise RuntimeError(f"FFmpeg writer main gagal: {stderr_data[-1000:]}")
        
        if writer_dev:
            writer_dev.stdin.close()
            stderr_data_dev = writer_dev.stderr.read().decode("utf-8", errors="ignore")
            return_code_dev = writer_dev.wait()
            if return_code_dev != 0:
                raise RuntimeError(f"FFmpeg writer dev gagal: {stderr_data_dev[-1000:]}")

        print(f"✅ {label} selesai.", flush=True)

    finally:
        cap.release()

    def get_x_final(t):
        if not tracking_log: return default_x_full
        # Interpolate for smoother subtitle tracking
        for i in range(len(tracking_log)-1):
            if tracking_log[i][0] <= t <= tracking_log[i+1][0]:
                t1, t2 = tracking_log[i][0], tracking_log[i+1][0]
                cx1, cx2 = tracking_log[i][1], tracking_log[i+1][1]
                cx = cx1 + (cx2 - cx1) * (t - t1) / (t2 - t1)
                return int(max(0, min(cx - crop_w_full / 2, width - crop_w_full)))
        
        last_cx = tracking_log[-1][1]
        return int(max(0, min(last_cx - crop_w_full / 2, width - crop_w_full)))

    return get_x_final
