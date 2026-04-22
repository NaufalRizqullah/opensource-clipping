"""
clipping.studio.split — Split-Screen Renderer (Podcast 2 Speaker)

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

from .utils import (
    detect_video_encoder,
    format_seconds,
    open_ffmpeg_video_writer,
)
from .face import get_face_detector


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
    from ..diarization import get_active_speaker, get_active_speakers

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
    panel_h = (1920 - DIVIDER_HEIGHT) // 2
    panel_w = 1080

    # --- Panel Crop dimensions ---
    panel_ratio = panel_w / panel_h
    if (width / height) > panel_ratio:
        crop_h = height
        crop_w = int(height * panel_ratio)
    else:
        crop_w = width
        crop_h = int(width / panel_ratio)

    # --- Full 9:16 Crop dimensions ---
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

    if diarization_data:
        all_speakers_in_clip = sorted(set(s["speaker"] for s in diarization_data))
        speaking_time = {spk: 0.0 for spk in all_speakers_in_clip}
        for seg in diarization_data:
            eff_start = max(seg["start"], start_clip)
            eff_end = min(seg["end"], end_clip)
            if eff_end > eff_start:
                speaking_time[seg["speaker"]] += eff_end - eff_start
        ranked = sorted(all_speakers_in_clip, key=lambda s: speaking_time[s], reverse=True)
        speaker_top = ranked[0]
        speaker_bottom = ranked[1] if len(ranked) > 1 else ranked[0]
    else:
        all_speakers_in_clip = ["FACE_L", "FACE_R"]
        speaker_top = "FACE_L"
        speaker_bottom = "FACE_R"
        ranked = all_speakers_in_clip

    print(f"🧠 {label} - Analisa wajah (split-screen) dimulai...", flush=True)

    all_frame_data = [] 
    speaker_solo_cxs = {} 
    solo_counts = {spk: 0 for spk in all_speakers_in_clip}
    multi_counts = {spk: 0 for spk in all_speakers_in_clip}
    current_time = 0.0
    last_detect_percent = -1

    while current_time <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, (start_clip + current_time) * 1000)
        ret, frame = cap.read()
        if not ret: break

        face_centers = []
        face_boxes = []

        if cfg.face_detector == "yolo":
            det_conf = getattr(cfg, "track_conf", 0.55)
            yolo_results = yolo_model(frame, verbose=False, conf=det_conf)
            if yolo_results and len(yolo_results[0].boxes) > 0:
                raw_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                final_boxes = []
                for rb in raw_boxes:
                    merged = False
                    for i, fb in enumerate(final_boxes):
                        xi1, yi1 = max(rb[0], fb[0]), max(rb[1], fb[1])
                        xi2, yi2 = min(rb[2], fb[2]), min(rb[3], fb[3])
                        inter = max(0, xi2-xi1)*max(0, yi2-yi1)
                        if inter / ((rb[2]-rb[0])*(rb[3]-rb[1]) + (fb[2]-fb[0])*(fb[3]-fb[1]) - inter + 1e-6) > 0.2:
                            merged = True; break
                    if not merged: final_boxes.append(rb)
                for box in final_boxes:
                    face_centers.append(((box[0]+box[2])/2, (box[1]+box[3])/2))
                    face_boxes.append(box)
        else:
            results = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if results.detections:
                for d in results.detections:
                    bb = d.bounding_box
                    face_centers.append((bb.origin_x + bb.width/2, bb.origin_y + bb.height/2))
                    face_boxes.append((bb.origin_x, bb.origin_y, bb.origin_x+bb.width, bb.origin_y+bb.height))

        face_centers.sort(key=lambda fc: fc[0])
        from ..diarization import get_active_speakers
        active_now = get_active_speakers(diarization_data, start_clip + current_time) if diarization_data else (["FACE_L"] if len(face_centers)==1 else ["FACE_L","FACE_R"] if len(face_centers)>=2 else [])

        if len(face_centers) == 1 and len(active_now) == 1:
            speaker_solo_cxs.setdefault(active_now[0], []).append(face_centers[0][0])
            solo_counts[active_now[0]] += 1
        elif len(face_centers) >= 2 and len(active_now) >= 1:
            for spk in active_now: multi_counts[spk] += 1

        all_frame_data.append({"time": current_time, "face_centers": face_centers, "face_boxes": face_boxes, "active_now": active_now})
        current_time += STEP_DETEKSI

    import statistics as _stats
    speaker_canonical_cx = {spk: _stats.median(cxs) for spk, cxs in speaker_solo_cxs.items() if cxs}
    if not diarization_data:
        speaker_canonical_cx["FACE_L"] = speaker_canonical_cx.get("FACE_L", width * 0.25)
        speaker_canonical_cx["FACE_R"] = speaker_canonical_cx.get("FACE_R", width * 0.75)
    speaker_is_solo = {spk: (solo_counts.get(spk, 0) > multi_counts.get(spk, 0)) for spk in all_speakers_in_clip}

    raw_data = {spk: [] for spk in all_speakers_in_clip}
    for fd in all_frame_data:
        fc_list, act_list = fd["face_centers"], fd["active_now"]
        nf, na = len(fc_list), len(act_list)
        if nf == 0 or na == 0: continue
        if nf == 1 and na == 1:
            spk = act_list[0]
            if spk in raw_data: raw_data[spk].append({"time": fd["time"], "cx": fc_list[0][0]})
        elif nf >= 2 and na >= 2:
            remaining = list(fc_list)
            for spk in sorted(act_list, key=lambda s: speaker_canonical_cx.get(s, width/2)):
                if not remaining or spk not in raw_data: break
                best = min(remaining, key=lambda fc: abs(fc[0] - speaker_canonical_cx.get(s, width/2)))
                remaining.remove(best); raw_data[spk].append({"time": fd["time"], "cx": best[0]})
        elif nf >= 2 and na == 1:
            spk = act_list[0]
            if spk not in raw_data: continue
            remaining = list(fc_list)
            for other in all_speakers_in_clip:
                if other != spk and other in speaker_canonical_cx and remaining:
                    claimed = min(remaining, key=lambda fc: abs(fc[0] - speaker_canonical_cx[other]))
                    remaining.remove(claimed)
            pool = remaining if remaining else fc_list
            best = min(pool, key=lambda fc: abs(fc[0] - raw_data[spk][-1]["cx"])) if raw_data[spk] else min(pool, key=lambda fc: abs(fc[0] - speaker_canonical_cx.get(spk, width/2)))
            raw_data[spk].append({"time": fd["time"], "cx": best[0]})

    def _smooth_p(raw):
        res = []
        if not raw: return res
        cam_cx = raw[0]["cx"]
        for d in raw:
            f_cx = d["cx"]
            if abs(f_cx - cam_cx) > width*SNAP_THRESHOLD: cam_cx = f_cx
            else:
                dz = crop_w_full * DEADZONE_RATIO
                if f_cx > cam_cx + dz: cam_cx += (f_cx - (cam_cx + dz)) * SMOOTH_FACTOR
                elif f_cx < cam_cx - dz: cam_cx += (f_cx - (cam_cx - dz)) * SMOOTH_FACTOR
            res.append({"time": d["time"], "cx": cam_cx})
        return res

    smooth = {spk: _smooth_p(raw_data[spk]) for spk in all_speakers_in_clip}

    def _get_cx(spk, t):
        sd = smooth.get(spk, [])
        if not sd: return width/2
        if t <= sd[0]["time"]: return sd[0]["cx"]
        if t >= sd[-1]["time"]: return sd[-1]["cx"]
        for i in range(len(sd)-1):
            if sd[i]["time"] <= t <= sd[i+1]["time"]:
                return sd[i]["cx"] + (sd[i+1]["cx"]-sd[i]["cx"]) * (t-sd[i]["time"])/(sd[i+1]["time"]-sd[i]["time"])
        return width/2

    def _get_all_boxes(t):
        if not all_frame_data: return []
        if t <= all_frame_data[0]["time"]: return all_frame_data[0]["face_boxes"]
        if t >= all_frame_data[-1]["time"]: return all_frame_data[-1]["face_boxes"]
        for i in range(len(all_frame_data)-1):
            if all_frame_data[i]["time"] <= t <= all_frame_data[i+1]["time"]:
                if len(all_frame_data[i]["face_boxes"]) != len(all_frame_data[i+1]["face_boxes"]):
                    return all_frame_data[i]["face_boxes"] if abs(t-all_frame_data[i]["time"]) < abs(t-all_frame_data[i+1]["time"]) else all_frame_data[i+1]["face_boxes"]
                f = (t-all_frame_data[i]["time"])/(all_frame_data[i+1]["time"]-all_frame_data[i]["time"])
                return [(b1[j] + (b2[j]-b1[j])*f) for j in range(4)] for b1, b2 in zip(all_frame_data[i]["face_boxes"], all_frame_data[i+1]["face_boxes"])
        return []

    writer_main = open_ffmpeg_video_writer(output_video if not dual_output else output_video, 1080 if not (merge_output or (dev_visualize and not dual_output)) else out_w_final, 1920 if not (merge_output or (dev_visualize and not dual_output)) else out_h_final, orig_fps, video_encoder)
    writer_dev = open_ffmpeg_video_writer(output_video.replace(".ts","_dev.ts").replace(".mp4","_dev.mp4"), 1920, 1080, orig_fps, video_encoder) if dual_output else None

    dark_overlay = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    last_valid_crop = {}
    current_layout, current_speaker, last_switch_time = "split", None, -getattr(cfg, "switch_hold_duration", 2.0)
    face_count_history, prev_small_gray = [], None

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_clip * 1000)
        frame_count, tracking_log = 0, []
        while True:
            ret, frame = cap.read()
            if not ret: break
            t = frame_count / orig_fps
            if t > duration: break
            
            curr_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
            if prev_small_gray is not None and np.mean(cv2.absdiff(curr_small, prev_small_gray)) > getattr(cfg, "scene_cut_threshold", 18):
                face_count_history.clear(); last_switch_time = t - getattr(cfg, "switch_hold_duration", 2.0)
            prev_small_gray = curr_small

            from ..diarization import get_active_speaker, get_active_speakers
            active_speakers = get_active_speakers(diarization_data, start_clip + t) if diarization_data else (["FACE_L"] if len(_get_all_boxes(t))==1 else ["FACE_L","FACE_R"] if len(_get_all_boxes(t))>=2 else [])
            active_speaker = get_active_speaker(diarization_data, start_clip + t) if diarization_data else (active_speakers[0] if active_speakers else None)

            if getattr(cfg, "use_dynamic_split", False):
                now_boxes = _get_all_boxes(t)
                face_count_history.append(len(now_boxes))
                if len(face_count_history) > getattr(cfg, "track_smooth_window", 12): face_count_history.pop(0)
                stable_count = max(set(face_count_history), key=face_count_history.count) if face_count_history else len(now_boxes)

                if current_layout == "split" and len(now_boxes) == 1:
                    current_layout, current_speaker, last_switch_time = "full", ranked[0], t; face_count_history.clear()
                elif (t - last_switch_time) >= getattr(cfg, "switch_hold_duration", 2.0):
                    target_layout = "full" if stable_count == 1 else "split" if stable_count >= 2 else current_layout
                    if target_layout != current_layout: current_layout, last_switch_time = target_layout, t
            
            if current_layout == "full":
                spk = current_speaker or ranked[0]
                cx = _get_cx(spk, t)
                x_f = int(max(0, min(cx - crop_w_full/2, width - crop_w_full)))
                final_frame = cv2.resize(frame[0:crop_h_full, x_f : x_f + crop_w_full], (1080, 1920))
                tracking_log.append((t, cx))
            else:
                p_top_spk = active_speaker if active_speaker and active_speaker not in (speaker_top, speaker_bottom) else speaker_top
                p_bot_spk = speaker_bottom
                def _b_p(spk):
                    smooth_cx = _get_cx(spk, t)
                    x_p = int(max(0, min(smooth_cx - crop_w/2, width - crop_w)))
                    crop = cv2.resize(frame[0:crop_h, x_p : x_p + crop_w], (panel_w, panel_h))
                    last_valid_crop[spk] = crop.copy()
                    return crop
                p_top, p_bot = _b_p(p_top_spk), _b_p(p_bot_spk)
                if active_speaker == p_top_spk:
                    p_bot = cv2.addWeighted(p_bot, 1.0-0.15, dark_overlay, 0.15, 0)
                    cv2.rectangle(p_top, (0,0), (panel_w-1, panel_h-1), (0,255,255), 3)
                elif active_speaker == p_bot_spk:
                    p_top = cv2.addWeighted(p_top, 1.0-0.15, dark_overlay, 0.15, 0)
                    cv2.rectangle(p_bot, (0,0), (panel_w-1, panel_h-1), (0,255,255), 3)
                final_frame = np.vstack([p_top, np.full((4, panel_w, 3), 80, dtype=np.uint8), p_bot])
                tracking_log.append((t, width/2))

            if writer_main: writer_main.stdin.write(cv2.resize(final_frame, (out_w_final, out_h_final)).tobytes())
            frame_count += 1
        if writer_main: writer_main.stdin.close(); writer_main.wait()
    finally: cap.release()

    def get_x_final(t):
        for i in range(len(tracking_log)-1):
            if tracking_log[i][0] <= t <= tracking_log[i+1][0]:
                cx = tracking_log[i][1] + (tracking_log[i+1][1]-tracking_log[i][1]) * (t-tracking_log[i][0])/(tracking_log[i+1][0]-tracking_log[i][0])
                return int(max(0, min(cx - crop_w_full/2, width - crop_w_full)))
        return int(max(0, min(tracking_log[-1][1] - crop_w_full/2, width - crop_w_full))) if tracking_log else default_x_full
    return get_x_final
