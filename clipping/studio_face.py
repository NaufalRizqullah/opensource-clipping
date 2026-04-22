"""
clipping.studio_face — Face Detection (Singleton)

Berisi singleton MediaPipe face detector dan fungsi estimasi jumlah speaker
dari frame video. Digunakan oleh renderer (hybrid, split-screen, camera-switch)
dan oleh runner.py (via studio.py) untuk auto-detect speaker count.

Dipindahkan dari studio.py tanpa perubahan logic.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ==============================================================================
# MEDIAPIPE FACE DETECTOR (SINGLETON)
# ==============================================================================

_FACE_DETECTOR = None


def get_face_detector(cfg):
    """Dapatkan instance singleton MediaPipe FaceDetector.

    Jika model belum ada di disk, akan didownload otomatis dari URL di config.

    Parameters
    ----------
    cfg : SimpleNamespace
        Konfigurasi yang berisi ``file_mediapipe_model`` dan ``url_mediapipe_model``.

    Returns
    -------
    mp_vision.FaceDetector
        Instance face detector yang siap digunakan.
    """
    global _FACE_DETECTOR

    if _FACE_DETECTOR is None:
        if not os.path.exists(cfg.file_mediapipe_model):
            urllib.request.urlretrieve(
                cfg.url_mediapipe_model, cfg.file_mediapipe_model
            )

        base_options = mp_python.BaseOptions(model_asset_path=cfg.file_mediapipe_model)
        _FACE_DETECTOR = mp_vision.FaceDetector.create_from_options(
            mp_vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=0.5,
            )
        )

    return _FACE_DETECTOR


def estimate_speaker_count_from_video(video_path: str, cfg) -> int:
    """
    Sample frames from the video to estimate the max number of visible faces.
    Used for automatically setting min_speakers for pyannote.

    Parameters
    ----------
    video_path : str
        Path ke file video.
    cfg : SimpleNamespace
        Konfigurasi (``face_detector``, ``yolo_size``, dll.).

    Returns
    -------
    int
        Estimasi jumlah speaker (minimal 1).
    """
    import cv2

    print("🔍 Auto-detecting speaker count via visual scanning...", flush=True)

    yolo_model = None
    detector = None

    if cfg.face_detector == "yolo":
        from ultralytics import YOLO
        import logging

        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        try:
            model_name = f"yolov{cfg.yolo_size}-face.pt"
            yolo_model = YOLO(model_name)
        except Exception as e:
            print(f"⚠️ YOLO face detect gagal: {e}. Fallback ke Mediapipe.")
            cfg.face_detector = "mediapipe"

    if cfg.face_detector != "yolo":
        detector = get_face_detector(cfg)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps if fps > 0 else 0

    if duration == 0:
        cap.release()
        return 2

    sample_count = 20
    step = duration / sample_count
    max_faces = 0

    for i in range(sample_count):
        t = i * step
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        faces_in_frame = 0

        if cfg.face_detector == "yolo" and yolo_model:
            results = yolo_model(frame, verbose=False)
            if results and len(results[0].boxes) > 0:
                faces_in_frame = len(results[0].boxes)
        else:
            mp_image = mp_python.Image(
                image_format=mp_python.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            results = detector.detect(mp_image)
            if results.detections:
                faces_in_frame = len(results.detections)

        if faces_in_frame > max_faces:
            max_faces = faces_in_frame

    cap.release()
    print(f"   ✅ Ditemukan maksimum {max_faces} wajah dalam satu frame.", flush=True)
    return max(1, max_faces)
