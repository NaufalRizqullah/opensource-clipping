"""Face detector lifecycle contracts without native libraries or downloads."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def face_module(monkeypatch):
    def module(name, **attributes):
        result = ModuleType(name)
        result.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    class Image:
        def __init__(self, *, image_format, data):
            self.image_format = image_format
            self.data = data

    vision = module(
        "mediapipe.tasks.python.vision",
        FaceDetector=SimpleNamespace(),
        FaceDetectorOptions=SimpleNamespace,
        RunningMode=SimpleNamespace(IMAGE="IMAGE"),
    )
    tasks_python = module("mediapipe.tasks.python", BaseOptions=SimpleNamespace, vision=vision)
    tasks = module("mediapipe.tasks", python=tasks_python)
    module("mediapipe", tasks=tasks, Image=Image, ImageFormat=SimpleNamespace(SRGB="SRGB"))
    module("numpy", zeros=lambda shape, dtype: SimpleNamespace(shape=shape, dtype=dtype), uint8="uint8")
    module("cv2")
    module("requests")
    module("PIL", Image=object(), ImageDraw=object(), ImageFont=object())
    module("yt_dlp", YoutubeDL=object)

    spec = importlib.util.spec_from_file_location(
        "_test_face_detection", ROOT / "clipping" / "studio" / "face_detection.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    monkeypatch.setattr(
        result.urllib.request, "urlretrieve",
        lambda *args: pytest.fail("Unexpected face model download"),
    )
    return result


@pytest.fixture
def cfg(tmp_path):
    model = tmp_path / "test-face-model.tflite"
    model.write_bytes(b"synthetic model")
    return SimpleNamespace(
        file_mediapipe_model=str(model),
        url_mediapipe_model="https://example.test/model.tflite",
        face_detector="mediapipe",
    )


def test_detector_is_cached_only_after_inference_and_reused(face_module, cfg):
    constructed = []
    images = []

    def detect(image):
        assert face_module._FACE_DETECTOR is None
        images.append(image)
        return SimpleNamespace(detections=[])

    detector = SimpleNamespace(detect=detect)

    def create(options):
        constructed.append(options)
        return detector

    face_module.mp_vision.FaceDetector.create_from_options = create
    assert face_module.get_face_detector(cfg) is detector
    assert face_module.get_face_detector(cfg) is detector
    assert len(constructed) == len(images) == 1
    assert constructed[0].running_mode == "IMAGE"
    assert constructed[0].base_options.model_asset_path == cfg.file_mediapipe_model
    assert images[0].image_format == "SRGB"
    assert images[0].data.shape == (128, 128, 3)
    assert images[0].data.dtype == "uint8"


@pytest.mark.parametrize("close_fails", [False, True])
def test_failed_inference_is_not_cached_and_cleanup_preserves_cause(face_module, cfg, close_fails):
    original = RuntimeError("RET_CHECK failure: raw_box_tensor num_boxes_")
    closed = []

    def detect(image):
        raise original

    def close():
        closed.append(True)
        if close_fails:
            raise RuntimeError("Graph has errors during cleanup")

    bad = SimpleNamespace(detect=detect, close=close)
    good = SimpleNamespace(detect=lambda image: SimpleNamespace(detections=[]))
    detectors = iter([bad, good])
    face_module.mp_vision.FaceDetector.create_from_options = lambda options: next(detectors)

    with pytest.raises(RuntimeError, match="failed its inference check") as captured:
        face_module.get_face_detector(cfg)
    assert captured.value.__cause__ is original
    assert face_module._FACE_DETECTOR is None
    assert closed == [True]
    assert face_module.get_face_detector(cfg) is good


def test_missing_model_downloads_to_configured_path_before_inference(face_module, cfg, tmp_path, monkeypatch):
    cfg.file_mediapipe_model = str(tmp_path / "new-model.tflite")
    downloads = []

    def download(url, filename):
        downloads.append((url, filename))
        Path(filename).write_bytes(b"downloaded synthetic model")

    def create(options):
        assert Path(options.base_options.model_asset_path).read_bytes() == b"downloaded synthetic model"
        return SimpleNamespace(detect=lambda image: SimpleNamespace(detections=[]))

    monkeypatch.setattr(face_module.urllib.request, "urlretrieve", download)
    face_module.mp_vision.FaceDetector.create_from_options = create
    face_module.get_face_detector(cfg)
    assert downloads == [(cfg.url_mediapipe_model, cfg.file_mediapipe_model)]


def test_auto_speaker_scan_uses_mediapipe_image_namespace(face_module, cfg):
    frames = []
    released = []

    class Capture:
        def isOpened(self):
            return True

        def get(self, prop):
            return 30 if prop == "fps" else 600

        def set(self, prop, value):
            frames.append(value)

        def read(self):
            return True, "synthetic BGR frame"

        def release(self):
            released.append(True)

    face_module.cv2.VideoCapture = lambda path: Capture()
    face_module.cv2.CAP_PROP_FPS = "fps"
    face_module.cv2.CAP_PROP_FRAME_COUNT = "frames"
    face_module.cv2.CAP_PROP_POS_MSEC = "position"
    face_module.cv2.COLOR_BGR2RGB = "BGR2RGB"
    face_module.cv2.cvtColor = lambda frame, code: "synthetic RGB frame"
    images = []

    def detect(image):
        images.append(image)
        return SimpleNamespace(detections=[object()] * (len(images) % 4))

    face_module._FACE_DETECTOR = SimpleNamespace(detect=detect)
    assert face_module.estimate_speaker_count_from_video("synthetic-video.mp4", cfg) == 3
    assert len(frames) == len(images) == 20
    assert all(image.image_format == "SRGB" and image.data == "synthetic RGB frame" for image in images)
    assert released == [True]
