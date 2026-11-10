# 🎯 Face Tracking & Auto-Framing

OpenSource Clipping uses AI-powered face detection to automatically keep the subject centered in the frame when cropping from 16:9 to vertical (9:16, 1:1, 3:4, 4:5) formats.

---

## How It Works

1. **Detection**: AI detects faces in the source video at regular intervals (default: every 0.25 seconds)
2. **Tracking**: The camera position smoothly follows the detected face using a deadzone + smoothing algorithm
3. **Anti-Jitter**: Micro-shakes below the jitter threshold are ignored for a steady shot
4. **Scene Cut Detection**: When a camera cut is detected, the tracking history resets instantly to prevent lag

```
Source Frame (16:9)
┌─────────────────────────────────┐
│                                 │
│         ┌─────────┐             │
│         │  9:16   │             │
│         │  crop   │  ← Follows  │
│         │ window  │    the face │
│         │         │             │
│         └─────────┘             │
│                                 │
└─────────────────────────────────┘
```

---

## Face Detection Models

### MediaPipe BlazeFace (Default)

- **Flag**: `--face-detector mediapipe`
- **Device**: CPU only
- **Speed**: Fast and lightweight
- **Best for**: Standard single-speaker content
- **Model**: [BlazeFace Short-Range](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector), float16 version 1, cached as `blaze_face_short_range_v1.tflite`

The Tasks FaceDetector used by the constrained MediaPipe 0.10.21 runtime expects the short-range model's 896 boxes. The former full-range model can initialize successfully but fails on its first image with a `raw_box_tensor` / `num_boxes_` assertion. Both CLI and Studio use the compatible model and a distinct cache filename, leaving any old full-range file untouched. The detector runs a synthetic image inference check before it is cached for reuse.

Short-range detection may be less effective for small or distant faces than the previously intended full-range behavior. For wide shots and multi-speaker content, the existing YOLO option below remains available. See [the version-specific Tasks graph](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/tasks/cc/vision/face_detector/face_detector_graph.cc#L71-L107) for its fixed model parameters.

### YOLOv8 ADetailer

- **Flag**: `--face-detector yolo`
- **Device**: GPU (CUDA) if available
- **Speed**: Slightly slower but more accurate
- **Best for**: Podcasts, multi-speaker, and profile-heavy scenarios
- **Sizes**: `8n` (nano), `8s` (small), `8m` (medium, default), `8n_v2`, `9c`

```bash
# Use YOLO with medium model
python main.py --url "VIDEO_URL" --face-detector yolo --yolo-size 8m
```

---

## Tracking Parameters

The tracking algorithm uses five core parameters that can be tuned via CLI flags:

| Parameter | Flag | Default | Description |
|---|---|---|---|
| **Detection Frequency** | `--track-step` | `0.25` | How often faces are checked (seconds). Lower = more responsive but heavier |
| **Deadzone** | `--track-deadzone` | `0.15` | Ratio of the "safe zone" where the camera doesn't move. Lower = tighter framing |
| **Smoothing** | `--track-smooth` | `0.30` | Camera catch-up speed. Higher = faster following |
| **Jitter Threshold** | `--track-jitter` | `5` | Pixel threshold to ignore micro-shakes |
| **Snap Threshold** | `--track-snap` | `0.25` | Jump threshold to trigger hard cut between speakers |

### Tuning for Different Scenarios

**Solo Speaker (Talking Head)**
```bash
# Tighter framing, very responsive
--track-deadzone 0.10 --track-smooth 0.35
```

**Interview / 2 Speakers**
```bash
# Wider deadzone to prevent constant panning
--track-deadzone 0.20 --track-smooth 0.25 --track-snap 0.30
```

**High-Energy / Active Movement**
```bash
# Very responsive tracking
--track-step 0.15 --track-deadzone 0.08 --track-smooth 0.40
```

---

## Advanced / Experimental Parameters

| Parameter | Flag | Default | Description |
|---|---|---|---|
| **Confidence Threshold** | `--track-conf` | `0.55` | Raise to prevent ghost detections, lower if faces disappear |
| **Smooth Window** | `--track-smooth-window` | `12` | Frame window for layout stability (12 ≈ 0.5s at 24fps) |
| **Scene Cut Threshold** | `--scene-cut-threshold` | `18` | Sensitivity for camera cut detection (resets history). Range: 15-20 (dark/studio), 30-45 (bright) |
| **IOU Threshold** | `--track-iou-threshold` | `0.2` | Overlap threshold for merging duplicate face detections. Range: 0.1-0.5 |

---

## Aspect Ratios & Face Tracking

| Ratio | Resolution | Face Tracking | Best For |
|---|---|---|---|
| `9:16` | 1080×1920 | ✅ Always active | TikTok, Reels, YouTube Shorts |
| `16:9` | 1920×1080 | ❌ No (letterbox if source differs) | YouTube, Landscape |
| `1:1` | 1080×1080 | ✅ Active (disable with `--static-crop`) | Instagram Feed, Twitter/X |
| `3:4` | 1080×1440 | ✅ Active (disable with `--static-crop`) | Instagram Portrait, Pinterest |
| `4:5` | 1080×1350 | ✅ Active (disable with `--static-crop`) | Instagram/Facebook Feed |

> **Note:** When using `16:9` output with a non-16:9 source, the system applies **letterboxing** (black bars) to preserve proportions.

---

## Static Crop Mode

If you don't need face tracking for square/portrait ratios, use `--static-crop`:

```bash
# Fast center crop without AI detection
python main.py --url "VIDEO_URL" --ratio "1:1" --static-crop
```

This dramatically speeds up rendering by bypassing the face detection step entirely.

---

## Developer Visualization

### Dev Mode
See the tracking algorithm in action with a 16:9 "Director's Console" view:

```bash
# Visualize tracking (generates dev video only)
python main.py --url "VIDEO_URL" --dev-mode

# Generate BOTH final output + dev visualization
python main.py --url "VIDEO_URL" --dev-mode-with-output

# Merged side-by-side ultrawide view
python main.py --url "VIDEO_URL" --dev-mode-with-output-merge
```

### Debug Overlays
```bash
# Draw yellow bounding boxes around detected faces
python main.py --url "VIDEO_URL" --box-face-detection

# Draw crosshair tracking lines from face to crop boundaries
python main.py --url "VIDEO_URL" --track-lines
```

---

## See Also

- [Podcast Modes](Podcast-Modes) — Face tracking in split-screen and camera-switch
- [Video Quality & Rendering](Video-Quality-and-Rendering) — High-resolution rendering options
