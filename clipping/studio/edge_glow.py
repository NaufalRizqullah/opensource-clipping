"""
clipping.studio.edge_glow — Ambient Gradient Edge Glow Generator

Generates a short looping video of slow-moving gradient light along
the edges of the frame.  Used as an overlay during voice-over intros
to give a premium, Gemini-like ambient lighting feel.

Strategy:
  1. Build a static alpha mask (bright at edges, fading inward).
  2. For each frame, compute a slowly-rotating hue gradient that
     flows around the perimeter of the rectangle.
  3. Write out as MP4 with black background — the caller composites
     it using FFmpeg ``blend=screen`` or ``blend=lighten``.
"""

import math
import os
import subprocess
import numpy as np
import cv2


def generate_edge_glow_video(
    output_path: str,
    width: int,
    height: int,
    duration: float = 10.0,
    fps: int = 30,
    edge_thickness: int = 0,
    glow_speed: float = 0.15,
    opacity: float = 0.45,
):
    """
    Generate a looping edge-glow overlay video (black-based, to be blended).

    Parameters
    ----------
    output_path : str
        Where to write the output .mp4 file.
    width, height : int
        Frame dimensions (should match the VO intro canvas).
    duration : float
        Length of the loop in seconds.
    fps : int
        Frame rate.
    edge_thickness : int
        Glow depth from each edge.  0 = auto (10% of min dimension).
    glow_speed : float
        Hue rotation speed (rotations per second).  Lower = calmer.
    opacity : float
        Peak brightness of the glow (0-1).
    """
    if edge_thickness <= 0:
        edge_thickness = max(40, int(min(width, height) * 0.10))

    total_frames = int(duration * fps)

    # ---- 1. Build a static alpha mask (float 0-1) ----
    # Uses vectorised distance-from-edge computation.
    ys = np.arange(height).reshape(-1, 1).astype(np.float32)
    xs = np.arange(width).reshape(1, -1).astype(np.float32)

    d_top = ys                              # distance from top
    d_bottom = (height - 1) - ys            # distance from bottom
    d_left = xs                             # distance from left
    d_right = (width - 1) - xs              # distance from right

    # Minimum distance to any edge
    d_edge = np.minimum(np.minimum(d_top, d_bottom),
                        np.minimum(d_left, d_right))

    # Normalise: 1.0 at the edge, 0.0 at edge_thickness pixels inward
    alpha = np.clip(1.0 - d_edge / edge_thickness, 0, 1)
    # Quadratic ease-out for a softer falloff
    alpha = alpha * alpha
    alpha = (alpha * opacity * 255).astype(np.uint8)

    # ---- 2. Build a perimeter-position map (0→1 around the rectangle) ----
    # Each pixel gets a value indicating its closest point on the perimeter
    # normalised by total perimeter length, so hue can "flow" around the edges.
    perimeter = float(2 * (width + height))

    # For each pixel, compute the perimeter position of its nearest edge point.
    # Top-nearest: perim_pos = x
    # Right-nearest: perim_pos = width + y
    # Bottom-nearest: perim_pos = width + height + (width - x)
    # Left-nearest: perim_pos = 2*width + height + (height - y)

    p_top = xs.copy()                                                   # shape (1, W)
    p_right = np.full_like(ys, width, dtype=np.float32) + ys            # shape (H, 1)
    p_bottom = np.full_like(xs, width + height, dtype=np.float32) + ((width - 1) - xs)
    p_left = np.full_like(ys, 2 * width + height, dtype=np.float32) + ((height - 1) - ys)

    # Pick the perimeter position of the nearest edge
    # d_top, d_bottom are (H,1), d_left, d_right are (1,W)
    # We need to broadcast them to (H,W)
    d_top_b = np.broadcast_to(d_top, (height, width))
    d_bottom_b = np.broadcast_to(d_bottom, (height, width))
    d_left_b = np.broadcast_to(d_left, (height, width))
    d_right_b = np.broadcast_to(d_right, (height, width))

    p_top_b = np.broadcast_to(p_top, (height, width))
    p_right_b = np.broadcast_to(p_right, (height, width))
    p_bottom_b = np.broadcast_to(p_bottom, (height, width))
    p_left_b = np.broadcast_to(p_left, (height, width))

    # Stack and pick by argmin
    dists = np.stack([d_top_b, d_right_b, d_bottom_b, d_left_b], axis=-1)
    positions = np.stack([p_top_b, p_right_b, p_bottom_b, p_left_b], axis=-1)

    nearest_idx = np.argmin(dists, axis=-1)
    pos_map = np.take_along_axis(positions, nearest_idx[..., np.newaxis], axis=-1).squeeze(-1)
    pos_map = pos_map / perimeter  # normalise to 0→1

    # ---- 3. Render frames via piped FFmpeg ----
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-t", str(duration),
        output_path,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Pre-allocate
    hsv_frame = np.zeros((height, width, 3), dtype=np.uint8)
    hsv_frame[:, :, 1] = 200  # saturation (constant)

    for fi in range(total_frames):
        t = fi / fps
        hue_offset = t * glow_speed  # fraction of full rotation

        # Hue = perimeter_position + time_offset, mapped to 0-180 for OpenCV
        hue = ((pos_map + hue_offset) % 1.0 * 180).astype(np.uint8)

        hsv_frame[:, :, 0] = hue
        # Value channel = alpha mask (bright at edges, dark inside)
        hsv_frame[:, :, 2] = alpha

        bgr = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
        proc.stdin.write(bgr.tobytes())

    proc.stdin.close()
    stderr = proc.stderr.read().decode("utf-8", errors="ignore")
    rc = proc.wait()
    if rc != 0:
        print(f"   ⚠️ Edge glow generation warning: {stderr[-500:]}")

    print(f"   ✨ Edge glow video generated: {output_path} ({total_frames} frames)")
    return output_path
