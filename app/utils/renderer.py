"""Frame rendering utilities.

Three rendering modes:
  skeleton  — coloured skeleton on black background with glow/bloom
  overlay   — skeleton on animated dark studio background
  heatmap   — joint confidence heatmap

All functions accept either:
  - (52, 2) normalised [0,1] keypoints  (procedural avatar)
  - (18, 3) + (21, 3) + (21, 3) COCO keypoints from OpenPose JSON
"""
from __future__ import annotations
import math
import cv2
import numpy as np

from app.utils.rig import (
    BODY_EDGES_COCO, HAND_EDGES,
    LEFT_HAND_COLOUR, RIGHT_HAND_COLOUR, KP_COLOUR, KP_RADIUS,
    LIMBS_52, N_KP,
)


# ── Glow / bloom ──────────────────────────────────────────────────────────────

def apply_glow(frame: np.ndarray, strength: float = 0.55,
               blur_k: int = 21) -> np.ndarray:
    bright  = np.clip(frame.astype(np.float32) * 1.4, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(bright, (blur_k, blur_k), 0)
    return np.clip(cv2.addWeighted(frame, 1.0, blurred, strength, 0), 0, 255).astype(np.uint8)


# ── Studio background ─────────────────────────────────────────────────────────

def _studio_bg(h: int, w: int, t: float = 0.0) -> np.ndarray:
    bg = np.zeros((h, w, 3), dtype=np.float32)
    for row in range(h):
        v = 18 + 12 * (1 - row / h)
        bg[row] = [v * 0.8, v * 0.9, v]
    pulse = 0.5 + 0.5 * math.sin(2 * math.pi * t)
    bg[:, :, 0] += pulse * 4
    bg[:, :, 2] += (1 - pulse) * 4
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X - w / 2) / (w / 2)) ** 2 + ((Y - h / 2) / (h / 2)) ** 2)
    vignette = np.clip(1 - dist * 0.6, 0, 1)[..., np.newaxis]
    return np.clip(bg * vignette, 0, 255).astype(np.uint8)


# ── Drawing primitives ────────────────────────────────────────────────────────

def _draw_limbs_and_joints(
    canvas: np.ndarray,
    kp_px: np.ndarray,                          # (N, 2) int pixel coords
    edges: list[tuple],                          # (a, b, colour) or (a, b)
    default_colour: tuple = (0, 200, 0),
    glow: bool = True,
    line_thickness: int = 2,
) -> None:
    h, w = canvas.shape[:2]
    for edge in edges:
        a, b = edge[0], edge[1]
        colour = edge[2] if len(edge) > 2 else default_colour
        if a >= len(kp_px) or b >= len(kp_px):
            continue
        pt_a = tuple(kp_px[a].tolist())
        pt_b = tuple(kp_px[b].tolist())
        if not (0 <= pt_a[0] < w and 0 <= pt_a[1] < h and
                0 <= pt_b[0] < w and 0 <= pt_b[1] < h):
            continue
        if glow:
            cv2.line(canvas, pt_a, pt_b,
                     tuple(int(c * 0.3) for c in colour),
                     line_thickness + 4, cv2.LINE_AA)
        cv2.line(canvas, pt_a, pt_b, colour, line_thickness, cv2.LINE_AA)

    for pt in kp_px:
        x, y = int(pt[0]), int(pt[1])
        if not (0 <= x < w and 0 <= y < h):
            continue
        if glow:
            cv2.circle(canvas, (x, y), KP_RADIUS + 3,
                       tuple(int(c * 0.3) for c in KP_COLOUR), -1)
        cv2.circle(canvas, (x, y), KP_RADIUS, KP_COLOUR, -1, cv2.LINE_AA)


# ── Render from 52-keypoint normalised array (procedural avatar) ──────────────

def render_52kp(
    kp_norm: np.ndarray,          # (52, 2) in [0, 1]
    width: int = 640,
    height: int = 480,
    mode: str = "skeleton",       # "skeleton" | "overlay" | "heatmap"
    frame_idx: int = 0,
    n_frames: int = 1,
    glow: bool = True,
) -> np.ndarray:
    t = frame_idx / max(n_frames - 1, 1)
    if mode == "overlay":
        canvas = _studio_bg(height, width, t)
    else:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

    kp_px = np.stack([
        np.clip(kp_norm[:, 0] * width,  0, width  - 1),
        np.clip(kp_norm[:, 1] * height, 0, height - 1),
    ], axis=1).astype(np.int32)

    if mode == "heatmap":
        heat = np.zeros((height, width), dtype=np.float32)
        for x, y in kp_px:
            cv2.circle(heat, (int(x), int(y)), 20, 1.0, -1)
        heat = cv2.GaussianBlur(heat, (31, 31), 0)
        heat_u8 = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        canvas = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    else:
        _draw_limbs_and_joints(canvas, kp_px, LIMBS_52, glow=glow)
        if glow:
            canvas = apply_glow(canvas, strength=0.5)

    return canvas


# ── Render from OpenPose JSON keypoints ───────────────────────────────────────

def _flat_to_px(flat: list[float], n: int,
                W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Parse flat [x,y,c, x,y,c, ...] into pixel coords and confidence mask."""
    arr  = np.array(flat, dtype=np.float32).reshape(n, 3)
    conf = arr[:, 2]
    px   = np.stack([
        np.clip(arr[:, 0], 0, W - 1),
        np.clip(arr[:, 1], 0, H - 1),
    ], axis=1).astype(np.int32)
    return px, conf


def render_openpose_frame(
    frame_bgr: np.ndarray,
    body_flat: list[float],
    lhand_flat: list[float],
    rhand_flat: list[float],
    mode: str = "overlay",        # "overlay" | "skeleton" | "heatmap"
    conf_thresh: float = 0.05,
) -> np.ndarray:
    """Render one OpenPose JSON frame onto a copy of frame_bgr."""
    H, W = frame_bgr.shape[:2]

    if mode == "skeleton":
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
    elif mode == "heatmap":
        canvas = frame_bgr.copy()
    else:
        canvas = frame_bgr.copy()

    # Body
    body_px, body_conf = _flat_to_px(body_flat, 18, W, H)
    valid_body = body_px[body_conf > conf_thresh]
    for edge in BODY_EDGES_COCO:
        a, b, colour = edge
        if body_conf[a] < conf_thresh or body_conf[b] < conf_thresh:
            continue
        pt_a = tuple(body_px[a].tolist())
        pt_b = tuple(body_px[b].tolist())
        cv2.line(canvas, pt_a, pt_b, colour, 2, cv2.LINE_AA)
    for pt, c in zip(body_px, body_conf):
        if c < conf_thresh:
            continue
        cv2.circle(canvas, tuple(pt.tolist()), KP_RADIUS, KP_COLOUR, -1, cv2.LINE_AA)

    # Hands
    for flat, colour in [(lhand_flat, LEFT_HAND_COLOUR),
                         (rhand_flat, RIGHT_HAND_COLOUR)]:
        if not flat or all(v == 0 for v in flat):
            continue
        hpx, hconf = _flat_to_px(flat, 21, W, H)
        for a, b in HAND_EDGES:
            if hconf[a] < conf_thresh or hconf[b] < conf_thresh:
                continue
            cv2.line(canvas, tuple(hpx[a].tolist()), tuple(hpx[b].tolist()),
                     colour, 1, cv2.LINE_AA)
        for pt, c in zip(hpx, hconf):
            if c < conf_thresh:
                continue
            cv2.circle(canvas, tuple(pt.tolist()), 2, colour, -1, cv2.LINE_AA)

    if mode == "heatmap":
        heat = np.zeros((H, W), dtype=np.float32)
        for pt, c in zip(body_px, body_conf):
            if c > conf_thresh:
                cv2.circle(heat, tuple(pt.tolist()), 25, float(c), -1)
        heat = cv2.GaussianBlur(heat, (31, 31), 0)
        heat_u8 = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        canvas = cv2.addWeighted(frame_bgr, 0.5, heatmap, 0.5, 0)

    return canvas


# ── FPS overlay ───────────────────────────────────────────────────────────────

def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def draw_label(frame: np.ndarray, text: str,
               colour: tuple = (255, 255, 255)) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, text, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)
    return out
