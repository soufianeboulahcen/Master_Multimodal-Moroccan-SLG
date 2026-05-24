"""Render high-quality avatar videos from pose-map PNGs without a GPU.

Composites a stylised human figure (filled body silhouette + skin-toned
limbs + clothing colours + face) on top of the ControlNet pose maps,
producing cinematic-quality avatar videos at 512×768 / 30fps.

This is the CPU-only path — no diffusion models required.
Output is written to outputs/avatar/videos/<motion>_avatar.mp4

Usage:
    python3 scripts/avatar/render_avatar_local.py --motion walking
    python3 scripts/avatar/render_avatar_local.py --motion all
    python3 scripts/avatar/render_avatar_local.py --motion waving --style dark
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "generate_openpose"))

from motions import MOTIONS
from rig import N_KEYPOINTS, LIMBS

# ── Avatar style presets ──────────────────────────────────────────────────

STYLES = {
    "studio": {
        "bg_top":    (18, 14, 12),
        "bg_bottom": (8,  6,  5),
        "skin":      (210, 170, 140),   # warm skin tone
        "shirt":     (60,  80, 120),    # dark blue shirt
        "pants":     (35,  35,  45),    # dark trousers
        "shoes":     (20,  18,  15),    # dark shoes
        "hair":      (30,  20,  10),    # dark hair
        "face_light":(230, 195, 165),
        "glow_col":  (80, 100, 160),
        "floor_col": (25,  22,  20),
    },
    "dark": {
        "bg_top":    (5,   5,  10),
        "bg_bottom": (2,   2,   5),
        "skin":      (190, 150, 120),
        "shirt":     (20,  20,  30),
        "pants":     (15,  15,  20),
        "shoes":     (10,  10,  12),
        "hair":      (15,  10,   5),
        "face_light":(210, 175, 145),
        "glow_col":  (40,  60, 120),
        "floor_col": (12,  10,   8),
    },
    "bright": {
        "bg_top":    (220, 215, 210),
        "bg_bottom": (180, 175, 170),
        "skin":      (220, 185, 155),
        "shirt":     (255, 255, 255),
        "pants":     (50,  60,  80),
        "shoes":     (30,  25,  20),
        "hair":      (40,  25,  10),
        "face_light":(240, 210, 185),
        "glow_col":  (100, 120, 200),
        "floor_col": (160, 155, 150),
    },
}

# Body part assignment: which limb indices belong to which region
# (used to pick fill colour per segment)
SHIRT_LIMBS  = {0,1,2,3,4,5,6,7,8,21}          # torso + head connection
ARM_LIMBS    = {9,10,11,12,13,14}               # arms
PANTS_LIMBS  = {15,16,17,18,19,20,21,22,23,24} # legs
HAND_LIMBS   = set(range(25, len(LIMBS)))       # hands


# ── Background generators ─────────────────────────────────────────────────

def make_background(h: int, w: int, t: float, style: dict) -> np.ndarray:
    """Gradient background with vignette, floor reflection, and colour pulse."""
    bg = np.zeros((h, w, 3), dtype=np.float32)

    # Vertical gradient
    for row in range(h):
        alpha = row / h
        for c in range(3):
            bg[row, :, c] = (
                style["bg_top"][c] * (1 - alpha) +
                style["bg_bottom"][c] * alpha
            )

    # Subtle colour pulse
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * t * 0.4)
    bg[:, :, 0] += pulse * 3
    bg[:, :, 2] += (1 - pulse) * 5

    # Radial vignette
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h * 0.45
    dist = np.sqrt(((X - cx) / (w * 0.6)) ** 2 + ((Y - cy) / (h * 0.6)) ** 2)
    vignette = np.clip(1.0 - dist * 0.55, 0.0, 1.0)[..., np.newaxis]
    bg = bg * vignette

    # Floor line (subtle)
    floor_y = int(h * 0.82)
    floor_alpha = 0.18
    gc = style["floor_col"]
    bg[floor_y:floor_y+2, :] = (
        bg[floor_y:floor_y+2] * (1 - floor_alpha) +
        np.array(gc, dtype=np.float32) * floor_alpha
    )

    # Floor reflection gradient below floor line
    for row in range(floor_y, min(floor_y + 60, h)):
        fade = 1.0 - (row - floor_y) / 60.0
        bg[row] *= (1.0 - fade * 0.3)

    return np.clip(bg, 0, 255).astype(np.uint8)


# ── Avatar body renderer ──────────────────────────────────────────────────

def _px(kp_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    """Normalised [0,1] → pixel int coords."""
    out = kp_norm.copy()
    out[:, 0] = (out[:, 0] * w).astype(np.int32)
    out[:, 1] = (out[:, 1] * h).astype(np.int32)
    return out.astype(np.int32)


def _in_bounds(pt, w, h):
    return 0 <= pt[0] < w and 0 <= pt[1] < h


def draw_avatar(canvas: np.ndarray, kp_norm: np.ndarray,
                style: dict, t: float) -> np.ndarray:
    """Draw a stylised human avatar onto canvas (in-place)."""
    h, w = canvas.shape[:2]
    kp = _px(kp_norm, w, h)
    s = 0.28  # scale factor (matches rig.py default)

    # ── Shadow ────────────────────────────────────────────────────────────
    shadow_y = int(h * 0.82)
    shadow_cx = int(kp[8, 0])  # MidHip x
    shadow_rx = int(w * s * 0.18)
    shadow_ry = int(h * s * 0.04)
    shadow_layer = canvas.copy()
    cv2.ellipse(shadow_layer, (shadow_cx, shadow_y),
                (shadow_rx, shadow_ry), 0, 0, 360, (0, 0, 0), -1)
    cv2.addWeighted(shadow_layer, 0.35, canvas, 0.65, 0, canvas)

    # ── Limb thickness map ────────────────────────────────────────────────
    # Thicker for torso, thinner for hands/fingers
    def limb_thickness(idx: int) -> int:
        if idx in SHIRT_LIMBS:  return max(int(w * s * 0.055), 8)
        if idx in ARM_LIMBS:    return max(int(w * s * 0.042), 6)
        if idx in PANTS_LIMBS:  return max(int(w * s * 0.052), 7)
        return max(int(w * s * 0.018), 3)  # hands

    # ── Colour per limb ───────────────────────────────────────────────────
    def limb_colour(idx: int) -> tuple:
        if idx in SHIRT_LIMBS:  return style["shirt"]
        if idx in ARM_LIMBS:    return style["skin"]
        if idx in PANTS_LIMBS:  return style["pants"]
        return style["skin"]  # hands

    # ── Draw limbs (back-to-front: legs → torso → arms → hands) ──────────
    draw_order = (
        list(PANTS_LIMBS) +
        list(SHIRT_LIMBS) +
        list(ARM_LIMBS) +
        list(HAND_LIMBS)
    )

    for idx in draw_order:
        if idx >= len(LIMBS):
            continue
        a, b, _ = LIMBS[idx]
        if a >= N_KEYPOINTS or b >= N_KEYPOINTS:
            continue
        pa = tuple(kp[a])
        pb = tuple(kp[b])
        if not (_in_bounds(pa, w, h) and _in_bounds(pb, w, h)):
            continue
        thick = limb_thickness(idx)
        col   = limb_colour(idx)

        # Soft edge: draw slightly thicker darker under-layer first
        under = tuple(max(0, c - 40) for c in col)
        cv2.line(canvas, pa, pb, under, thick + 4, cv2.LINE_AA)
        cv2.line(canvas, pa, pb, col,   thick,     cv2.LINE_AA)

    # ── Shoes ─────────────────────────────────────────────────────────────
    for ankle_idx in (11, 14):  # RAnkle, LAnkle
        if ankle_idx >= N_KEYPOINTS:
            continue
        ax, ay = kp[ankle_idx]
        if not _in_bounds((ax, ay), w, h):
            continue
        shoe_w = int(w * s * 0.07)
        shoe_h = int(h * s * 0.025)
        pts = np.array([
            [ax - shoe_w, ay],
            [ax + shoe_w, ay],
            [ax + shoe_w + 4, ay + shoe_h],
            [ax - shoe_w - 2, ay + shoe_h],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], style["shoes"])

    # ── Head ──────────────────────────────────────────────────────────────
    nose_x, nose_y = kp[0]
    neck_x, neck_y = kp[1]
    if _in_bounds((nose_x, nose_y), w, h):
        head_r = int(w * s * 0.11)
        head_cx = nose_x
        head_cy = nose_y - int(head_r * 0.3)

        # Hair (slightly larger circle behind)
        cv2.circle(canvas, (head_cx, head_cy - int(head_r * 0.1)),
                   head_r + 3, style["hair"], -1, cv2.LINE_AA)

        # Face skin
        cv2.circle(canvas, (head_cx, head_cy),
                   head_r, style["skin"], -1, cv2.LINE_AA)

        # Face highlight (subtle)
        hl_x = head_cx - int(head_r * 0.2)
        hl_y = head_cy - int(head_r * 0.2)
        hl_r = int(head_r * 0.55)
        hl_layer = canvas.copy()
        cv2.circle(hl_layer, (hl_x, hl_y), hl_r, style["face_light"], -1)
        cv2.addWeighted(hl_layer, 0.25, canvas, 0.75, 0, canvas)

        # Eyes
        eye_y = head_cy - int(head_r * 0.12)
        eye_off = int(head_r * 0.28)
        eye_r = max(int(head_r * 0.10), 2)
        for ex in (head_cx - eye_off, head_cx + eye_off):
            cv2.circle(canvas, (ex, eye_y), eye_r + 1, (20, 15, 10), -1)
            cv2.circle(canvas, (ex, eye_y), eye_r,     (40, 30, 20), -1)
            # Eye white highlight
            cv2.circle(canvas, (ex - 1, eye_y - 1), max(eye_r - 2, 1),
                       (220, 215, 210), -1)

        # Eyebrows
        brow_y = eye_y - int(head_r * 0.18)
        brow_w = int(head_r * 0.22)
        brow_t = max(int(head_r * 0.06), 2)
        for bx in (head_cx - eye_off, head_cx + eye_off):
            cv2.line(canvas,
                     (bx - brow_w, brow_y + 1),
                     (bx + brow_w, brow_y - 1),
                     style["hair"], brow_t, cv2.LINE_AA)

        # Nose
        nose_tip_y = head_cy + int(head_r * 0.28)
        cv2.circle(canvas, (head_cx, nose_tip_y),
                   max(int(head_r * 0.07), 2), style["skin"], -1)

        # Mouth
        mouth_y = head_cy + int(head_r * 0.48)
        mouth_w = int(head_r * 0.28)
        mouth_col = tuple(max(0, c - 50) for c in style["skin"])
        cv2.ellipse(canvas, (head_cx, mouth_y),
                    (mouth_w, max(int(head_r * 0.08), 2)),
                    0, 0, 180, mouth_col, max(int(head_r * 0.05), 1),
                    cv2.LINE_AA)

        # Neck
        if _in_bounds((neck_x, neck_y), w, h):
            neck_w = int(w * s * 0.04)
            cv2.line(canvas, (head_cx, head_cy + head_r - 2),
                     (neck_x, neck_y), style["skin"],
                     neck_w * 2, cv2.LINE_AA)

    # ── Ambient glow around figure ────────────────────────────────────────
    glow_layer = canvas.copy()
    glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=18)
    gc = style["glow_col"]
    glow_tint = np.zeros_like(canvas, dtype=np.float32)
    glow_tint[:] = gc
    glow_blend = cv2.addWeighted(
        glow_layer.astype(np.float32), 0.18,
        glow_tint, 0.04, 0
    ).astype(np.uint8)
    cv2.addWeighted(glow_blend, 1.0, canvas, 1.0, 0, canvas)

    return canvas


# ── Cinematic post-processing ─────────────────────────────────────────────

def apply_film_grain(frame: np.ndarray, strength: float = 4.0) -> np.ndarray:
    """Add subtle film grain for cinematic feel."""
    noise = np.random.normal(0, strength, frame.shape).astype(np.float32)
    out = frame.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_color_grade(frame: np.ndarray, t: float) -> np.ndarray:
    """Cinematic colour grade: slight warm shadows, cool highlights."""
    f = frame.astype(np.float32)
    # Lift shadows slightly warm
    shadow_mask = np.clip(1.0 - f / 128.0, 0, 1)
    f[:, :, 2] += shadow_mask[:, :, 2] * 4   # red channel
    f[:, :, 0] -= shadow_mask[:, :, 0] * 2   # blue channel
    # Cool highlights
    highlight_mask = np.clip(f / 200.0 - 0.5, 0, 1)
    f[:, :, 0] += highlight_mask[:, :, 0] * 3
    return np.clip(f, 0, 255).astype(np.uint8)


def apply_letterbox(frame: np.ndarray, ratio: float = 0.06) -> np.ndarray:
    h = frame.shape[0]
    bar = int(h * ratio)
    out = frame.copy()
    out[:bar] = 0
    out[-bar:] = 0
    return out


def apply_camera_drift(frame: np.ndarray, t: float,
                       amp_x: float = 4.0, amp_y: float = 2.5) -> np.ndarray:
    dx = int(amp_x * np.sin(2 * np.pi * t * 0.25))
    dy = int(amp_y * np.cos(2 * np.pi * t * 0.18))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))


# ── Video writer ──────────────────────────────────────────────────────────

def render_motion_video(
    motion_name: str,
    out_path: Path,
    width: int = 512,
    height: int = 768,
    fps: int = 30,
    style_name: str = "studio",
    n_frames: int = 0,
) -> None:
    style = STYLES[style_name]
    gen_fn = MOTIONS[motion_name]
    kp_frames = gen_fn(n_frames=n_frames or 150, fps=fps)
    T = len(kp_frames)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {out_path}")

    print(f"[avatar] rendering {motion_name} ({T} frames @ {fps}fps) → {out_path}")
    np.random.seed(42)  # reproducible grain

    for i, kp_norm in enumerate(kp_frames):
        t = i / max(T - 1, 1)

        # Background
        canvas = make_background(height, width, t, style)

        # Avatar figure
        canvas = draw_avatar(canvas, kp_norm, style, t)

        # Post-processing
        canvas = apply_color_grade(canvas, t)
        canvas = apply_film_grain(canvas, strength=3.5)
        canvas = apply_camera_drift(canvas, t)
        canvas = apply_letterbox(canvas, ratio=0.055)

        writer.write(canvas)

        if (i + 1) % 30 == 0 or i == T - 1:
            print(f"  frame {i+1}/{T}", flush=True)

    writer.release()
    print(f"[avatar] ✓ {out_path}  ({T} frames)")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion", default="all",
                   help="Motion name or 'all' (default: all)")
    p.add_argument("--style", default="studio",
                   choices=list(STYLES.keys()))
    p.add_argument("--width",  type=int, default=512)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--fps",    type=int, default=30)
    p.add_argument("--n-frames", type=int, default=0,
                   help="Number of frames (0 = motion default)")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else ROOT / "outputs" / "avatar" / "videos"
    )

    motions = (
        list(MOTIONS.keys()) if args.motion == "all"
        else [args.motion]
    )

    for motion in motions:
        out_path = out_dir / f"{motion}_avatar.mp4"
        render_motion_video(
            motion_name=motion,
            out_path=out_path,
            width=args.width,
            height=args.height,
            fps=args.fps,
            style_name=args.style,
            n_frames=args.n_frames,
        )

    print(f"\n[avatar] all done → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
