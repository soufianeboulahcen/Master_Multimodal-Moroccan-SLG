"""Convert ControlNet pose-map PNGs into a cinematic avatar video (CPU-only).

Reads the DWPose RGB pose maps produced by video_to_pose_maps.py and
composites a stylised human figure on a cinematic background, producing
a smooth avatar video without requiring any GPU or diffusion models.

This bridges the gap between the pose-map extraction stage and the
full diffusion pipeline: it produces a watchable avatar video immediately
from any pose-map directory.

Usage:
    python scripts/avatar/pose_maps_to_video.py --clip walking
    python scripts/avatar/pose_maps_to_video.py --clip all
    python scripts/avatar/pose_maps_to_video.py --clip walking --style dark --fps 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "avatar"))

POSE_BASE  = ROOT / "outputs" / "avatar" / "pose_maps"
VIDEO_BASE = ROOT / "outputs" / "avatar" / "videos" / "from_pose_maps"

# ── Cinematic background ──────────────────────────────────────────────────

STYLES = {
    "studio": {
        "bg_top":    (18, 14, 12),
        "bg_bottom": (8,  6,  5),
        "glow_col":  (80, 100, 160),
        "floor_col": (25, 22, 20),
    },
    "dark": {
        "bg_top":    (5,  5,  10),
        "bg_bottom": (2,  2,   5),
        "glow_col":  (40, 60, 120),
        "floor_col": (12, 10,   8),
    },
    "bright": {
        "bg_top":    (220, 215, 210),
        "bg_bottom": (180, 175, 170),
        "glow_col":  (100, 120, 200),
        "floor_col": (160, 155, 150),
    },
}


def make_background(h: int, w: int, t: float, style: dict) -> np.ndarray:
    bg = np.zeros((h, w, 3), dtype=np.float32)
    for row in range(h):
        alpha = row / h
        for c in range(3):
            bg[row, :, c] = (
                style["bg_top"][c] * (1 - alpha) +
                style["bg_bottom"][c] * alpha
            )
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * t * 0.4)
    bg[:, :, 0] += pulse * 3
    bg[:, :, 2] += (1 - pulse) * 5
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h * 0.45
    dist = np.sqrt(((X - cx) / (w * 0.6)) ** 2 + ((Y - cy) / (h * 0.6)) ** 2)
    vignette = np.clip(1.0 - dist * 0.55, 0.0, 1.0)[..., np.newaxis]
    bg = bg * vignette
    floor_y = int(h * 0.82)
    gc = style["floor_col"]
    bg[floor_y:floor_y + 2, :] = (
        bg[floor_y:floor_y + 2] * 0.82 +
        np.array(gc, dtype=np.float32) * 0.18
    )
    return np.clip(bg, 0, 255).astype(np.uint8)


def apply_color_grade(frame: np.ndarray, t: float) -> np.ndarray:
    f = frame.astype(np.float32)
    shadow_mask = np.clip(1.0 - f / 128.0, 0, 1)
    f[:, :, 2] += shadow_mask[:, :, 2] * 4
    f[:, :, 0] -= shadow_mask[:, :, 0] * 2
    highlight_mask = np.clip(f / 200.0 - 0.5, 0, 1)
    f[:, :, 0] += highlight_mask[:, :, 0] * 3
    return np.clip(f, 0, 255).astype(np.uint8)


def apply_film_grain(frame: np.ndarray, strength: float = 3.5) -> np.ndarray:
    noise = np.random.normal(0, strength, frame.shape).astype(np.float32)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_letterbox(frame: np.ndarray, ratio: float = 0.055) -> np.ndarray:
    h = frame.shape[0]
    bar = int(h * ratio)
    out = frame.copy()
    out[:bar] = 0
    out[-bar:] = 0
    return out


def apply_camera_drift(frame: np.ndarray, t: float) -> np.ndarray:
    dx = int(4.0 * np.sin(2 * np.pi * t * 0.25))
    dy = int(2.5 * np.cos(2 * np.pi * t * 0.18))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))


# ── Pose map compositor ───────────────────────────────────────────────────

def composite_pose_on_background(
    pose_bgr: np.ndarray,
    background: np.ndarray,
    glow_col: tuple,
    blend_alpha: float = 0.82,
) -> np.ndarray:
    """Composite a DWPose skeleton frame onto a cinematic background.

    Strategy:
    - Black pixels in the pose map = background (transparent)
    - Coloured pixels = skeleton lines → overlay with additive blend + glow
    """
    h, w = background.shape[:2]
    pose_resized = cv2.resize(pose_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Mask: pixels that are not black in the pose map
    gray = cv2.cvtColor(pose_resized, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    # Glow: blur the skeleton lines and tint with glow colour
    blurred = cv2.GaussianBlur(pose_resized, (0, 0), sigmaX=12)
    glow_tint = np.zeros_like(pose_resized, dtype=np.float32)
    glow_tint[:] = glow_col
    glow = cv2.addWeighted(
        blurred.astype(np.float32), 0.4,
        glow_tint, 0.15, 0
    ).astype(np.uint8)

    # Composite: background + glow everywhere + skeleton lines on top
    result = background.copy().astype(np.float32)
    result += glow.astype(np.float32) * 0.25          # ambient glow
    result = np.clip(result, 0, 255)
    # Overlay skeleton lines
    result = result * (1 - mask_3ch * blend_alpha) + \
             pose_resized.astype(np.float32) * mask_3ch * blend_alpha
    return np.clip(result, 0, 255).astype(np.uint8)


# ── Main renderer ─────────────────────────────────────────────────────────

def render_clip(
    clip_name: str,
    style_name: str = "studio",
    fps: int = 30,
    width: int = 512,
    height: int = 768,
    max_frames: int = 0,
) -> Path:
    pose_dir = POSE_BASE / clip_name
    pngs = sorted(pose_dir.glob("*_pose.png"))
    if not pngs:
        raise FileNotFoundError(
            f"No pose maps in {pose_dir}. "
            "Run video_to_pose_maps.py first."
        )
    if max_frames > 0:
        pngs = pngs[:max_frames]

    style = STYLES[style_name]
    T = len(pngs)

    out_path = VIDEO_BASE / f"{clip_name}_{style_name}_avatar.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {out_path}")

    print(f"[pose→video] {clip_name} ({T} frames @ {fps}fps, style={style_name})")
    np.random.seed(42)

    for i, png_path in enumerate(pngs):
        t = i / max(T - 1, 1)

        pose_bgr = cv2.imread(str(png_path))
        if pose_bgr is None:
            continue

        bg = make_background(height, width, t, style)
        frame = composite_pose_on_background(
            pose_bgr, bg, style["glow_col"]
        )
        frame = apply_color_grade(frame, t)
        frame = apply_film_grain(frame, strength=3.0)
        frame = apply_camera_drift(frame, t)
        frame = apply_letterbox(frame, ratio=0.055)

        writer.write(frame)

        if (i + 1) % 50 == 0 or i == T - 1:
            print(f"  frame {i+1}/{T}", flush=True)

    writer.release()
    print(f"[pose→video] ✓ {out_path}")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", default="all",
                   help="Clip name or 'all' (default: all)")
    p.add_argument("--style", default="studio",
                   choices=list(STYLES.keys()))
    p.add_argument("--fps",   type=int, default=30)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()

    if args.clip == "all":
        clips = [d.name for d in sorted(POSE_BASE.iterdir()) if d.is_dir()]
        if not clips:
            print(f"[error] no pose map directories found in {POSE_BASE}")
            return 1
    else:
        clips = [args.clip]

    for clip in clips:
        try:
            render_clip(clip, args.style, args.fps,
                        args.width, args.height, args.max_frames)
        except FileNotFoundError as e:
            print(f"[skip] {clip}: {e}")

    print(f"\n[pose→video] all done → {VIDEO_BASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
