"""Generate cinematic avatar videos from outputs/pose_control/ PNG frames.

Reads the pose_control directories (each containing pose_XXXXXX.png files
and a manifest.json) and composites them onto a cinematic background with
glow, colour grading, film grain, camera drift, and letterbox.

Three output styles: studio (dark warm), dark (near-black), bright (light).

Usage:
    # All clips, studio style
    python scripts/avatar/pose_control_to_video.py

    # Single clip, all styles
    python scripts/avatar/pose_control_to_video.py \
        --clip أَنْتِ_keypoints --style all

    # Custom FPS / resolution
    python scripts/avatar/pose_control_to_video.py \
        --clip أَنْتِ_keypoints --style dark --fps 25 --size 768
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT       = Path(__file__).resolve().parents[2]
CTRL_BASE  = ROOT / "outputs" / "pose_control"
VIDEO_BASE = ROOT / "outputs" / "avatar" / "videos" / "pose_control"

# ── Style presets ─────────────────────────────────────────────────────────

STYLES: dict[str, dict] = {
    "studio": {
        "bg_top":    (18,  14,  12),
        "bg_bottom": (8,   6,   5),
        "glow_col":  (80,  100, 160),
        "floor_col": (25,  22,  20),
        "grain":     3.5,
    },
    "dark": {
        "bg_top":    (5,   5,   10),
        "bg_bottom": (2,   2,   5),
        "glow_col":  (40,  60,  120),
        "floor_col": (12,  10,  8),
        "grain":     2.5,
    },
    "bright": {
        "bg_top":    (220, 215, 210),
        "bg_bottom": (180, 175, 170),
        "glow_col":  (100, 120, 200),
        "floor_col": (160, 155, 150),
        "grain":     2.0,
    },
}

# ── Background ────────────────────────────────────────────────────────────

def make_background(h: int, w: int, t: float, style: dict) -> np.ndarray:
    bg = np.zeros((h, w, 3), dtype=np.float32)
    ys = np.linspace(0, 1, h)[:, None]          # (H,1)
    for c in range(3):
        bg[:, :, c] = (
            style["bg_top"][c]    * (1 - ys) +
            style["bg_bottom"][c] * ys
        )
    # Subtle colour pulse
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * t * 0.4)
    bg[:, :, 0] += pulse * 3
    bg[:, :, 2] += (1 - pulse) * 5
    # Radial vignette
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X - w/2) / (w*0.6))**2 + ((Y - h*0.45) / (h*0.6))**2)
    vignette = np.clip(1.0 - dist * 0.55, 0.0, 1.0)[..., np.newaxis]
    bg *= vignette
    # Floor line
    fy = int(h * 0.82)
    gc = np.array(style["floor_col"], dtype=np.float32)
    bg[fy:fy+2, :] = bg[fy:fy+2] * 0.82 + gc * 0.18
    for row in range(fy, min(fy + 60, h)):
        fade = 1.0 - (row - fy) / 60.0
        bg[row] *= (1.0 - fade * 0.3)
    return np.clip(bg, 0, 255).astype(np.uint8)

# ── Post-processing ───────────────────────────────────────────────────────

def color_grade(frame: np.ndarray, t: float) -> np.ndarray:
    f = frame.astype(np.float32)
    shadow = np.clip(1.0 - f / 128.0, 0, 1)
    f[:, :, 2] += shadow[:, :, 2] * 4
    f[:, :, 0] -= shadow[:, :, 0] * 2
    hi = np.clip(f / 200.0 - 0.5, 0, 1)
    f[:, :, 0] += hi[:, :, 0] * 3
    return np.clip(f, 0, 255).astype(np.uint8)


def film_grain(frame: np.ndarray, strength: float) -> np.ndarray:
    noise = np.random.normal(0, strength, frame.shape).astype(np.float32)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def camera_drift(frame: np.ndarray, t: float) -> np.ndarray:
    dx = int(4.0 * np.sin(2 * np.pi * t * 0.25))
    dy = int(2.5 * np.cos(2 * np.pi * t * 0.18))
    M  = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))


def letterbox(frame: np.ndarray, ratio: float = 0.055) -> np.ndarray:
    bar = int(frame.shape[0] * ratio)
    out = frame.copy()
    out[:bar]  = 0
    out[-bar:] = 0
    return out

# ── Compositor ────────────────────────────────────────────────────────────

def composite(
    pose_bgr: np.ndarray,
    bg: np.ndarray,
    glow_col: tuple,
) -> np.ndarray:
    """Overlay a DWPose skeleton frame onto a cinematic background.

    Black pixels in the pose map are treated as transparent.
    Coloured skeleton lines are blended with additive glow.
    """
    h, w = bg.shape[:2]
    pose = cv2.resize(pose_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Binary mask: non-black pixels = skeleton
    gray = cv2.cvtColor(pose, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    m = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    # Glow halo around skeleton lines
    blurred   = cv2.GaussianBlur(pose, (0, 0), sigmaX=14)
    glow_tint = np.full_like(pose, glow_col, dtype=np.float32)
    glow      = cv2.addWeighted(
        blurred.astype(np.float32), 0.45,
        glow_tint, 0.12, 0
    ).astype(np.uint8)

    result = bg.astype(np.float32)
    result += glow.astype(np.float32) * 0.30          # ambient glow on bg
    result  = np.clip(result, 0, 255)
    # Skeleton lines on top
    result  = result * (1 - m * 0.85) + pose.astype(np.float32) * m * 0.85
    return np.clip(result, 0, 255).astype(np.uint8)

# ── Renderer ──────────────────────────────────────────────────────────────

def render_clip(
    clip_dir: Path,
    style_name: str,
    fps: int,
    size: int,
) -> Path:
    pngs = sorted(clip_dir.glob("pose_*.png"))
    if not pngs:
        raise FileNotFoundError(f"No pose_*.png in {clip_dir}")

    # Read native FPS from manifest if available
    manifest = clip_dir / "manifest.json"
    native_fps = fps
    if manifest.exists():
        with open(manifest) as f:
            meta = json.load(f)
        native_fps = meta.get("fps", fps)
        canvas     = meta.get("canvas", size)
    else:
        canvas = size

    out_fps = fps or native_fps
    T       = len(pngs)
    style   = STYLES[style_name]
    h = w   = size   # square output matching pose_control canvas

    out_path = VIDEO_BASE / f"{clip_dir.name}_{style_name}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {out_path}")

    print(f"[pose_control→video] {clip_dir.name}  "
          f"{T} frames @ {out_fps}fps  style={style_name}  {w}×{h}")

    np.random.seed(42)
    for i, png in enumerate(pngs):
        t = i / max(T - 1, 1)

        pose_bgr = cv2.imread(str(png))
        if pose_bgr is None:
            print(f"  [warn] cannot read {png.name}")
            continue

        bg    = make_background(h, w, t, style)
        frame = composite(pose_bgr, bg, style["glow_col"])
        frame = color_grade(frame, t)
        frame = film_grain(frame, style["grain"])
        frame = camera_drift(frame, t)
        frame = letterbox(frame)

        writer.write(frame)

        if (i + 1) % 20 == 0 or i == T - 1:
            print(f"  frame {i+1}/{T}", flush=True)

    writer.release()
    print(f"[pose_control→video] ✓  {out_path.name}")
    return out_path

# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--clip",  default="all",
                   help="Clip directory name under outputs/pose_control/, "
                        "or 'all' (default: all)")
    p.add_argument("--style", default="all",
                   choices=list(STYLES.keys()) + ["all"],
                   help="Visual style (default: all)")
    p.add_argument("--fps",  type=int, default=25,
                   help="Output FPS (default: 25, or from manifest)")
    p.add_argument("--size", type=int, default=768,
                   help="Output frame size in pixels — square (default: 768)")
    args = p.parse_args()

    # Resolve clip directories
    if args.clip == "all":
        clip_dirs = sorted(d for d in CTRL_BASE.iterdir()
                           if d.is_dir() and list(d.glob("pose_*.png")))
    else:
        clip_dirs = [CTRL_BASE / args.clip]

    if not clip_dirs:
        print(f"[error] no pose_control clip directories found in {CTRL_BASE}",
              file=sys.stderr)
        return 1

    # Resolve styles
    styles = list(STYLES.keys()) if args.style == "all" else [args.style]

    total = 0
    for clip_dir in clip_dirs:
        for style in styles:
            try:
                render_clip(clip_dir, style, args.fps, args.size)
                total += 1
            except Exception as e:
                print(f"[error] {clip_dir.name} / {style}: {e}", file=sys.stderr)

    print(f"\n[pose_control→video] done — {total} video(s) → {VIDEO_BASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
