"""Prepare multi-source ControlNet conditioning maps from all video types.

Converts the project's seven OpenPose render types into the two conditioning
signals that the dual-ControlNet generation pipeline actually needs:

  1. Pose maps  (from skeleton videos or JSON keypoints)
     → outputs/avatar/pose_maps/<clip>/XXXXXX_pose.png
     → fed to ControlNet OpenPose (weight 0.85)

  2. Depth maps (from heatmap videos)
     → outputs/avatar/depth_maps/<clip>/XXXXXX_depth.png
     → fed to ControlNet Depth (weight 0.35)

The other video types (neon, mosaic, studio, slowmo) are not used as
conditioning — see docs/AVATAR_DIFFUSION_ARCHITECTURE.md §1 for rationale.

Usage:
    # Extract depth maps from all heatmap videos
    python scripts/avatar/multi_controlnet_conditioning.py \\
        --extract-depth \\
        --input outputs/videos/heatmap/ \\
        --batch

    # Extract depth maps from a single heatmap video
    python scripts/avatar/multi_controlnet_conditioning.py \\
        --extract-depth \\
        --input outputs/videos/heatmap/walking_heatmap.mp4 \\
        --name walking

    # Extract pose maps from skeleton videos (delegates to video_to_pose_maps)
    python scripts/avatar/multi_controlnet_conditioning.py \\
        --extract-pose \\
        --input outputs/videos/skeleton/ \\
        --batch

    # Extract both in one pass
    python scripts/avatar/multi_controlnet_conditioning.py \\
        --extract-pose --extract-depth \\
        --skeleton-dir outputs/videos/skeleton/ \\
        --heatmap-dir  outputs/videos/heatmap/ \\
        --batch

    # Verify conditioning maps for a clip
    python scripts/avatar/multi_controlnet_conditioning.py \\
        --verify --clip walking
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "avatar"))

DEFAULT_WIDTH  = 512
DEFAULT_HEIGHT = 768

# ── Heatmap → depth map conversion ───────────────────────────────────────

def heatmap_frame_to_depth(
    bgr_frame: np.ndarray,
    width: int,
    height: int,
    blur_sigma: float = 3.0,
    normalize: bool = True,
) -> np.ndarray:
    """Convert one heatmap video frame to a ControlNet-compatible depth map.

    The heatmap encodes per-joint Gaussian confidence blobs in colour.
    Strategy:
      1. Convert to grayscale (max across channels captures peak blob intensity)
      2. Gaussian blur to smooth blob boundaries
      3. Normalize to [0, 255] uint8
      4. Resize to target resolution

    The result is a single-channel image where bright pixels = high-confidence
    joint regions (body silhouette) and dark pixels = background. This gives
    ControlNet Depth a soft spatial prior without over-constraining appearance.

    Returns (H, W) uint8 grayscale image.
    """
    # Max across channels — captures the brightest blob regardless of colour
    gray = np.max(bgr_frame, axis=2).astype(np.float32)

    # Smooth blob boundaries
    if blur_sigma > 0:
        k = int(blur_sigma * 4) | 1  # odd kernel size
        gray = cv2.GaussianBlur(gray, (k, k), blur_sigma)

    # Normalize to [0, 255]
    if normalize:
        lo, hi = gray.min(), gray.max()
        if hi > lo:
            gray = (gray - lo) / (hi - lo) * 255.0
        else:
            gray = np.zeros_like(gray)

    gray_u8 = gray.clip(0, 255).astype(np.uint8)

    # Resize to target resolution
    resized = cv2.resize(gray_u8, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return resized


def extract_depth_maps_from_video(
    video_path: Path,
    out_dir: Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    max_frames: int = 0,
    blur_sigma: float = 3.0,
) -> int:
    """Extract depth maps from a heatmap MP4. Returns number of frames written."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        depth = heatmap_frame_to_depth(bgr, width, height, blur_sigma)
        # Save as 3-channel grayscale PNG (ControlNet Depth expects RGB input)
        depth_rgb = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        out_path = out_dir / f"{count:06d}_depth.png"
        cv2.imwrite(str(out_path), depth_rgb)
        count += 1
        if max_frames > 0 and count >= max_frames:
            break

    cap.release()
    print(f"[conditioning] {count} depth maps → {out_dir}")
    return count


def batch_extract_depth(
    heatmap_dir: Path,
    out_base: Path,
    width: int,
    height: int,
    max_frames: int,
) -> None:
    videos = sorted(heatmap_dir.glob("*_heatmap.mp4"))
    if not videos:
        raise FileNotFoundError(f"No *_heatmap.mp4 found in {heatmap_dir}")
    print(f"[conditioning] batch depth extraction: {len(videos)} heatmap videos")
    for vp in videos:
        name = vp.stem.replace("_heatmap", "")
        out_dir = out_base / name
        extract_depth_maps_from_video(vp, out_dir, width, height, max_frames)


# ── Pose map extraction (delegates to video_to_pose_maps) ────────────────

def extract_pose_maps(
    skeleton_dir: Path,
    out_base: Path,
    width: int,
    height: int,
    max_frames: int,
    batch: bool,
) -> None:
    """Delegate pose map extraction to video_to_pose_maps.py."""
    from video_to_pose_maps import (
        extract_frames_from_video,
        save_video_frames_as_pose_maps,
        batch_video,
    )

    if batch:
        batch_video(skeleton_dir, out_base, width, height, max_frames)
    else:
        name = skeleton_dir.stem.replace("_skeleton", "")
        out_dir = out_base / name
        frames = extract_frames_from_video(skeleton_dir, width, height, max_frames)
        save_video_frames_as_pose_maps(frames, out_dir)


# ── Conditioning map loader (used by generate_cinematic_avatar.py) ────────

class ConditioningMaps:
    """Load and pair pose + depth maps for a clip.

    Ensures both sequences have the same length (truncates to the shorter).
    Falls back to pose-only if depth maps are not available.
    """

    def __init__(
        self,
        clip_name: str,
        pose_base: Path | None = None,
        depth_base: Path | None = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        n_frames: int = 0,
    ) -> None:
        self.clip_name = clip_name
        self.width = width
        self.height = height

        if pose_base is None:
            pose_base = ROOT / "outputs" / "avatar" / "pose_maps"
        if depth_base is None:
            depth_base = ROOT / "outputs" / "avatar" / "depth_maps"

        self.pose_dir  = pose_base  / clip_name
        self.depth_dir = depth_base / clip_name
        self.n_frames  = n_frames

    def load_pose_maps(self) -> list:
        """Load pose-map PNGs as PIL Images."""
        from PIL import Image
        pngs = sorted(self.pose_dir.glob("*_pose.png"))
        if not pngs:
            raise FileNotFoundError(
                f"No pose maps found in {self.pose_dir}. "
                "Run multi_controlnet_conditioning.py --extract-pose first."
            )
        if self.n_frames > 0:
            pngs = pngs[:self.n_frames]
        imgs = [
            Image.open(p).convert("RGB").resize(
                (self.width, self.height), Image.LANCZOS
            )
            for p in pngs
        ]
        print(f"[conditioning] loaded {len(imgs)} pose maps for '{self.clip_name}'")
        return imgs

    def load_depth_maps(self) -> list | None:
        """Load depth-map PNGs as PIL Images. Returns None if not available."""
        from PIL import Image
        if not self.depth_dir.exists():
            print(f"[conditioning] no depth maps for '{self.clip_name}' — "
                  "running single-ControlNet mode")
            return None
        pngs = sorted(self.depth_dir.glob("*_depth.png"))
        if not pngs:
            return None
        if self.n_frames > 0:
            pngs = pngs[:self.n_frames]
        imgs = [
            Image.open(p).convert("RGB").resize(
                (self.width, self.height), Image.LANCZOS
            )
            for p in pngs
        ]
        print(f"[conditioning] loaded {len(imgs)} depth maps for '{self.clip_name}'")
        return imgs

    def load_paired(self) -> tuple[list, list | None]:
        """Load pose and depth maps, truncated to the same length."""
        pose_maps  = self.load_pose_maps()
        depth_maps = self.load_depth_maps()

        if depth_maps is not None:
            n = min(len(pose_maps), len(depth_maps))
            pose_maps  = pose_maps[:n]
            depth_maps = depth_maps[:n]

        return pose_maps, depth_maps


# ── Verification ──────────────────────────────────────────────────────────

def verify_conditioning(clip_name: str) -> None:
    """Print a summary of available conditioning maps for a clip."""
    pose_dir  = ROOT / "outputs" / "avatar" / "pose_maps"  / clip_name
    depth_dir = ROOT / "outputs" / "avatar" / "depth_maps" / clip_name

    pose_pngs  = sorted(pose_dir.glob("*_pose.png"))  if pose_dir.exists()  else []
    depth_pngs = sorted(depth_dir.glob("*_depth.png")) if depth_dir.exists() else []

    print(f"\n[verify] clip: {clip_name}")
    print(f"  pose maps  : {len(pose_pngs):4d} frames  ({pose_dir})")
    print(f"  depth maps : {len(depth_pngs):4d} frames  ({depth_dir})")

    if pose_pngs and depth_pngs:
        n = min(len(pose_pngs), len(depth_pngs))
        print(f"  paired     : {n} frames available for dual-ControlNet")
        if len(pose_pngs) != len(depth_pngs):
            print(f"  warning    : frame count mismatch "
                  f"(pose={len(pose_pngs)}, depth={len(depth_pngs)}) "
                  f"— will truncate to {n}")
    elif pose_pngs:
        print(f"  mode       : single-ControlNet (pose only)")
    else:
        print(f"  warning    : no conditioning maps found — run extraction first")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode flags
    p.add_argument("--extract-pose",  action="store_true",
                   help="Extract pose maps from skeleton videos")
    p.add_argument("--extract-depth", action="store_true",
                   help="Extract depth maps from heatmap videos")
    p.add_argument("--verify", action="store_true",
                   help="Verify conditioning maps for a clip")

    # Input paths
    p.add_argument("--input", default=None,
                   help="Input video or directory (single-type mode)")
    p.add_argument("--skeleton-dir", default=None,
                   help="Skeleton video directory (combined mode)")
    p.add_argument("--heatmap-dir",  default=None,
                   help="Heatmap video directory (combined mode)")
    p.add_argument("--name", default=None,
                   help="Clip name for output subdirectory")
    p.add_argument("--clip", default=None,
                   help="Clip name for --verify")
    p.add_argument("--batch", action="store_true",
                   help="Process all videos in the input directory")

    # Output
    p.add_argument("--out-dir", default=None,
                   help="Override output base directory")

    # Settings
    p.add_argument("--width",      type=int,   default=DEFAULT_WIDTH)
    p.add_argument("--height",     type=int,   default=DEFAULT_HEIGHT)
    p.add_argument("--max-frames", type=int,   default=0)
    p.add_argument("--blur-sigma", type=float, default=3.0,
                   help="Gaussian blur sigma for depth map smoothing (default: 3.0)")

    args = p.parse_args()

    if args.verify:
        clip = args.clip or args.name
        if not clip:
            print("[error] --clip required for --verify", file=sys.stderr)
            return 1
        verify_conditioning(clip)
        return 0

    if not args.extract_pose and not args.extract_depth:
        print("[error] specify --extract-pose and/or --extract-depth",
              file=sys.stderr)
        return 1

    out_base = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / "avatar"

    # Combined mode: both skeleton and heatmap dirs provided
    if args.skeleton_dir or args.heatmap_dir:
        if args.extract_pose and args.skeleton_dir:
            extract_pose_maps(
                Path(args.skeleton_dir), out_base / "pose_maps",
                args.width, args.height, args.max_frames, batch=True,
            )
        if args.extract_depth and args.heatmap_dir:
            batch_extract_depth(
                Path(args.heatmap_dir), out_base / "depth_maps",
                args.width, args.height, args.max_frames,
            )
        return 0

    # Single-type mode
    if not args.input:
        print("[error] --input required", file=sys.stderr)
        return 1

    inp = Path(args.input)
    if not inp.exists():
        print(f"[error] not found: {inp}", file=sys.stderr)
        return 1

    if args.extract_pose:
        extract_pose_maps(
            inp, out_base / "pose_maps",
            args.width, args.height, args.max_frames, batch=args.batch,
        )

    if args.extract_depth:
        if args.batch:
            batch_extract_depth(
                inp, out_base / "depth_maps",
                args.width, args.height, args.max_frames,
            )
        else:
            name = args.name or inp.stem.replace("_heatmap", "")
            out_dir = out_base / "depth_maps" / name
            extract_depth_maps_from_video(
                inp, out_dir,
                args.width, args.height, args.max_frames, args.blur_sigma,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
