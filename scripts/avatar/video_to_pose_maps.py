"""Convert existing OpenPose skeleton videos into ControlNet pose-map PNGs.

Two input modes:

  --source video   Read frames directly from a skeleton MP4 (e.g.
                   outputs/videos/skeleton/walking_skeleton.mp4).
                   Each frame is already a rendered OpenPose skeleton on a
                   black background — resize and save as PNG.

  --source json    Read per-frame OpenPose-compatible JSON files from a
                   keypoints directory (e.g. outputs/openpose_json/walking_keypoints/).
                   Re-render clean DWPose-palette pose maps from the raw
                   keypoint coordinates.  Preferred: avoids JPEG compression
                   artefacts and gives exact colour-palette control.

Output: one *_pose.png per frame in outputs/avatar/pose_maps/<name>/
        Format: RGB 512×768 (portrait) — matches generate_avatar_video.py defaults.

Usage:
    # From skeleton video (fastest — no re-render)
    python scripts/avatar/video_to_pose_maps.py \\
        --source video \\
        --input outputs/videos/skeleton/walking_skeleton.mp4 \\
        --name walking

    # From JSON keypoints (best quality — clean re-render)
    python scripts/avatar/video_to_pose_maps.py \\
        --source json \\
        --input outputs/openpose_json/walking_keypoints/ \\
        --name walking

    # Batch: all skeleton videos in a directory
    python scripts/avatar/video_to_pose_maps.py \\
        --source video \\
        --input outputs/videos/skeleton/ \\
        --batch

    # Batch: all JSON keypoint directories
    python scripts/avatar/video_to_pose_maps.py \\
        --source json \\
        --input outputs/openpose_json/ \\
        --batch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "avatar"))

# Re-use the DWPose drawing primitives from pose_to_controlnet_map
from pose_to_controlnet_map import render_pose_map  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────

DEFAULT_WIDTH  = 512
DEFAULT_HEIGHT = 768

# Confidence threshold below which a keypoint is treated as absent
CONF_THR = 0.1


# ── Source: skeleton video ────────────────────────────────────────────────

def extract_frames_from_video(
    video_path: Path,
    width: int,
    height: int,
    max_frames: int = 0,
) -> list[np.ndarray]:
    """Read every frame from a skeleton MP4 and resize to (height, width).

    Returns a list of (H, W, 3) uint8 BGR arrays.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        resized = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LANCZOS4)
        frames.append(resized)
        if max_frames > 0 and len(frames) >= max_frames:
            break

    cap.release()
    print(f"[video_to_pose_maps] read {len(frames)} frames from {video_path.name}")
    return frames


def save_video_frames_as_pose_maps(
    frames: list[np.ndarray],
    out_dir: Path,
) -> None:
    """Save BGR frames as *_pose.png files (converted to RGB for diffusers)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, bgr in enumerate(frames):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out_path = out_dir / f"{i:06d}_pose.png"
        # Save as BGR (cv2 convention) — the RGB conversion is for the
        # diffusers pipeline which reads PIL images; cv2.imwrite expects BGR.
        cv2.imwrite(str(out_path), bgr)
    print(f"[video_to_pose_maps] wrote {len(frames)} pose maps → {out_dir}")


# ── Source: OpenPose JSON keypoints ──────────────────────────────────────

def _parse_keypoints(flat: list[float], n_joints: int) -> np.ndarray:
    """Parse a flat OpenPose keypoint list into (n_joints, 3) float32."""
    arr = np.array(flat, dtype=np.float32).reshape(-1, 3)
    # Pad or truncate to exactly n_joints rows
    out = np.zeros((n_joints, 3), dtype=np.float32)
    n = min(len(arr), n_joints)
    out[:n] = arr[:n]
    return out


def load_keypoints_from_json(json_path: Path) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """Load one OpenPose JSON frame → (body18, left_hand21, right_hand21).

    Handles both the project's own JSON schema (frame_index at top level)
    and the standard CMU OpenPose schema.
    """
    with open(json_path) as f:
        data = json.load(f)

    people = data.get("people", [])
    if not people:
        # Return zero arrays — no person detected in this frame
        return (
            np.zeros((18, 3), dtype=np.float32),
            np.zeros((21, 3), dtype=np.float32),
            np.zeros((21, 3), dtype=np.float32),
        )

    person = people[0]
    body  = _parse_keypoints(person.get("pose_keypoints_2d",  []), 18)
    lhand = _parse_keypoints(person.get("hand_left_keypoints_2d",  []), 21)
    rhand = _parse_keypoints(person.get("hand_right_keypoints_2d", []), 21)
    return body, lhand, rhand


def render_json_keypoints_as_pose_maps(
    json_dir: Path,
    out_dir: Path,
    width: int,
    height: int,
    src_width: int,
    src_height: int,
    max_frames: int = 0,
) -> None:
    """Re-render clean DWPose pose maps from per-frame JSON keypoint files.

    Scales keypoint pixel coordinates from the original video resolution
    (src_width × src_height) to the target (width × height).
    """
    json_files = sorted(json_dir.glob("*_keypoints.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No *_keypoints.json files found in {json_dir}. "
            "Run generate_openpose_video.py first."
        )
    if max_frames > 0:
        json_files = json_files[:max_frames]

    out_dir.mkdir(parents=True, exist_ok=True)
    scale_x = width  / max(src_width,  1)
    scale_y = height / max(src_height, 1)

    for i, jf in enumerate(json_files):
        body, lhand, rhand = load_keypoints_from_json(jf)

        # Scale pixel coordinates to target resolution
        for kps in (body, lhand, rhand):
            kps[:, 0] *= scale_x
            kps[:, 1] *= scale_y

        img_rgb = render_pose_map(
            body, lhand, rhand,
            width=width,
            height=height,
            background="black",
        )
        out_path = out_dir / f"{i:06d}_pose.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        if (i + 1) % 30 == 0 or i == len(json_files) - 1:
            print(f"  rendered {i+1}/{len(json_files)}", flush=True)

    print(f"[video_to_pose_maps] wrote {len(json_files)} pose maps → {out_dir}")


def _detect_source_resolution(json_dir: Path) -> tuple[int, int]:
    """Read the first JSON file to detect the original video resolution."""
    json_files = sorted(json_dir.glob("*_keypoints.json"))
    if not json_files:
        return 1280, 720  # safe fallback

    with open(json_files[0]) as f:
        data = json.load(f)

    size = data.get("image_size", {})
    w = int(size.get("width",  1280))
    h = int(size.get("height",  720))
    return w, h


# ── Batch helpers ─────────────────────────────────────────────────────────

def _stem_from_skeleton_name(name: str) -> str:
    """'walking_skeleton' → 'walking'."""
    return name.replace("_skeleton", "")


def batch_video(
    video_dir: Path,
    out_base: Path,
    width: int,
    height: int,
    max_frames: int,
) -> None:
    videos = sorted(video_dir.glob("*_skeleton.mp4"))
    if not videos:
        raise FileNotFoundError(f"No *_skeleton.mp4 files found in {video_dir}")
    print(f"[video_to_pose_maps] batch: {len(videos)} skeleton videos")
    for vp in videos:
        name = _stem_from_skeleton_name(vp.stem)
        out_dir = out_base / name
        frames = extract_frames_from_video(vp, width, height, max_frames)
        save_video_frames_as_pose_maps(frames, out_dir)


def batch_json(
    json_base: Path,
    out_base: Path,
    width: int,
    height: int,
    max_frames: int,
) -> None:
    json_dirs = sorted(d for d in json_base.iterdir()
                       if d.is_dir() and list(d.glob("*_keypoints.json")))
    if not json_dirs:
        raise FileNotFoundError(
            f"No keypoint directories found under {json_base}"
        )
    print(f"[video_to_pose_maps] batch: {len(json_dirs)} keypoint directories")
    for jd in json_dirs:
        name = jd.name.replace("_keypoints", "")
        out_dir = out_base / name
        src_w, src_h = _detect_source_resolution(jd)
        print(f"  {jd.name}  src={src_w}×{src_h}")
        render_json_keypoints_as_pose_maps(
            jd, out_dir, width, height, src_w, src_h, max_frames
        )


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source", required=True, choices=["video", "json"],
        help="Input source: 'video' = skeleton MP4, 'json' = keypoint JSON dir",
    )
    p.add_argument(
        "--input", required=True,
        help="Path to skeleton MP4 / keypoints dir, or parent dir when --batch",
    )
    p.add_argument(
        "--name", default=None,
        help="Output subdirectory name under outputs/avatar/pose_maps/ "
             "(inferred from filename if omitted)",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Override output directory (default: outputs/avatar/pose_maps/<name>)",
    )
    p.add_argument("--width",  type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument(
        "--max-frames", type=int, default=0,
        help="Limit number of frames (0 = all)",
    )
    p.add_argument(
        "--batch", action="store_true",
        help="Process all videos/JSON dirs under --input",
    )
    # JSON-source only
    p.add_argument(
        "--src-width",  type=int, default=0,
        help="Original video width for JSON source (auto-detected if 0)",
    )
    p.add_argument(
        "--src-height", type=int, default=0,
        help="Original video height for JSON source (auto-detected if 0)",
    )
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"[error] not found: {inp}", file=sys.stderr)
        return 1

    out_base = ROOT / "outputs" / "avatar" / "pose_maps"

    if args.batch:
        if args.source == "video":
            batch_video(inp, out_base, args.width, args.height, args.max_frames)
        else:
            batch_json(inp, out_base, args.width, args.height, args.max_frames)
        return 0

    # Single-item mode
    name = args.name or (
        _stem_from_skeleton_name(inp.stem)
        if args.source == "video"
        else inp.name.replace("_keypoints", "")
    )
    out_dir = Path(args.out_dir) if args.out_dir else out_base / name

    if args.source == "video":
        frames = extract_frames_from_video(
            inp, args.width, args.height, args.max_frames
        )
        save_video_frames_as_pose_maps(frames, out_dir)
    else:
        src_w = args.src_width  or None
        src_h = args.src_height or None
        if src_w is None or src_h is None:
            detected_w, detected_h = _detect_source_resolution(inp)
            src_w = src_w or detected_w
            src_h = src_h or detected_h
        render_json_keypoints_as_pose_maps(
            inp, out_dir,
            args.width, args.height,
            src_w, src_h,
            args.max_frames,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
