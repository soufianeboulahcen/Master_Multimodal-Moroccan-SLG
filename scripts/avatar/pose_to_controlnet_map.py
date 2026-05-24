"""Convert existing keypoint data to ControlNet OpenPose conditioning images.

Reads from any of the three keypoint sources already produced by this project:

  --source procedural   outputs/frames/<motion>/  (JPEG skeleton frames)
                        → re-renders as clean pose maps from rig.py keypoints
  --source npz          data/processed/keypoints_2d/<category>/<clip>.npz
                        → (T,54) body + (T,63) hand × 2
  --source prediction   predictions/<run>_<text>.npz
                        → (T,150) SignLLM predicted pose

Output: one PNG per frame in outputs/avatar/pose_maps/<name>/
        Format: RGB 512×768 (portrait) or 768×512 (landscape)
        Colour convention: standard OpenPose DWPose palette

Usage:
    python scripts/avatar/pose_to_controlnet_map.py --source procedural --motion walking
    python scripts/avatar/pose_to_controlnet_map.py --source npz \
        --clip data/processed/keypoints_2d/Pronouns/أَنَا.npz
    python scripts/avatar/pose_to_controlnet_map.py --source prediction \
        --npz predictions/baseline_mse_text.npz
    python scripts/avatar/pose_to_controlnet_map.py --source procedural \
        --motion all   # process all 6 procedural motions
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "generate_openpose"))

# ── OpenPose DWPose colour palette (BGR) ─────────────────────────────────
# Standard 18-point COCO body limb colours used by ControlNet OpenPose.
# These match what ControlNet was trained on — using the same palette
# ensures the model recognises the conditioning correctly.

BODY_LIMBS_COCO18 = [
    # (kp_a, kp_b, BGR_colour)
    (0,  1,  (255, 0,   85)),   # nose - neck
    (1,  2,  (255, 0,   0)),    # neck - R shoulder
    (2,  3,  (255, 85,  0)),    # R shoulder - R elbow
    (3,  4,  (255, 170, 0)),    # R elbow - R wrist
    (1,  5,  (255, 255, 0)),    # neck - L shoulder
    (5,  6,  (170, 255, 0)),    # L shoulder - L elbow
    (6,  7,  (85,  255, 0)),    # L elbow - L wrist
    (1,  8,  (0,   255, 0)),    # neck - mid hip
    (8,  9,  (0,   255, 85)),   # mid hip - R hip
    (9,  10, (0,   255, 170)),  # R hip - R knee
    (10, 11, (0,   255, 255)),  # R knee - R ankle
    (8,  12, (0,   170, 255)),  # mid hip - L hip
    (12, 13, (0,   85,  255)),  # L hip - L knee
    (13, 14, (0,   0,   255)),  # L knee - L ankle
    (0,  15, (255, 0,   170)),  # nose - R eye
    (0,  16, (170, 0,   255)),  # nose - L eye
    (15, 17, (255, 0,   255)),  # R eye - R ear
    (16, 18, (85,  0,   255)),  # L eye - L ear
]

# Hand finger chains (21 keypoints: 0=wrist, 1-4=thumb, 5-8=index, ...)
HAND_FINGER_CHAINS = [
    [0, 1, 2, 3, 4],    # thumb
    [0, 5, 6, 7, 8],    # index
    [0, 9, 10, 11, 12], # middle
    [0, 13, 14, 15, 16],# ring
    [0, 17, 18, 19, 20],# pinky
]
HAND_COLOUR_LEFT  = (255, 200, 100)   # warm yellow
HAND_COLOUR_RIGHT = (100, 200, 255)   # cool cyan

# ── Drawing helpers ───────────────────────────────────────────────────────

def _draw_body(canvas: np.ndarray, kps: np.ndarray,
               thickness: int = 3) -> np.ndarray:
    """Draw COCO-18 body skeleton.

    kps: (N, 3) array of (x_px, y_px, confidence). N >= 19.
    Skips limbs where either endpoint has confidence == 0.
    """
    h, w = canvas.shape[:2]
    for a, b, colour in BODY_LIMBS_COCO18:
        if a >= len(kps) or b >= len(kps):
            continue
        xa, ya, ca = kps[a]
        xb, yb, cb = kps[b]
        if ca < 0.01 or cb < 0.01:
            continue
        xa, ya, xb, yb = int(xa), int(ya), int(xb), int(yb)
        if not (0 <= xa < w and 0 <= ya < h and 0 <= xb < w and 0 <= yb < h):
            continue
        cv2.line(canvas, (xa, ya), (xb, yb), colour, thickness, cv2.LINE_AA)
    # Draw keypoint dots
    for x, y, c in kps[:19]:
        if c < 0.01:
            continue
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas, (x, y), thickness + 1, (255, 255, 255), -1, cv2.LINE_AA)
    return canvas


def _draw_hand(canvas: np.ndarray, kps: np.ndarray,
               colour: tuple, thickness: int = 2) -> np.ndarray:
    """Draw one hand skeleton.

    kps: (21, 3) array of (x_px, y_px, confidence).
    """
    h, w = canvas.shape[:2]
    for chain in HAND_FINGER_CHAINS:
        for a, b in zip(chain, chain[1:]):
            xa, ya, ca = kps[a]
            xb, yb, cb = kps[b]
            if ca < 0.01 or cb < 0.01:
                continue
            xa, ya, xb, yb = int(xa), int(ya), int(xb), int(yb)
            if not (0 <= xa < w and 0 <= ya < h and 0 <= xb < w and 0 <= yb < h):
                continue
            cv2.line(canvas, (xa, ya), (xb, yb), colour, thickness, cv2.LINE_AA)
    for x, y, c in kps:
        if c < 0.01:
            continue
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas, (x, y), thickness, (255, 255, 255), -1, cv2.LINE_AA)
    return canvas


def render_pose_map(
    body_kps: np.ndarray,           # (18 or 19, 3)  x_px, y_px, conf
    left_hand_kps: np.ndarray,      # (21, 3)
    right_hand_kps: np.ndarray,     # (21, 3)
    width: int = 512,
    height: int = 768,
    background: str = "black",      # "black" | "white" | "grey"
) -> np.ndarray:
    """Render a ControlNet-compatible OpenPose conditioning image.

    Returns (H, W, 3) uint8 RGB image.
    """
    if background == "white":
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    elif background == "grey":
        canvas = np.full((height, width, 3), 64, dtype=np.uint8)
    else:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

    _draw_body(canvas, body_kps, thickness=3)
    _draw_hand(canvas, left_hand_kps,  HAND_COLOUR_LEFT,  thickness=2)
    _draw_hand(canvas, right_hand_kps, HAND_COLOUR_RIGHT, thickness=2)

    # Convert BGR → RGB for diffusers (which expects RGB)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


# ── Source-specific loaders ───────────────────────────────────────────────

def load_from_npz(npz_path: Path, width: int, height: int
                  ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load real MoSL clip NPZ → list of (body, left_hand, right_hand) per frame.

    NPZ layout:
        pose_keypoints_2d        (T, 54)  18 body × (x, y, conf)  pixel coords
        hand_left_keypoints_2d   (T, 63)  21 hand × (x, y, conf)
        hand_right_keypoints_2d  (T, 63)
        width, height            scalars  original video resolution
    """
    data = np.load(npz_path)
    T = data["pose_keypoints_2d"].shape[0]
    src_w = float(data.get("width", width))
    src_h = float(data.get("height", height))
    scale_x = width  / src_w
    scale_y = height / src_h

    frames = []
    for t in range(T):
        body = data["pose_keypoints_2d"][t].reshape(18, 3).copy()
        body[:, 0] *= scale_x
        body[:, 1] *= scale_y

        lhand = data["hand_left_keypoints_2d"][t].reshape(21, 3).copy()
        lhand[:, 0] *= scale_x
        lhand[:, 1] *= scale_y

        rhand = data["hand_right_keypoints_2d"][t].reshape(21, 3).copy()
        rhand[:, 0] *= scale_x
        rhand[:, 1] *= scale_y

        frames.append((body, lhand, rhand))
    return frames


def load_from_prediction(npz_path: Path, width: int, height: int
                         ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load SignLLM predicted pose NPZ → list of (body, left_hand, right_hand).

    NPZ layout:
        pose  (T, 150)  50 joints × (x, y, z)
          joints 0..7   = 8 upper-body (COCO subset, Prompt2Sign /9 normalized)
          joints 8..28  = 21 left hand
          joints 29..49 = 21 right hand

    The /9 normalization from pipeline_demo_03_txt2skels.py is undone here.
    Coordinates are then scaled to the target resolution.
    """
    data = np.load(npz_path, allow_pickle=False)
    pose = data["pose"]                         # (T, 150)
    T = pose.shape[0]
    xyz = pose.reshape(T, 50, 3)               # (T, 50, 3)

    # Undo /9 normalization → approximate pixel coords in 460×460 space
    # (MoSL videos are 460×460 per the paper)
    src_size = 460.0
    scale_x = width  / src_size
    scale_y = height / src_size

    # COCO-18 body: we have 8 joints (0..7), pad to 18 with zeros
    body_8 = xyz[:, :8, :2] * 9               # (T, 8, 2) pixel coords
    body_8_scaled = body_8.copy()
    body_8_scaled[:, :, 0] *= scale_x
    body_8_scaled[:, :, 1] *= scale_y

    # Left hand: joints 8..28 (21 joints)
    lhand = xyz[:, 8:29, :2] * 9              # (T, 21, 2)
    lhand[:, :, 0] *= scale_x
    lhand[:, :, 1] *= scale_y

    # Right hand: joints 29..49 (21 joints)
    rhand = xyz[:, 29:50, :2] * 9             # (T, 21, 2)
    rhand[:, :, 0] *= scale_x
    rhand[:, :, 1] *= scale_y

    frames = []
    for t in range(T):
        # Build 18-point body array: fill known 8 joints, zero the rest
        body18 = np.zeros((18, 3), dtype=np.float32)
        # Map 8 upper-body joints to COCO-18 positions
        # COCO-18: 0=Nose,1=Neck,2=RShoulder,3=RElbow,4=RWrist,
        #          5=LShoulder,6=LElbow,7=LWrist
        # Our 8: same order (0..7 from Prompt2Sign idxsPose=[0..7])
        for j in range(8):
            body18[j, 0] = body_8_scaled[t, j, 0]
            body18[j, 1] = body_8_scaled[t, j, 1]
            # Confidence: 1.0 if non-zero, 0.0 if placeholder
            body18[j, 2] = 0.0 if (
                abs(body_8_scaled[t, j, 0]) < 0.5 and
                abs(body_8_scaled[t, j, 1]) < 0.5
            ) else 1.0

        lh = np.zeros((21, 3), dtype=np.float32)
        lh[:, :2] = lhand[t]
        lh[:, 2] = np.where(
            (np.abs(lhand[t, :, 0]) < 0.5) & (np.abs(lhand[t, :, 1]) < 0.5),
            0.0, 1.0
        )

        rh = np.zeros((21, 3), dtype=np.float32)
        rh[:, :2] = rhand[t]
        rh[:, 2] = np.where(
            (np.abs(rhand[t, :, 0]) < 0.5) & (np.abs(rhand[t, :, 1]) < 0.5),
            0.0, 1.0
        )

        frames.append((body18, lh, rh))
    return frames


def load_from_procedural_motion(motion_name: str, width: int, height: int
                                ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Re-generate procedural motion keypoints and convert to pose-map format.

    Imports motions.py from scripts/generate_openpose/ and re-runs the
    generator to get (52, 2) normalized keypoints per frame.
    """
    from motions import MOTIONS
    from rig import N_KEYPOINTS

    gen_fn = MOTIONS[motion_name]
    kp_frames = gen_fn(n_frames=150, fps=30)   # list of (52, 2) normalized

    frames = []
    for kp_norm in kp_frames:
        # Scale to pixel coords
        kp_px = kp_norm.copy()
        kp_px[:, 0] *= width
        kp_px[:, 1] *= height

        # Build COCO-18 body from rig indices 0..18
        # rig.py uses BODY_25 convention: 0=Nose,1=Neck,2=RShoulder,...
        body18 = np.zeros((18, 3), dtype=np.float32)
        for j in range(min(18, N_KEYPOINTS)):
            body18[j, 0] = kp_px[j, 0]
            body18[j, 1] = kp_px[j, 1]
            body18[j, 2] = 1.0  # procedural = always confident

        # Left hand: rig indices 30..40 (11 points) → pad to 21
        lh = np.zeros((21, 3), dtype=np.float32)
        for j in range(11):
            lh[j, 0] = kp_px[30 + j, 0]
            lh[j, 1] = kp_px[30 + j, 1]
            lh[j, 2] = 1.0

        # Right hand: rig indices 41..51 (11 points) → pad to 21
        rh = np.zeros((21, 3), dtype=np.float32)
        for j in range(11):
            rh[j, 0] = kp_px[41 + j, 0]
            rh[j, 1] = kp_px[41 + j, 1]
            rh[j, 2] = 1.0

        frames.append((body18, lh, rh))
    return frames


# ── Main ──────────────────────────────────────────────────────────────────

def process_frames(
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_dir: Path,
    width: int,
    height: int,
    background: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (body, lhand, rhand) in enumerate(frames):
        img = render_pose_map(body, lhand, rhand, width, height, background)
        # Save as PNG (lossless — ControlNet conditioning should not be JPEG-compressed)
        out_path = out_dir / f"{i:06d}_pose.png"
        # PIL not required: cv2 saves RGB correctly if we convert back
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[pose_map] wrote {len(frames)} pose maps → {out_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True,
                   choices=["procedural", "npz", "prediction"],
                   help="Keypoint source")
    p.add_argument("--motion", default="walking",
                   help="Motion name for --source procedural (or 'all')")
    p.add_argument("--clip", default=None,
                   help="Path to .npz for --source npz")
    p.add_argument("--npz", default=None,
                   help="Path to prediction .npz for --source prediction")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: outputs/avatar/pose_maps/<name>)")
    p.add_argument("--width",  type=int, default=512)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--background", default="black",
                   choices=["black", "white", "grey"])
    args = p.parse_args()

    if args.source == "procedural":
        motions_to_run = (
            ["walking", "running", "dancing", "jumping", "hand_face", "waving"]
            if args.motion == "all" else [args.motion]
        )
        for motion in motions_to_run:
            frames = load_from_procedural_motion(motion, args.width, args.height)
            out_dir = (
                Path(args.out_dir) if args.out_dir
                else ROOT / "outputs" / "avatar" / "pose_maps" / motion
            )
            process_frames(frames, out_dir, args.width, args.height, args.background)

    elif args.source == "npz":
        if not args.clip:
            print("error: --clip required for --source npz", file=sys.stderr)
            return 1
        clip_path = Path(args.clip)
        frames = load_from_npz(clip_path, args.width, args.height)
        out_dir = (
            Path(args.out_dir) if args.out_dir
            else ROOT / "outputs" / "avatar" / "pose_maps" / clip_path.stem
        )
        process_frames(frames, out_dir, args.width, args.height, args.background)

    elif args.source == "prediction":
        if not args.npz:
            print("error: --npz required for --source prediction", file=sys.stderr)
            return 1
        npz_path = Path(args.npz)
        frames = load_from_prediction(npz_path, args.width, args.height)
        out_dir = (
            Path(args.out_dir) if args.out_dir
            else ROOT / "outputs" / "avatar" / "pose_maps" / npz_path.stem
        )
        process_frames(frames, out_dir, args.width, args.height, args.background)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
