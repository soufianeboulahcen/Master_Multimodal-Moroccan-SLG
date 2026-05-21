"""Convert OpenPose JSON frames into ControlNet RGB pose map images.

This module is the primary bridge between the SignLLM skeleton pipeline and
the diffusion rendering backends.  It consumes the per-frame OpenPose v1.3
JSON files produced by mosl/pose/export_openpose_json.py (and the procedural
generator in scripts/generate_openpose/) and renders them as RGB images that
ControlNet-OpenPose expects as conditioning input.

Keypoint remapping handled here:
    Body:  our JSON has BODY_25 (25 kpts) or COCO-18 (18 kpts) depending on
           source.  ControlNet-OpenPose was trained on COCO-18 (indices 0–17).
           Indices 0–17 are identical between BODY_25 and COCO-18 for the
           upper-body joints that matter for sign language.  Indices 18–24
           (feet, mid-hip) are dropped.
    Hands: our JSON has 11 keypoints per hand (palm + 2 joints per finger,
           no fingertips).  ControlNet expects 21 keypoints (full OpenPose
           hand model: palm + 4 joints per finger).  Missing distal joints
           are linearly extrapolated from the two proximal joints so the
           finger topology is complete.

Two rendering modes:
    body_map:  full skeleton on black canvas — fed to the body ControlNet branch
    hand_map:  hands only, optionally supersampled — fed to the hand ControlNet branch

Colour convention follows DWPose (the current ControlNet training standard):
    body limbs:  HSV colour wheel, one hue per limb
    hands:       left=blue family, right=red family; per-finger colours

Usage:
    renderer = PoseMapRenderer(cfg)
    body_maps, hand_maps = renderer.render_sequence(json_dir)
    renderer.save_maps(body_maps, out_dir / "body")
    renderer.save_maps(hand_maps, out_dir / "hand")
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from avatar.config import PoseMapConfig


# ---------------------------------------------------------------------------
# DWPose colour tables
# ---------------------------------------------------------------------------

# COCO-18 body limb pairs (joint_a, joint_b).
# Indices: 0=Nose 1=Neck 2=RShoulder 3=RElbow 4=RWrist 5=LShoulder
#          6=LElbow 7=LWrist 8=RHip 9=RKnee 10=RAnkle 11=LHip
#          12=LKnee 13=LAnkle 14=REye 15=LEye 16=REar 17=LEar
BODY_LIMBS: list[tuple[int, int]] = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),   # arms
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),  # legs
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),      # head
    (2, 8), (5, 11),                                    # torso
]

# Per-limb BGR colours (DWPose convention).
_BODY_LIMB_COLOURS: list[tuple[int, int, int]] = [
    (255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0),
    (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
    (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255),
    (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]

# Per-joint BGR colours for body keypoints.
_BODY_KP_COLOURS: list[tuple[int, int, int]] = [
    (255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0),
    (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255),
]

# Hand finger topology: each finger is a chain from palm (index 0).
# Full 21-point OpenPose hand model:
#   0=palm, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
_HAND_FINGERS: list[list[int]] = [
    [0, 1, 2, 3, 4],       # thumb
    [0, 5, 6, 7, 8],       # index
    [0, 9, 10, 11, 12],    # middle
    [0, 13, 14, 15, 16],   # ring
    [0, 17, 18, 19, 20],   # pinky
]

# Per-finger BGR colours (left hand: blue family; right hand: red family).
_LEFT_HAND_FINGER_COLOURS: list[tuple[int, int, int]] = [
    (255, 153, 0), (255, 204, 0), (255, 255, 0),
    (153, 255, 0), (0, 255, 0),
]
_RIGHT_HAND_FINGER_COLOURS: list[tuple[int, int, int]] = [
    (0, 153, 255), (0, 204, 255), (0, 255, 255),
    (0, 255, 153), (0, 255, 0),
]


# ---------------------------------------------------------------------------
# Keypoint parsing helpers
# ---------------------------------------------------------------------------

def _parse_kpts(flat: list[float], n_kpts: int) -> np.ndarray:
    """Parse a flat [x, y, conf, x, y, conf, ...] list into (n_kpts, 3)."""
    arr = np.array(flat, dtype=np.float32).reshape(-1, 3)
    if arr.shape[0] < n_kpts:
        pad = np.zeros((n_kpts - arr.shape[0], 3), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return arr[:n_kpts]


def _scale_kpts(
    kpts: np.ndarray,
    src_w: int, src_h: int,
    dst_w: int, dst_h: int,
) -> np.ndarray:
    """Scale pixel coordinates from source to destination resolution."""
    out = kpts.copy()
    out[:, 0] *= dst_w / src_w
    out[:, 1] *= dst_h / src_h
    return out


def _extrapolate_hand_21(kpts_11: np.ndarray) -> np.ndarray:
    """Expand 11-keypoint hand (palm + 2 joints/finger) to full 21-point model.

    Our OpenPose JSON has 11 hand keypoints:
        0=palm, 1-2=thumb, 3-4=index, 5-6=middle, 7-8=ring, 9-10=pinky
    Full model needs 21:
        0=palm, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky

    Missing joints (fingertips and one intermediate per finger) are linearly
    extrapolated: given joints A and B along a finger, C = B + (B - A).
    Confidence of extrapolated joints is set to 0.5 (lower than detected).
    """
    if kpts_11.shape[0] == 0:
        return np.zeros((21, 3), dtype=np.float32)

    # Mapping: 11-pt index → (palm_idx, proximal_idx) in 11-pt space
    # 11-pt layout: 0=palm, then pairs (proximal, distal) per finger
    # finger order: thumb(1,2), index(3,4), middle(5,6), ring(7,8), pinky(9,10)
    finger_pairs_11 = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

    out = np.zeros((21, 3), dtype=np.float32)
    out[0] = kpts_11[0]  # palm

    for f_idx, (prox_11, dist_11) in enumerate(finger_pairs_11):
        # Full model finger joints: palm, j1, j2, j3, tip (4 joints after palm)
        base_21 = 1 + f_idx * 4   # first joint of this finger in 21-pt model

        palm = kpts_11[0, :2]
        prox = kpts_11[prox_11, :2]
        dist = kpts_11[dist_11, :2]
        conf_prox = float(kpts_11[prox_11, 2])
        conf_dist = float(kpts_11[dist_11, 2])

        if conf_prox < 0.05 and conf_dist < 0.05:
            # Both detected joints are missing — leave all four as zero.
            continue

        # j1: interpolate between palm and proximal (1/3 of the way)
        j1 = palm + (prox - palm) * 0.5
        # j2: the proximal joint itself
        j2 = prox
        # j3: interpolate between proximal and distal
        j3 = prox + (dist - prox) * 0.5 if conf_dist > 0.05 else prox + (prox - palm) * 0.3
        # tip: extrapolate beyond distal
        tip = dist + (dist - prox) * 0.4 if conf_dist > 0.05 else dist

        out[base_21 + 0] = [j1[0], j1[1], conf_prox * 0.8]
        out[base_21 + 1] = [j2[0], j2[1], conf_prox]
        out[base_21 + 2] = [j3[0], j3[1], (conf_prox + conf_dist) * 0.4]
        out[base_21 + 3] = [tip[0], tip[1], conf_dist * 0.7]

    return out


# ---------------------------------------------------------------------------
# Rendering primitives
# ---------------------------------------------------------------------------

def _draw_limb(
    canvas: np.ndarray,
    kpts: np.ndarray,
    a: int, b: int,
    colour: tuple[int, int, int],
    thickness: int,
    conf_thresh: float,
) -> None:
    """Draw a single limb between keypoints a and b if both are confident."""
    if kpts[a, 2] < conf_thresh or kpts[b, 2] < conf_thresh:
        return
    xa, ya = int(round(kpts[a, 0])), int(round(kpts[a, 1]))
    xb, yb = int(round(kpts[b, 0])), int(round(kpts[b, 1]))
    cv2.line(canvas, (xa, ya), (xb, yb), colour, thickness, cv2.LINE_AA)


def _draw_joint(
    canvas: np.ndarray,
    kpts: np.ndarray,
    idx: int,
    colour: tuple[int, int, int],
    radius: int,
    conf_thresh: float,
) -> None:
    if kpts[idx, 2] < conf_thresh:
        return
    x, y = int(round(kpts[idx, 0])), int(round(kpts[idx, 1]))
    cv2.circle(canvas, (x, y), radius, colour, -1, cv2.LINE_AA)


def _render_body(
    canvas: np.ndarray,
    body_kpts: np.ndarray,   # (18, 3) COCO-18
    conf_thresh: float,
    limb_thickness: int = 3,
    joint_radius: int = 4,
) -> None:
    """Draw body skeleton onto canvas in-place."""
    for i, (a, b) in enumerate(BODY_LIMBS):
        colour = _BODY_LIMB_COLOURS[i % len(_BODY_LIMB_COLOURS)]
        _draw_limb(canvas, body_kpts, a, b, colour, limb_thickness, conf_thresh)
    for i in range(min(18, body_kpts.shape[0])):
        colour = _BODY_KP_COLOURS[i % len(_BODY_KP_COLOURS)]
        _draw_joint(canvas, body_kpts, i, colour, joint_radius, conf_thresh)


def _render_hand(
    canvas: np.ndarray,
    hand_kpts: np.ndarray,   # (21, 3) full model
    finger_colours: list[tuple[int, int, int]],
    conf_thresh: float,
    limb_thickness: int = 2,
    joint_radius: int = 3,
) -> None:
    """Draw one hand skeleton onto canvas in-place."""
    for f_idx, finger in enumerate(_HAND_FINGERS):
        colour = finger_colours[f_idx % len(finger_colours)]
        for a, b in zip(finger, finger[1:]):
            _draw_limb(canvas, hand_kpts, a, b, colour, limb_thickness, conf_thresh)
        for j in finger:
            _draw_joint(canvas, hand_kpts, j, colour, joint_radius, conf_thresh)


# ---------------------------------------------------------------------------
# Main renderer class
# ---------------------------------------------------------------------------

class PoseMapRenderer:
    """Renders a directory of OpenPose JSON frames into ControlNet pose maps.

    Produces two sets of images per frame:
        body map:  full skeleton (body + hands) on black canvas
        hand map:  hands only, optionally supersampled

    Both sets are used as separate ControlNet conditioning branches in the
    diffusion pipeline, with the hand branch receiving higher weight.
    """

    def __init__(self, cfg: Optional[PoseMapConfig] = None) -> None:
        self.cfg = cfg or PoseMapConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_sequence(
        self,
        json_dir: str | Path,
        verbose: bool = False,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Render all JSON frames in json_dir.

        Returns
        -------
        body_maps : list of (H, W, 3) uint8 BGR arrays
            Full skeleton pose maps for the body ControlNet branch.
        hand_maps : list of (H, W, 3) uint8 BGR arrays
            Hands-only pose maps for the hand ControlNet branch.
        """
        json_dir = Path(json_dir)
        json_files = sorted(json_dir.glob("*_keypoints.json"))
        if not json_files:
            raise FileNotFoundError(f"No *_keypoints.json files found in {json_dir}")

        body_maps: list[np.ndarray] = []
        hand_maps: list[np.ndarray] = []

        for i, jf in enumerate(json_files):
            body, hand = self._render_frame(jf)
            body_maps.append(body)
            hand_maps.append(hand)
            if verbose and (i + 1) % 25 == 0:
                print(f"  rendered {i + 1}/{len(json_files)} frames")

        return body_maps, hand_maps

    def render_frame(self, json_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Render a single JSON frame.  Returns (body_map, hand_map)."""
        return self._render_frame(Path(json_path))

    def save_maps(
        self,
        maps: list[np.ndarray],
        out_dir: str | Path,
        prefix: str = "",
    ) -> list[Path]:
        """Save a list of pose map images to out_dir as PNG files."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for i, img in enumerate(maps):
            name = f"{prefix}{i:06d}.png" if prefix else f"{i:06d}.png"
            p = out_dir / name
            cv2.imwrite(str(p), img)
            paths.append(p)
        return paths

    def render_and_save(
        self,
        json_dir: str | Path,
        body_out_dir: str | Path,
        hand_out_dir: str | Path,
        verbose: bool = False,
    ) -> tuple[list[Path], list[Path]]:
        """Convenience: render sequence and save both map sets."""
        body_maps, hand_maps = self.render_sequence(json_dir, verbose=verbose)
        body_paths = self.save_maps(body_maps, body_out_dir, prefix="body_")
        hand_paths = self.save_maps(hand_maps, hand_out_dir, prefix="hand_")
        return body_paths, hand_paths

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _render_frame(self, json_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Parse one JSON file and produce (body_map, hand_map)."""
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        cfg = self.cfg
        W, H = cfg.width, cfg.height
        src_W, src_H = cfg.source_width, cfg.source_height
        conf_thresh = cfg.confidence_threshold

        # Extract keypoints from the first person entry.
        people = data.get("people", [])
        if people:
            person = people[0]
        else:
            person = {}

        body_flat = person.get("pose_keypoints_2d", [])
        lhand_flat = person.get("hand_left_keypoints_2d", [])
        rhand_flat = person.get("hand_right_keypoints_2d", [])

        # Parse into (N, 3) arrays.
        # Body: take first 18 keypoints (COCO-18 subset of BODY_25).
        body_kpts = _parse_kpts(body_flat, 25)[:18]   # drop feet/mid-hip
        body_kpts = _scale_kpts(body_kpts, src_W, src_H, W, H)

        # Hands: expand 11-pt → 21-pt, then scale.
        lhand_kpts_11 = _parse_kpts(lhand_flat, 11)
        rhand_kpts_11 = _parse_kpts(rhand_flat, 11)
        lhand_kpts = _extrapolate_hand_21(lhand_kpts_11)
        rhand_kpts = _extrapolate_hand_21(rhand_kpts_11)
        lhand_kpts = _scale_kpts(lhand_kpts, src_W, src_H, W, H)
        rhand_kpts = _scale_kpts(rhand_kpts, src_W, src_H, W, H)

        # --- Body map (full skeleton) ---
        body_canvas = np.zeros((H, W, 3), dtype=np.uint8)
        _render_body(body_canvas, body_kpts, conf_thresh)
        _render_hand(body_canvas, lhand_kpts, _LEFT_HAND_FINGER_COLOURS, conf_thresh)
        _render_hand(body_canvas, rhand_kpts, _RIGHT_HAND_FINGER_COLOURS, conf_thresh)

        # --- Hand map (hands only, optionally supersampled) ---
        if cfg.hand_supersample:
            sf = cfg.hand_supersample_factor
            hand_canvas_big = np.zeros((H * sf, W * sf, 3), dtype=np.uint8)
            lhand_big = lhand_kpts.copy()
            lhand_big[:, :2] *= sf
            rhand_big = rhand_kpts.copy()
            rhand_big[:, :2] *= sf
            _render_hand(
                hand_canvas_big, lhand_big, _LEFT_HAND_FINGER_COLOURS,
                conf_thresh, limb_thickness=3, joint_radius=5,
            )
            _render_hand(
                hand_canvas_big, rhand_big, _RIGHT_HAND_FINGER_COLOURS,
                conf_thresh, limb_thickness=3, joint_radius=5,
            )
            hand_canvas = cv2.resize(
                hand_canvas_big, (W, H), interpolation=cv2.INTER_AREA
            )
        else:
            hand_canvas = np.zeros((H, W, 3), dtype=np.uint8)
            _render_hand(hand_canvas, lhand_kpts, _LEFT_HAND_FINGER_COLOURS, conf_thresh)
            _render_hand(hand_canvas, rhand_kpts, _RIGHT_HAND_FINGER_COLOURS, conf_thresh)

        return body_canvas, hand_canvas

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def render_debug_grid(
        self,
        json_dir: str | Path,
        out_path: str | Path,
        n_frames: int = 9,
    ) -> None:
        """Save a 3×3 grid of body+hand maps for visual inspection."""
        json_dir = Path(json_dir)
        json_files = sorted(json_dir.glob("*_keypoints.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files in {json_dir}")

        indices = np.linspace(0, len(json_files) - 1, n_frames, dtype=int)
        rows = []
        for i in range(0, n_frames, 3):
            row_imgs = []
            for idx in indices[i:i + 3]:
                body, hand = self._render_frame(json_files[idx])
                # Side-by-side: body | hand
                combined = np.concatenate([body, hand], axis=1)
                row_imgs.append(combined)
            rows.append(np.concatenate(row_imgs, axis=0))
        grid = np.concatenate(rows, axis=1)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), grid)
        print(f"[PoseMapRenderer] debug grid saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print("Usage: python -m avatar.conditioning.pose_to_controlnet <json_dir> [out_dir]")
        raise SystemExit(1)

    json_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/avatar/debug")

    cfg = PoseMapConfig()
    renderer = PoseMapRenderer(cfg)

    print(f"Rendering {json_dir} → {out_dir}")
    t0 = time.perf_counter()
    body_paths, hand_paths = renderer.render_and_save(
        json_dir,
        out_dir / "body",
        out_dir / "hand",
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"Done: {len(body_paths)} body maps + {len(hand_paths)} hand maps in {elapsed:.2f}s")
    print(f"  ({elapsed / max(len(body_paths), 1) * 1000:.1f} ms/frame)")

    # Debug grid
    renderer.render_debug_grid(json_dir, out_dir / "debug_grid.png")
