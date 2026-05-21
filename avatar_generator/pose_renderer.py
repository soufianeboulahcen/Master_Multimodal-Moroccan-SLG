"""Render FramePose keypoints into RGB ControlNet conditioning images.

Produces one RGB image per frame that ControlNet-OpenPose uses as its
conditioning input.  The image shows the skeleton drawn on a black canvas
using the DWPose colour convention that ControlNet was trained on.

Key design decisions:
    - Hands are rendered at 2× resolution then downsampled (anti-aliasing).
    - Body and hand skeletons are drawn on the same canvas (single ControlNet
      branch) for simplicity.  The avatar/ module uses dual branches; here
      we use one branch for a simpler, faster pipeline.
    - Output is a PIL Image (RGB) ready to pass directly to the diffusion
      pipeline's `image` argument.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from avatar_generator.config import GeneratorConfig
from avatar_generator.pose_loader import FramePose


# ── DWPose colour tables ─────────────────────────────────────────────────────

# COCO-18 body limb pairs.
_BODY_LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
    (2, 8), (5, 11),
]

_BODY_LIMB_COLOURS = [
    (255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0),
    (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
    (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255),
    (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]

_BODY_KP_COLOURS = [
    (255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0),
    (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255),
]

# Full 21-point hand finger chains.
_HAND_FINGERS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]

_LEFT_HAND_COLOURS = [
    (255, 153, 0), (255, 204, 0), (255, 255, 0),
    (153, 255, 0), (0, 255, 0),
]
_RIGHT_HAND_COLOURS = [
    (0, 153, 255), (0, 204, 255), (0, 255, 255),
    (0, 255, 153), (0, 255, 0),
]


# ── Drawing primitives ───────────────────────────────────────────────────────

def _draw_limb(canvas, kpts, a, b, colour, thickness, thresh):
    if kpts[a, 2] < thresh or kpts[b, 2] < thresh:
        return
    xa, ya = int(round(kpts[a, 0])), int(round(kpts[a, 1]))
    xb, yb = int(round(kpts[b, 0])), int(round(kpts[b, 1]))
    cv2.line(canvas, (xa, ya), (xb, yb), colour, thickness, cv2.LINE_AA)


def _draw_joint(canvas, kpts, idx, colour, radius, thresh):
    if kpts[idx, 2] < thresh:
        return
    x, y = int(round(kpts[idx, 0])), int(round(kpts[idx, 1]))
    cv2.circle(canvas, (x, y), radius, colour, -1, cv2.LINE_AA)


def _scale(kpts: np.ndarray, src_w, src_h, dst_w, dst_h) -> np.ndarray:
    out = kpts.copy()
    out[:, 0] = out[:, 0] * dst_w / src_w
    out[:, 1] = out[:, 1] * dst_h / src_h
    return out


# ── Main renderer ────────────────────────────────────────────────────────────

class PoseMapRenderer:
    """Renders FramePose objects into RGB PIL Images for ControlNet."""

    def __init__(self, cfg: Optional[GeneratorConfig] = None) -> None:
        self.cfg = cfg or GeneratorConfig()

    def render(self, frame: FramePose) -> "PIL.Image.Image":
        """Render one FramePose → RGB PIL Image (W×H)."""
        from PIL import Image

        cfg = self.cfg
        W, H = cfg.width, cfg.height
        src_W, src_H = cfg.pose_source_width, cfg.pose_source_height
        thresh = cfg.pose_confidence_threshold

        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # Body: take first 18 joints (COCO-18 subset of BODY_25).
        body = _scale(frame.body[:18], src_W, src_H, W, H)
        for i, (a, b) in enumerate(_BODY_LIMBS):
            if a < len(body) and b < len(body):
                _draw_limb(canvas, body, a, b, _BODY_LIMB_COLOURS[i % len(_BODY_LIMB_COLOURS)], 3, thresh)
        for i in range(min(18, len(body))):
            _draw_joint(canvas, body, i, _BODY_KP_COLOURS[i % len(_BODY_KP_COLOURS)], 4, thresh)

        # Hands.
        if cfg.hand_supersample:
            sf = 2
            big_W, big_H = W * sf, H * sf
            hand_canvas = np.zeros((big_H, big_W, 3), dtype=np.uint8)
            lhand = _scale(frame.left_hand, src_W, src_H, big_W, big_H)
            rhand = _scale(frame.right_hand, src_W, src_H, big_W, big_H)
            self._draw_hand(hand_canvas, lhand, _LEFT_HAND_COLOURS, thresh, 3, 5)
            self._draw_hand(hand_canvas, rhand, _RIGHT_HAND_COLOURS, thresh, 3, 5)
            hand_small = cv2.resize(hand_canvas, (W, H), interpolation=cv2.INTER_AREA)
            # Blend hands over body canvas.
            mask = hand_small.any(axis=2, keepdims=True)
            canvas = np.where(mask, hand_small, canvas)
        else:
            lhand = _scale(frame.left_hand, src_W, src_H, W, H)
            rhand = _scale(frame.right_hand, src_W, src_H, W, H)
            self._draw_hand(canvas, lhand, _LEFT_HAND_COLOURS, thresh)
            self._draw_hand(canvas, rhand, _RIGHT_HAND_COLOURS, thresh)

        # Convert BGR → RGB for PIL.
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def render_sequence(self, frames: list[FramePose]) -> list["PIL.Image.Image"]:
        """Render all frames.  Returns list of PIL Images."""
        return [self.render(f) for f in frames]

    @staticmethod
    def _draw_hand(canvas, kpts, finger_colours, thresh, thickness=2, radius=3):
        for f_idx, finger in enumerate(_HAND_FINGERS):
            colour = finger_colours[f_idx % len(finger_colours)]
            for a, b in zip(finger, finger[1:]):
                if a < len(kpts) and b < len(kpts):
                    _draw_limb(canvas, kpts, a, b, colour, thickness, thresh)
            for j in finger:
                if j < len(kpts):
                    _draw_joint(canvas, kpts, j, colour, radius, thresh)
