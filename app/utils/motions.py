"""Procedural motion generators for the demo avatar.

Each generator returns list[np.ndarray] of (52, 2) normalised keypoints.
No external models required — pure NumPy kinematics.
"""
from __future__ import annotations
import math
import numpy as np
from app.utils.rig import neutral_pose, N_KP


def _sin(t: float, freq: float = 1.0, phase: float = 0.0, amp: float = 1.0) -> float:
    return amp * math.sin(2 * math.pi * freq * t + phase)


def _attach_hands(kp: np.ndarray, s: float = 0.28) -> np.ndarray:
    """Recompute hand keypoints relative to wrists after body animation."""
    lw = kp[7].copy()
    offsets_l = [
        [0, 0], [-s*.04, s*.06], [-s*.04, s*.11],
        [-s*.01, s*.07], [-s*.01, s*.12],
        [ s*.02, s*.06], [ s*.02, s*.11],
        [ s*.05, s*.05], [ s*.05, s*.09],
        [-s*.07, s*.03], [-s*.10, s*.06],
    ]
    for i, off in enumerate(offsets_l):
        kp[30 + i] = lw + np.array(off)
    rw = kp[4].copy()
    offsets_r = [
        [0, 0], [ s*.04, s*.06], [ s*.04, s*.11],
        [ s*.01, s*.07], [ s*.01, s*.12],
        [-s*.02, s*.06], [-s*.02, s*.11],
        [-s*.05, s*.05], [-s*.05, s*.09],
        [ s*.07, s*.03], [ s*.10, s*.06],
    ]
    for i, off in enumerate(offsets_r):
        kp[41 + i] = rw + np.array(off)
    return kp


def generate_walking(n_frames: int = 90) -> list[np.ndarray]:
    cx, cy, s = 0.5, 0.52, 0.28
    frames = []
    for i in range(n_frames):
        t  = i / n_frames
        ph = 2 * math.pi * t
        kp = neutral_pose(cx, cy, s).copy()
        bob = _sin(t, freq=2, amp=s * 0.015)
        for j in [0,1,2,3,4,5,6,7,8,15,16,17,18,25,26,27,28,29]:
            kp[j, 1] += bob
        # Arms swing
        kp[3, 0] += _sin(t, freq=1, phase=0,    amp=s * 0.12)
        kp[4, 0] += _sin(t, freq=1, phase=0,    amp=s * 0.14)
        kp[6, 0] += _sin(t, freq=1, phase=math.pi, amp=s * 0.12)
        kp[7, 0] += _sin(t, freq=1, phase=math.pi, amp=s * 0.14)
        # Legs
        kp[10, 0] += _sin(t, freq=1, phase=0,    amp=s * 0.06)
        kp[11, 0] += _sin(t, freq=1, phase=0,    amp=s * 0.08)
        kp[13, 0] += _sin(t, freq=1, phase=math.pi, amp=s * 0.06)
        kp[14, 0] += _sin(t, freq=1, phase=math.pi, amp=s * 0.08)
        frames.append(_attach_hands(np.clip(kp, 0, 1)))
    return frames


def generate_waving(n_frames: int = 90) -> list[np.ndarray]:
    cx, cy, s = 0.5, 0.52, 0.28
    frames = []
    for i in range(n_frames):
        t  = i / n_frames
        kp = neutral_pose(cx, cy, s).copy()
        # Right arm waves
        kp[2, 1] -= s * 0.05
        kp[3, 0] += s * 0.20
        kp[3, 1] -= s * 0.15 + _sin(t, freq=2, amp=s * 0.05)
        kp[4, 0] += s * 0.22
        kp[4, 1] -= s * 0.30 + _sin(t, freq=2, amp=s * 0.08)
        frames.append(_attach_hands(np.clip(kp, 0, 1)))
    return frames


def generate_signing(n_frames: int = 90) -> list[np.ndarray]:
    """Both hands move in front of the body — generic signing motion."""
    cx, cy, s = 0.5, 0.52, 0.28
    frames = []
    for i in range(n_frames):
        t  = i / n_frames
        kp = neutral_pose(cx, cy, s).copy()
        # Left hand traces a figure-8
        kp[7, 0] = cx - s*0.15 + _sin(t, freq=1, amp=s * 0.12)
        kp[7, 1] = cy - s*0.10 + _sin(t, freq=2, amp=s * 0.08)
        # Right hand mirrors
        kp[4, 0] = cx + s*0.15 + _sin(t, freq=1, phase=math.pi, amp=s * 0.12)
        kp[4, 1] = cy - s*0.10 + _sin(t, freq=2, phase=math.pi/2, amp=s * 0.08)
        frames.append(_attach_hands(np.clip(kp, 0, 1)))
    return frames


def generate_jumping(n_frames: int = 60) -> list[np.ndarray]:
    cx, cy, s = 0.5, 0.52, 0.28
    frames = []
    for i in range(n_frames):
        t  = i / n_frames
        kp = neutral_pose(cx, cy, s).copy()
        jump = max(0.0, -_sin(t, freq=1, amp=s * 0.18))
        kp[:, 1] -= jump
        # Arms up during jump
        arm_raise = max(0.0, -_sin(t, freq=1, amp=s * 0.15))
        kp[3, 1] -= arm_raise
        kp[4, 1] -= arm_raise * 1.2
        kp[6, 1] -= arm_raise
        kp[7, 1] -= arm_raise * 1.2
        frames.append(_attach_hands(np.clip(kp, 0, 1)))
    return frames


def generate_dancing(n_frames: int = 120) -> list[np.ndarray]:
    cx, cy, s = 0.5, 0.52, 0.28
    frames = []
    for i in range(n_frames):
        t  = i / n_frames
        kp = neutral_pose(cx, cy, s).copy()
        sway = _sin(t, freq=1, amp=s * 0.04)
        kp[:, 0] += sway
        kp[3, 1] += _sin(t, freq=2, amp=s * 0.10)
        kp[4, 1] += _sin(t, freq=2, phase=0.3, amp=s * 0.14)
        kp[6, 1] += _sin(t, freq=2, phase=math.pi, amp=s * 0.10)
        kp[7, 1] += _sin(t, freq=2, phase=math.pi + 0.3, amp=s * 0.14)
        kp[10, 0] += _sin(t, freq=2, amp=s * 0.04)
        kp[13, 0] += _sin(t, freq=2, phase=math.pi, amp=s * 0.04)
        frames.append(_attach_hands(np.clip(kp, 0, 1)))
    return frames


MOTION_REGISTRY: dict[str, callable] = {
    "Walking":  generate_walking,
    "Waving":   generate_waving,
    "Signing":  generate_signing,
    "Jumping":  generate_jumping,
    "Dancing":  generate_dancing,
}
