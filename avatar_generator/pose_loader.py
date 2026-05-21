"""Load pose sequences from .skels files or OpenPose JSON directories.

Both formats are normalised to the same internal representation:
    list of FramePose — one entry per video frame

FramePose holds the raw keypoint arrays in OpenPose v1.3 layout so the
downstream pose map renderer can treat both sources identically.

.skels format (SignLLM):
    Space-separated floats, one frame per line.
    Each line has skels_joints * 3 values (x, y, z normalised to [-1, 1]).
    We project x, y onto pixel space and set confidence = 1.0 for all joints.
    The 50-joint layout used by this project:
        joints  0– 7 : body (8 joints, COCO subset)
        joints  8–28 : left hand (21 joints, full OpenPose hand model)
        joints 29–49 : right hand (21 joints)

OpenPose JSON directory:
    Directory of *_keypoints.json files (CMU OpenPose v1.3 schema).
    Files are sorted lexicographically — the naming convention
    XXXXXX_keypoints.json guarantees correct temporal order.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from avatar_generator.config import GeneratorConfig


@dataclass
class FramePose:
    """Keypoints for one video frame in OpenPose v1.3 layout.

    All arrays are (N, 3) float32: [x_pixel, y_pixel, confidence].
    Missing keypoints have confidence = 0.0 and x=y=0.
    """
    body: np.ndarray    # (25, 3) BODY_25 — or (18, 3) COCO-18
    left_hand: np.ndarray   # (21, 3) full hand model
    right_hand: np.ndarray  # (21, 3) full hand model
    face: np.ndarray = field(default_factory=lambda: np.zeros((5, 3), dtype=np.float32))
    frame_index: int = 0


class PoseLoader:
    """Loads pose sequences from .skels or OpenPose JSON into FramePose lists."""

    def __init__(self, cfg: Optional[GeneratorConfig] = None) -> None:
        self.cfg = cfg or GeneratorConfig()

    def load(self, source: str | Path) -> list[FramePose]:
        """Auto-detect source type and load.

        Parameters
        ----------
        source : str | Path
            Path to a .skels file  OR  a directory of *_keypoints.json files.

        Returns
        -------
        list[FramePose]  — one entry per frame, in temporal order.
        """
        source = Path(source)
        if source.is_dir():
            return self._load_json_dir(source)
        elif source.suffix == ".skels":
            return self._load_skels(source)
        else:
            raise ValueError(
                f"Unsupported pose source: {source}\n"
                "Expected: a directory of *_keypoints.json files, or a .skels file."
            )

    # ------------------------------------------------------------------
    # OpenPose JSON loader
    # ------------------------------------------------------------------

    def _load_json_dir(self, json_dir: Path) -> list[FramePose]:
        json_files = sorted(json_dir.glob("*_keypoints.json"))
        if not json_files:
            raise FileNotFoundError(f"No *_keypoints.json files in {json_dir}")

        frames: list[FramePose] = []
        for i, jf in enumerate(json_files):
            frames.append(self._parse_json_frame(jf, i))
        return frames

    def _parse_json_frame(self, json_path: Path, frame_index: int) -> FramePose:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        people = data.get("people", [{}])
        person = people[0] if people else {}

        body = self._parse_flat(person.get("pose_keypoints_2d", []), 25)
        lhand = self._parse_flat(person.get("hand_left_keypoints_2d", []), 21)
        rhand = self._parse_flat(person.get("hand_right_keypoints_2d", []), 21)
        face = self._parse_flat(person.get("face_keypoints_2d", []), 5)

        return FramePose(
            body=body,
            left_hand=lhand,
            right_hand=rhand,
            face=face,
            frame_index=frame_index,
        )

    @staticmethod
    def _parse_flat(flat: list[float], n_kpts: int) -> np.ndarray:
        """Parse flat [x, y, conf, ...] list into (n_kpts, 3) float32."""
        arr = np.array(flat, dtype=np.float32).reshape(-1, 3) if flat else np.zeros((0, 3), dtype=np.float32)
        if arr.shape[0] < n_kpts:
            pad = np.zeros((n_kpts - arr.shape[0], 3), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        return arr[:n_kpts]

    # ------------------------------------------------------------------
    # .skels loader
    # ------------------------------------------------------------------

    def _load_skels(self, skels_path: Path) -> list[FramePose]:
        """Parse a .skels file into FramePose list.

        .skels layout (per line):
            <time_marker> <x0> <y0> <z0> <x1> <y1> <z1> ... (50 joints × 3)
        The time marker is the first token; remaining tokens are joint coords.
        Coords are normalised to approximately [-1, 1].
        """
        cfg = self.cfg
        W, H = cfg.skels_canvas_width, cfg.skels_canvas_height
        n_joints = cfg.skels_joints  # 50

        frames: list[FramePose] = []
        with open(skels_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                # First token is the time marker (float); skip it.
                coords = [float(t) for t in tokens[1:]]
                if len(coords) < n_joints * 3:
                    # Pad with zeros if line is shorter than expected.
                    coords += [0.0] * (n_joints * 3 - len(coords))
                coords = coords[:n_joints * 3]
                pose_3d = np.array(coords, dtype=np.float32).reshape(n_joints, 3)

                # Project to pixel space.
                # Normalised coords are in [-1, 1]; map to [0, W] / [0, H].
                # z is discarded (2D projection).
                px = (pose_3d[:, 0] + 1.0) * 0.5 * W
                py = (1.0 - (pose_3d[:, 1] + 1.0) * 0.5) * H  # y-flip
                conf = np.ones(n_joints, dtype=np.float32)

                # Zero-confidence for joints that are exactly zero (missing).
                missing = (pose_3d[:, 0] == 0.0) & (pose_3d[:, 1] == 0.0)
                conf[missing] = 0.0

                kpts = np.stack([px, py, conf], axis=1)  # (50, 3)

                # Split into body / left hand / right hand per project layout:
                #   joints  0– 7 : body (8 joints)
                #   joints  8–28 : left hand (21 joints)
                #   joints 29–49 : right hand (21 joints)
                body_8 = kpts[0:8]
                # Pad body to 25 joints (BODY_25) with zeros for missing joints.
                body_25 = np.zeros((25, 3), dtype=np.float32)
                body_25[:8] = body_8

                lhand_21 = kpts[8:29]
                rhand_21 = kpts[29:50]

                frames.append(FramePose(
                    body=body_25,
                    left_hand=lhand_21,
                    right_hand=rhand_21,
                    frame_index=line_no,
                ))

        if not frames:
            raise ValueError(f"No frames parsed from {skels_path}")
        return frames

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def frame_count(source: str | Path) -> int:
        """Return the number of frames without loading all data."""
        source = Path(source)
        if source.is_dir():
            return len(list(source.glob("*_keypoints.json")))
        elif source.suffix == ".skels":
            with open(source, encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        return 0
