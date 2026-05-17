"""OpenPose JSON I/O utilities.

Reads per-frame keypoint JSON files produced by the project pipeline and
returns structured numpy arrays ready for rendering.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def load_frame(json_path: Path) -> dict:
    """Load one OpenPose JSON frame. Returns empty-person dict on failure."""
    empty = {
        "pose_keypoints_2d":       [0.0] * 54,
        "hand_left_keypoints_2d":  [0.0] * 63,
        "hand_right_keypoints_2d": [0.0] * 63,
    }
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        people = data.get("people", [])
        if not people:
            return empty
        p = people[0]
        return {
            "pose_keypoints_2d":       p.get("pose_keypoints_2d",       empty["pose_keypoints_2d"]),
            "hand_left_keypoints_2d":  p.get("hand_left_keypoints_2d",  empty["hand_left_keypoints_2d"]),
            "hand_right_keypoints_2d": p.get("hand_right_keypoints_2d", empty["hand_right_keypoints_2d"]),
        }
    except Exception:
        return empty


def load_sequence(seq_dir: Path) -> list[dict]:
    """Load all keypoint JSON files in a directory, sorted by frame index."""
    files = sorted(seq_dir.glob("*_keypoints.json"))
    return [load_frame(f) for f in files]


def list_sequences(json_root: Path) -> dict[str, Path]:
    """Return {name: path} for all sequence directories under json_root."""
    seqs: dict[str, Path] = {}
    if not json_root.exists():
        return seqs
    for d in sorted(json_root.iterdir()):
        if d.is_dir() and any(d.glob("*_keypoints.json")):
            seqs[d.name] = d
    return seqs


def export_sequence_json(
    frames: list[dict],
    out_dir: Path,
    stem: str = "output",
) -> list[Path]:
    """Write a list of frame dicts as per-frame OpenPose JSON files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, frame in enumerate(frames):
        payload = {
            "version": 1.3,
            "people": [{
                "person_id":               [-1],
                "pose_keypoints_2d":       frame.get("pose_keypoints_2d", []),
                "face_keypoints_2d":       [],
                "hand_left_keypoints_2d":  frame.get("hand_left_keypoints_2d", []),
                "hand_right_keypoints_2d": frame.get("hand_right_keypoints_2d", []),
                "pose_keypoints_3d":       [],
                "face_keypoints_3d":       [],
                "hand_left_keypoints_3d":  [],
                "hand_right_keypoints_3d": [],
            }],
        }
        p = out_dir / f"{stem}_{i:06d}_keypoints.json"
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        paths.append(p)
    return paths
