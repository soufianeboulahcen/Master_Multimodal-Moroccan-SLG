"""Video I/O helpers: read frames, write MP4, encode for Streamlit display."""
from __future__ import annotations
import io
import tempfile
import time
from pathlib import Path
from typing import Generator

import cv2
import numpy as np


def read_video_frames(
    source: str | Path | int,
    max_frames: int = 500,
) -> Generator[tuple[np.ndarray, float], None, None]:
    """Yield (bgr_frame, fps) for each frame in a video file or webcam.

    Args:
        source: file path, Path object, or 0 for default webcam
        max_frames: safety cap to avoid infinite webcam loops in batch mode
    """
    cap = cv2.VideoCapture(str(source) if not isinstance(source, int) else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    count = 0
    try:
        while count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame, fps
            count += 1
    finally:
        cap.release()


def frames_to_mp4(
    frames: list[np.ndarray],
    fps: float = 25.0,
    width: int | None = None,
    height: int | None = None,
) -> bytes:
    """Encode a list of BGR frames to an MP4 byte string (in-memory)."""
    if not frames:
        return b""
    h = height or frames[0].shape[0]
    w = width  or frames[0].shape[1]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
    writer.release()
    data = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return data


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resize_frame(frame: np.ndarray, width: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))


class FPSCounter:
    """Rolling-window FPS counter."""

    def __init__(self, window: int = 30) -> None:
        self._times: list[float] = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0
