"""Standalone real-time webcam demo using an OpenCV window.

Run:
    python app/webcam_demo.py [--style overlay|skeleton|heatmap] [--width 640]

Press  q  or  Esc  to quit.
Press  s  to cycle render styles.
Press  r  to start/stop recording.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.utils.renderer import draw_fps, draw_label
from app.utils.video_io import FPSCounter, frames_to_mp4, resize_frame

STYLES = ["overlay", "skeleton", "heatmap"]


def process_frame(frame: np.ndarray, style: str) -> np.ndarray:
    """Apply edge-based skeleton detection to one BGR frame."""
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges   = cv2.Canny(gray, 50, 150)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, kernel)

    if style == "skeleton":
        canvas = np.zeros_like(frame)
        canvas[dilated > 0] = (0, 200, 100)
        return canvas
    elif style == "heatmap":
        heat     = cv2.GaussianBlur(edges, (21, 21), 0)
        heat_col = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.5, heat_col, 0.5, 0)
    else:  # overlay
        overlay = frame.copy()
        overlay[dilated > 0] = (0, 255, 100)
        return cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time webcam OpenPose demo")
    parser.add_argument("--style", default="overlay", choices=STYLES)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        sys.exit(1)

    style_idx   = STYLES.index(args.style)
    fps_counter = FPSCounter(window=30)
    recording   = False
    rec_frames: list[np.ndarray] = []
    out_dir     = ROOT / "outputs" / "app_exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Controls:  q/Esc = quit  |  s = cycle style  |  r = record/stop")
    print(f"Style: {STYLES[style_idx]}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = resize_frame(frame, args.width)
        rendered = process_frame(frame, STYLES[style_idx])
        fps = fps_counter.tick()
        rendered = draw_fps(rendered, fps)

        style_label = f"Style: {STYLES[style_idx]}  |  {'● REC' if recording else ''}"
        cv2.putText(rendered, style_label, (10, rendered.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255) if recording else (80, 80, 80), 1)

        cv2.imshow("MoSL OpenPose Demo  [q=quit  s=style  r=record]", rendered)

        if recording:
            rec_frames.append(rendered.copy())

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            style_idx = (style_idx + 1) % len(STYLES)
            print(f"Style: {STYLES[style_idx]}")
        elif key == ord("r"):
            if not recording:
                recording = True
                rec_frames = []
                print("Recording started…")
            else:
                recording = False
                if rec_frames:
                    ts   = int(time.time())
                    path = out_dir / f"webcam_{ts}.mp4"
                    data = frames_to_mp4(rec_frames, fps=25.0)
                    path.write_bytes(data)
                    print(f"Saved {len(rec_frames)} frames → {path}")
                rec_frames = []

    cap.release()
    cv2.destroyAllWindows()

    if rec_frames:
        ts   = int(time.time())
        path = out_dir / f"webcam_{ts}.mp4"
        data = frames_to_mp4(rec_frames, fps=25.0)
        path.write_bytes(data)
        print(f"Saved {len(rec_frames)} frames → {path}")


if __name__ == "__main__":
    main()
