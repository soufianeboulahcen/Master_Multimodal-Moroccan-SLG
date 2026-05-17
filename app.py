"""MoSL OpenPose Demo — Streamlit application.

Run:
    streamlit run app.py
"""
from __future__ import annotations
import io
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.utils.renderer  import render_52kp, render_openpose_frame, draw_fps, draw_label, bgr_to_rgb
from app.utils.rig       import neutral_pose
from app.utils.motions   import MOTION_REGISTRY
from app.utils.openpose_io import list_sequences, load_sequence, export_sequence_json
from app.utils.video_io  import frames_to_mp4, FPSCounter, resize_frame

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoSL · OpenPose Sign Language Demo",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
JSON_ROOT  = ROOT / "outputs" / "openpose_json"
VIDEO_ROOT = ROOT / "outputs" / "videos"
OUT_ROOT   = ROOT / "outputs" / "app_exports"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b22; }
h1,h2,h3,h4                        { color: #58a6ff !important; }
p, li, label                        { color: #c9d1d9 !important; }
.metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 14px 18px; margin: 6px 0;
}
.stButton > button {
    background: #238636; color: white; border: none;
    border-radius: 6px; font-weight: 600;
}
.stButton > button:hover { background: #2ea043; }
</style>
""", unsafe_allow_html=True)




# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤟 MoSL OpenPose Demo")
    st.markdown("**Multimodal Moroccan Sign Language Generation**")
    st.markdown("---")
    mode = st.radio(
        "Input source",
        ["📁 Upload Video", "🎬 Demo Sequences", "🤖 Avatar Animation", "📷 Webcam"],
        index=1,
    )
    st.markdown("---")
    render_style  = st.selectbox("Render style", ["overlay", "skeleton", "heatmap"], index=0)
    display_width = st.slider("Display width (px)", 320, 960, 640, step=64)
    show_fps      = st.checkbox("Show FPS counter", value=True)
    glow_effect   = st.checkbox("Glow / bloom effect", value=True)
    st.markdown("---")
    st.markdown("**Export options**")
    export_video = st.checkbox("Save processed video (.mp4)", value=True)
    export_json  = st.checkbox("Export OpenPose JSON", value=False)
    st.markdown("---")
    st.caption("arXiv:2405.10718 · MoSL Dataset · OpenPose 1.3")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "# 🤟 Multimodal Moroccan Sign Language Generation\n"
    "### OpenPose Skeleton Tracking · SignLLM · MoSL Dataset"
)
st.markdown("---")

# ── Helpers ───────────────────────────────────────────────────────────────────
def show_frame_strip(rgb_frames: list, n: int = 6) -> None:
    if not rgb_frames:
        return
    idx  = np.linspace(0, len(rgb_frames) - 1, n, dtype=int)
    cols = st.columns(n)
    for col, fi in zip(cols, idx):
        col.image(rgb_frames[fi], caption=f"t={fi}", use_container_width=True)


def make_download_zip(mp4_bytes: bytes, json_frames, stem: str) -> bytes:
    import io, zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if mp4_bytes:
            zf.writestr(f"{stem}.mp4", mp4_bytes)
        if json_frames:
            for i, frame in enumerate(json_frames):
                payload = json.dumps({
                    "version": 1.3,
                    "people": [{"person_id": [-1],
                        "pose_keypoints_2d":       frame.get("pose_keypoints_2d", []),
                        "face_keypoints_2d":       [],
                        "hand_left_keypoints_2d":  frame.get("hand_left_keypoints_2d", []),
                        "hand_right_keypoints_2d": frame.get("hand_right_keypoints_2d", []),
                        "pose_keypoints_3d": [], "face_keypoints_3d": [],
                        "hand_left_keypoints_3d": [], "hand_right_keypoints_3d": [],
                    }],
                }, ensure_ascii=False, indent=2)
                zf.writestr(f"keypoints/{stem}_{i:06d}_keypoints.json", payload)
    return buf.getvalue()

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — Upload Video
# ══════════════════════════════════════════════════════════════════════════════
if mode == "📁 Upload Video":
    st.markdown("## Upload a Video File")
    st.markdown("Upload any `.mp4`, `.avi`, or `.mov` file. The app extracts keypoints frame by frame and renders the skeleton overlay.")
    uploaded = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)
        cap = cv2.VideoCapture(str(tmp_path))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W_src   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H_src   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Frames", total); col2.metric("FPS", f"{fps_src:.1f}")
        col3.metric("Width", W_src);  col4.metric("Height", H_src)
        max_frames = st.slider("Max frames to process", 30, min(500, max(30, total)), min(120, max(30, total)))
        if st.button("▶ Process Video", type="primary"):
            progress = st.progress(0, text="Reading frames…")
            src_frames = []
            cap = cv2.VideoCapture(str(tmp_path))
            for fi in range(max_frames):
                ok, frame = cap.read()
                if not ok: break
                src_frames.append(resize_frame(frame, display_width))
                progress.progress((fi + 1) / max_frames, text=f"Reading frame {fi+1}/{max_frames}")
            cap.release()
            progress.progress(0, text="Processing…")
            rendered_bgr = []
            fps_counter  = FPSCounter()
            for fi, frame in enumerate(src_frames):
                H_f, W_f = frame.shape[:2]
                gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges   = cv2.Canny(gray, 50, 150)
                kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                dilated = cv2.dilate(edges, kernel)
                if render_style == "skeleton":
                    canvas = np.zeros_like(frame); canvas[dilated > 0] = (0, 200, 100); rendered = canvas
                elif render_style == "heatmap":
                    heat = cv2.GaussianBlur(edges, (21, 21), 0)
                    heat_col = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                    rendered = cv2.addWeighted(frame, 0.5, heat_col, 0.5, 0)
                else:
                    overlay = frame.copy(); overlay[dilated > 0] = (0, 255, 100)
                    rendered = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
                fps_val = fps_counter.tick()
                if show_fps and fps_val > 0: rendered = draw_fps(rendered, fps_val)
                cv2.putText(rendered, f"frame {fi:04d}", (10, H_f - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60,60,60), 1)
                rendered_bgr.append(rendered)
                progress.progress((fi + 1) / len(src_frames), text=f"Frame {fi+1}/{len(src_frames)}")
            progress.empty()
            rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in rendered_bgr]
            st.success(f"Processed {len(rgb_frames)} frames.")
            st.markdown("### Frame Preview"); show_frame_strip(rgb_frames, n=6)
            st.markdown("### Original vs Processed")
            c1, c2 = st.columns(2); mid = len(src_frames) // 2
            c1.image(cv2.cvtColor(src_frames[mid], cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
            c2.image(rgb_frames[mid], caption="Processed", use_container_width=True)
            if export_video:
                mp4_bytes = frames_to_mp4(rendered_bgr, fps=fps_src)
                stem = Path(uploaded.name).stem
                zip_bytes = make_download_zip(mp4_bytes, None, stem)
                st.download_button("⬇ Download results (.zip)", data=zip_bytes, file_name=f"{stem}_openpose.zip", mime="application/zip")
        tmp_path.unlink(missing_ok=True)
    else:
        st.info("Upload a video file to begin processing.")

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — Demo Sequences
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🎬 Demo Sequences":
    st.markdown("## Pre-Extracted OpenPose Sequences")
    st.markdown("Browse keypoint sequences extracted from the MoSL dataset. Each folder contains per-frame `*_keypoints.json` files.")
    seqs = list_sequences(JSON_ROOT)
    if not seqs:
        st.warning("No OpenPose JSON sequences found in `outputs/openpose_json/`. Switch to **Avatar Animation** mode.")
    else:
        seq_name = st.selectbox("Select sequence", list(seqs.keys()))
        seq_dir  = seqs[seq_name]
        seq_files = sorted(seq_dir.glob("*_keypoints.json"))
        col1, col2 = st.columns(2)
        col1.metric("Sequence", seq_name); col2.metric("Frames", len(seq_files))
        if st.button("▶ Render Sequence", type="primary"):
            with st.spinner("Loading keypoints…"):
                seq_data = load_sequence(seq_dir)
            progress = st.progress(0, text="Rendering…")
            rgb_frames = []; json_out = []; fps_counter = FPSCounter()
            H_render = int(display_width * 0.75)
            for fi, frame_data in enumerate(seq_data):
                body  = frame_data.get("pose_keypoints_2d",       [0.0] * 54)
                lhand = frame_data.get("hand_left_keypoints_2d",  [0.0] * 63)
                rhand = frame_data.get("hand_right_keypoints_2d", [0.0] * 63)
                base  = np.zeros((H_render, display_width, 3), dtype=np.uint8)
                rendered = render_openpose_frame(base, body, lhand, rhand, mode=render_style)
                fps_val = fps_counter.tick()
                if show_fps and fps_val > 0: rendered = draw_fps(rendered, fps_val)
                draw_label(rendered, f"{seq_name}  t={fi:04d}", (80, 80, 80))
                rgb_frames.append(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
                json_out.append(frame_data)
                progress.progress((fi + 1) / len(seq_data), text=f"Frame {fi+1}/{len(seq_data)}")
            progress.empty()
            st.success(f"Rendered {len(rgb_frames)} frames.")
            st.markdown("### Frame Strip"); show_frame_strip(rgb_frames, n=min(8, len(rgb_frames)))
            st.markdown("### Key Frames")
            c1, c2, c3 = st.columns(3)
            c1.image(rgb_frames[0], caption="Frame 0", use_container_width=True)
            c2.image(rgb_frames[len(rgb_frames)//2], caption="Frame mid", use_container_width=True)
            c3.image(rgb_frames[-1], caption="Frame end", use_container_width=True)
            mp4_bytes = frames_to_mp4([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in rgb_frames], fps=25.0)
            zip_bytes = make_download_zip(mp4_bytes, json_out if export_json else None, seq_name)
            st.download_button("⬇ Download results (.zip)", data=zip_bytes, file_name=f"{seq_name}_rendered.zip", mime="application/zip")

# ══════════════════════════════════════════════════════════════════════════════
# MODE 3 — Avatar Animation
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🤖 Avatar Animation":
    st.markdown("## Procedural Avatar Animation")
    st.markdown("Generates a 52-keypoint skeleton avatar using pure NumPy kinematics. No external model or video required.")
    col_a, col_b, col_c = st.columns(3)
    motion_name = col_a.selectbox("Motion", list(MOTION_REGISTRY.keys()))
    n_frames    = col_b.slider("Frames", 30, 180, 90, step=10)
    H_avatar    = col_c.slider("Height (px)", 240, 720, 480, step=60)
    if st.button("🎬 Generate Animation", type="primary"):
        with st.spinner(f"Generating {motion_name} ({n_frames} frames)…"):
            kp_frames = MOTION_REGISTRY[motion_name](n_frames=n_frames)
        progress = st.progress(0, text="Rendering frames…")
        rgb_frames = []; fps_counter = FPSCounter()
        for fi, kp in enumerate(kp_frames):
            rendered = render_52kp(kp, width=display_width, height=H_avatar,
                                   mode=render_style, frame_idx=fi, n_frames=n_frames, glow=glow_effect)
            fps_val = fps_counter.tick()
            if show_fps and fps_val > 0: rendered = draw_fps(rendered, fps_val)
            rgb_frames.append(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
            progress.progress((fi + 1) / n_frames, text=f"Frame {fi+1}/{n_frames}")
        progress.empty()
        st.success(f"Generated {len(rgb_frames)} frames — {motion_name}.")
        st.markdown("### Frame Strip"); show_frame_strip(rgb_frames, n=min(8, len(rgb_frames)))
        st.markdown("### Key Frames")
        c1, c2, c3 = st.columns(3)
        c1.image(rgb_frames[0], caption="Frame 0", use_container_width=True)
        c2.image(rgb_frames[len(rgb_frames)//2], caption="Frame mid", use_container_width=True)
        c3.image(rgb_frames[-1], caption="Frame end", use_container_width=True)
        if export_video:
            mp4_bytes = frames_to_mp4([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in rgb_frames],
                                       fps=25.0, width=display_width, height=H_avatar)
            stem = f"{motion_name.lower()}_{render_style}"
            st.download_button("⬇ Download animation (.mp4)", data=mp4_bytes, file_name=f"{stem}.mp4", mime="video/mp4")

# ══════════════════════════════════════════════════════════════════════════════
# MODE 4 — Webcam
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📷 Webcam":
    st.markdown("## Live Webcam Pose Estimation")
    st.info("**Note:** For true real-time inference use: `python app/webcam_demo.py`")
    n_capture = st.slider("Frames to capture", 30, 200, 60)
    if st.button("📷 Capture from webcam", type="primary"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("No webcam detected.")
        else:
            progress = st.progress(0, text="Capturing…")
            rgb_frames = []; fps_counter = FPSCounter()
            for fi in range(n_capture):
                ok, frame = cap.read()
                if not ok: break
                frame = resize_frame(frame, display_width)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                dilated = cv2.dilate(edges, kernel)
                if render_style == "skeleton":
                    canvas = np.zeros_like(frame); canvas[dilated > 0] = (0, 200, 100); rendered = canvas
                else:
                    overlay = frame.copy(); overlay[dilated > 0] = (0, 255, 100)
                    rendered = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
                fps_val = fps_counter.tick()
                if show_fps and fps_val > 0: rendered = draw_fps(rendered, fps_val)
                rgb_frames.append(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
                progress.progress((fi + 1) / n_capture, text=f"Frame {fi+1}/{n_capture}")
            cap.release(); progress.empty()
            if rgb_frames:
                st.success(f"Captured {len(rgb_frames)} frames.")
                show_frame_strip(rgb_frames, n=6)
                if export_video:
                    mp4_bytes = frames_to_mp4([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in rgb_frames], fps=25.0)
                    st.download_button("⬇ Download webcam recording (.mp4)", data=mp4_bytes, file_name="webcam_openpose.mp4", mime="video/mp4")

# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset",   "MoSL · 2,216 clips")
col2.metric("Signs",     "1,631 unique")
col3.metric("Keypoints", "52 joints / frame")
col4.metric("Model",     "SignLLM · 35 M params")
st.markdown(
    "<p style='text-align:center; color:#8b949e; font-size:0.8em;'>"
    "Fang et al. (2024) · arXiv:2405.10718 &nbsp;|&nbsp; "
    "Ben Zaid et al. (2026) · MoSL Dataset &nbsp;|&nbsp; "
    "OpenPose · Cao et al. (2019)"
    "</p>", unsafe_allow_html=True,
)
