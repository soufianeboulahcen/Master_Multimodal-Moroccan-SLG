# MoSL OpenPose Demo — Application

A Streamlit web application that demonstrates the OpenPose skeleton tracking
and sign language generation pipeline from the
**Multimodal Moroccan Sign Language Generation** research project.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r app_requirements.txt

# 2. Launch the app
streamlit run app.py
# or
./run_app.sh
```

Open **http://localhost:8501** in your browser.

---

## Application Modes

| Mode | Description |
|------|-------------|
| **Upload Video** | Upload any `.mp4`/`.avi`/`.mov` file; extract and render skeleton overlay |
| **Demo Sequences** | Browse pre-extracted OpenPose JSON sequences from the MoSL dataset |
| **Avatar Animation** | Generate procedural 52-keypoint avatar animations (no model required) |
| **Webcam** | Capture live frames from a connected webcam |

---

## Render Styles

| Style | Description |
|-------|-------------|
| `overlay` | Skeleton drawn over the original video frame |
| `skeleton` | Skeleton only on black background with glow/bloom |
| `heatmap` | Joint confidence visualised as a thermal heatmap |

---

## Project Structure

```
app.py                    ← Streamlit entry point
run_app.sh                ← One-command launcher
app_requirements.txt      ← Application dependencies
app/
  __init__.py
  webcam_demo.py          ← Standalone real-time OpenCV webcam demo
  utils/
    __init__.py
    rig.py                ← Joint indices, limb connectivity, colours
    renderer.py           ← Frame rendering (skeleton, overlay, heatmap)
    motions.py            ← Procedural motion generators (walk, wave, sign…)
    openpose_io.py        ← OpenPose JSON read/write utilities
    video_io.py           ← Video I/O, MP4 encoding, FPS counter
outputs/
  openpose_json/          ← Pre-extracted keypoint sequences
  videos/                 ← Rendered demo videos
  app_exports/            ← Files saved by the application
```

---

## Standalone Webcam Demo (OpenCV window)

For true real-time inference without a browser:

```bash
python app/webcam_demo.py --style overlay --width 640

# Controls
#   q / Esc  — quit
#   s        — cycle render style (overlay → skeleton → heatmap)
#   r        — start / stop recording (saves to outputs/app_exports/)
```

---

## Export Options

- **Download processed video** — MP4 at source FPS
- **Export OpenPose JSON** — per-frame `*_keypoints.json` files bundled in a ZIP
- Webcam recordings saved automatically to `outputs/app_exports/`

---

## References

- Fang et al. (2024). *SignLLM: Sign Languages Production Large Language Models*. arXiv:2405.10718
- Ben Zaid et al. (2026). *Moroccan Sign Language Video Dataset (MoSL)*. Mendeley Data
- Cao et al. (2019). *OpenPose: Realtime Multi-Person 2D Pose Estimation*. IEEE TPAMI
