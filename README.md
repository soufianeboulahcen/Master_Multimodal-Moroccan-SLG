# Multimodal Moroccan Sign Language Generation

A paper-faithful PyTorch reimplementation of **SignLLM** ([Fang et al., 2024](https://arxiv.org/abs/2405.10718)) trained on the **Moroccan Sign Language (MoSL)** video dataset, with a full OpenPose-style pose tracking and video generation pipeline.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Generated Outputs](#generated-outputs)
- [Dataset](#dataset)
- [Git LFS](#git-lfs)
- [Citation](#citation)

---

## Overview

This project implements **Sign Language Production (SLP)** for Moroccan Sign Language (MoSL), used by ~300,000 people in Morocco.

**Three components:**

| Component | Description |
|---|---|
| **SignLLM** | Encoder-decoder transformer (2+2 layers, d_model=768, ~35M params) mapping Arabic text → 3D pose sequence |
| **Pose extraction** | Per-frame 2D keypoints via pytorch-openpose → 3D lifting via Prompt2Sign |
| **OpenPose demo pipeline** | 7-variant video generation: overlay, skeleton, neon, heatmap, slow-motion, studio, mosaic |

---

## Results

Autoregressive DTW on the MoSL test set (lower is better):

| Method | Dev DTW | Test DTW |
|---|---:|---:|
| Nearest-Neighbor (baseline) | 0.865 | **0.782** |
| Mean-Pose (baseline) | **0.815** | 0.868 |
| Random-Clip (baseline) | 0.974 | 0.962 |
| **SignLLM MSE (best model)** | 1.022 | 1.045 |
| SignLLM RL+PLC | 1.153 | 1.212 |
| SignLLM RL | 1.307 | 1.318 |

> All three SignLLM configurations fail to outperform deterministic baselines on MoSL. Root cause: ~18× smaller dataset than the paper's ASL setting + isolated-word structure. See [`docs/RESULTS.md`](docs/RESULTS.md).

---

## Repository Structure

```
.
├── mosl/                        # Core Python package
│   ├── data/                    # Dataset loading, splits, label extraction
│   ├── model/                   # SignLLM architecture (signllm.py, positional.py)
│   ├── pose/                    # Pose extraction pipeline
│   ├── text/                    # Word-level Arabic tokenizer (NFC)
│   └── train/                   # Training loop, losses, eval, scheduler
│
├── scripts/                     # CLI entry points
│   ├── train_signllm.py         # Training launcher
│   ├── predict.py               # Single-word inference
│   ├── evaluate_runs.py         # Full dev+test evaluation
│   ├── compute_baselines.py     # NN / mean-pose / random-clip baselines
│   ├── generate_openpose_video.py   # Single-video OpenPose demo
│   ├── multi_openpose_demo.py   # Multi-video 7-variant demo pipeline
│   └── visualize_pose.py        # Skeleton GIF renderer
│
├── outputs/                     # Generated artefacts (tracked via Git LFS)
│   ├── videos/                  # 7 render variants per clip
│   │   ├── overlay/             # Skeleton on original frame
│   │   ├── skeleton/            # Skeleton on black background
│   │   ├── neon/                # Glowing neon skeleton
│   │   ├── heatmap/             # Confidence heatmap blend
│   │   ├── slowmo/              # 0.5x speed
│   │   ├── studio/              # Dark gradient background
│   │   └── mosaic/              # 3x2 tile of all variants
│   ├── openpose_overlay/        # Original clips + skeleton overlay
│   ├── openpose_json/           # Per-frame OpenPose-compatible JSON
│   └── figures/                 # Notebook plots and animations
│
├── assets/                      # Model weights (tracked via Git LFS)
│   └── holistic_landmarker.task # MediaPipe Holistic model (13 MB)
│
├── data/                        # Dataset metadata (CSV only, no videos)
│   ├── labels.csv               # 2,216 clips with Arabic labels
│   ├── splits.csv               # Train/val/test assignment per clip
│   └── video_meta.csv           # fps, resolution, duration per clip
│
├── docs/                        # Methodology and decisions
│   ├── DECISIONS.md             # Design decision log
│   ├── MODEL.md                 # Architecture specification
│   ├── PIPELINE.md              # End-to-end pipeline description
│   ├── RESULTS.md               # Full evaluation results
│   └── STATS.md                 # Dataset statistics
│
├── docker/                      # Container setup (NGC PyTorch 26.04)
├── final_project.ipynb          # 45-cell research notebook (all stages)
├── requirements.txt             # Python dependencies
├── .gitattributes               # Git LFS tracking rules
└── .gitignore
```

---

## Quick Start

### 1. Clone with LFS objects

```bash
git clone https://github.com/soufianeboulahcen/Master_Multimodal-Moroccan-SLG.git
cd Master_Multimodal-Moroccan-SLG
git lfs pull          # downloads videos, model weights, NPZ files
```

### 2. Install dependencies

```bash
pip install torch numpy opencv-python-headless mediapipe scipy matplotlib tqdm
```

### 3. Run the OpenPose demo on any video

```python
import os, sys
from pathlib import Path

video = Path("/path/to/your/video.mp4").resolve()
sys.argv = ["multi_openpose_demo.py", "--clips", str(video),
            "--out-dir", "my_output"]
import runpy
runpy.run_path("scripts/multi_openpose_demo.py", run_name="__main__")
```

### 4. Open the research notebook

```bash
jupyter notebook final_project.ipynb
```

### 5. Train SignLLM (requires Docker + GPU)

```bash
docker/run.sh python scripts/train_signllm.py --mode mse --run-name baseline_mse
docker/run.sh bash scripts/run_ablation.sh   # all three ablation runs
```

---

## Pipeline Stages

```
Raw .mp4 videos (MoSL dataset)
        |
        v
[Stage 1] Pose Extraction
  mosl/pose/extract_dataset.py
  pytorch-openpose per frame -> data/processed/keypoints_2d/*.npz
        |
        v
[Stage 2] 2D -> 3D Lifting (Prompt2Sign)
  mosl/pose/export_openpose_json.py
  -> per-frame OpenPose JSON
  -> Prompt2Sign pipeline (json2h5 -> h5totxt -> txt2skels)
  -> final_data/{train,dev,test}.{skels,text,files}
        |
        v
[Stage 3] SignLLM Training
  scripts/train_signllm.py
  Three loss modes: MSE / RL / RL+PLC
  -> runs/<run_name>/best.pt
        |
        v
[Stage 4] Evaluation
  scripts/evaluate_runs.py      -> teacher-forced MSE + AR DTW
  scripts/compute_baselines.py  -> NN / mean-pose / random-clip
        |
        v
[Stage 5] OpenPose Video Generation
  scripts/generate_openpose_video.py  -> overlay + skeleton + JSON
  scripts/multi_openpose_demo.py      -> 7 variants + mosaic
```

---

## Generated Outputs

### Video Variants (`outputs/videos/`)

| Folder | Description | Background |
|---|---|---|
| `overlay/` | Skeleton on original frame | Original video |
| `skeleton/` | Clean skeleton only | Pure black |
| `neon/` | Glowing neon skeleton (Gaussian blur) | Pure black |
| `heatmap/` | Confidence heatmap blended over original | Original video |
| `slowmo/` | 0.5x speed (frame duplication) | Original video |
| `studio/` | Skeleton on dark vignette gradient | Charcoal gradient |
| `mosaic/` | 3x2 tile of all 6 variants | Combined |

### JSON Keypoint Schema (`outputs/openpose_json/`)

OpenPose-compatible per-frame JSON:

```json
{
  "version": 1.3,
  "frame_index": 0,
  "image_size": {"width": 460, "height": 460},
  "people": [{
    "pose_keypoints_2d":       [x0,y0,c0, x1,y1,c1, ...],
    "hand_left_keypoints_2d":  [x0,y0,c0, ...],
    "hand_right_keypoints_2d": [x0,y0,c0, ...],
    "face_keypoints_2d":       [x0,y0,c0, ...]
  }]
}
```

| Field | Joints | Values |
|---|---|---|
| `pose_keypoints_2d` | 18 COCO body joints | 54 |
| `hand_*_keypoints_2d` | 21 hand joints each | 63 each |
| `face_keypoints_2d` | 478 face mesh points | 1,434 |

---

## Dataset

The **MoSL video dataset** is available at:
[https://data.mendeley.com/datasets/23phgyt3mt/1](https://data.mendeley.com/datasets/23phgyt3mt/1)

| Category | Clips | Unique Signs |
|---|---:|---:|
| Diverse | 1,941 | 1,508 |
| Numbers | 130 | 51 |
| Letters | 71 | 39 |
| days_months_seasons | 59 | 23 |
| Pronouns | 15 | 10 |
| **Total** | **2,216** | **1,631** |

Raw `.mp4` files are **not included** (221 MB). Download from Mendeley and place under `data/vedios-dataset/`.

---

## Git LFS

This repository uses [Git LFS](https://git-lfs.github.com/) for large binary files:

| Pattern | Content |
|---|---|
| `*.mp4` | Generated videos |
| `*.task` | MediaPipe model weights |
| `*.npz` | NumPy pose arrays |
| `*.gif` | Skeleton animations |
| `*.pt` | PyTorch checkpoints |

```bash
# Install Git LFS before cloning
sudo apt-get install git-lfs   # Ubuntu/Debian
brew install git-lfs           # macOS
git lfs install

# Clone (LFS files download automatically)
git clone https://github.com/soufianeboulahcen/Master_Multimodal-Moroccan-SLG.git
```

---

## Citation

```bibtex
@misc{fang2024signllm,
  title         = {SignLLM: Sign Languages Production Large Language Models},
  author        = {Fang, Sen and others},
  year          = {2024},
  eprint        = {2405.10718},
  archivePrefix = {arXiv},
}

@data{benzaid2026mosl,
  title     = {Moroccan Sign Language Video Dataset},
  author    = {Ben Zaid and others},
  year      = {2026},
  publisher = {Mendeley Data},
  doi       = {10.17632/23phgyt3mt.1},
}
```

---

## License

[MIT](LICENSE)
