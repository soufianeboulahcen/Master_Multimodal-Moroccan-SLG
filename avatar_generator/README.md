# avatar_generator

Standalone module that converts SignLLM skeleton outputs into photorealistic
avatar images and videos.  No existing pipeline is modified.

---

## Pipeline

```
pose_source (.skels or OpenPose JSON dir)
        │
        ▼  pose_loader.py
list[FramePose]  — normalised keypoints per frame
        │
        ▼  pose_renderer.py
list[PIL Image]  — RGB ControlNet conditioning maps (768×768)
        │
        ├──► reference_image (optional)
        │    IP-Adapter identity conditioning
        │
        ▼  diffusion_engine.py
        SDXL 1.0 + ControlNet-OpenPose + IP-Adapter-Plus-Face
list[PIL Image]  — generated avatar frames
        │
        ▼  video_assembler.py
        MP4 video  or  PNG frames  or  contact sheet
```

---

## Folder structure

```
avatar_generator/
├── __init__.py          Public API: AvatarGenerator, GeneratorConfig
├── config.py            GeneratorConfig dataclass
├── config.yaml          Default configuration (edit this)
├── generator.py         AvatarGenerator — top-level orchestrator
├── pose_loader.py       Load .skels or OpenPose JSON → FramePose list
├── pose_renderer.py     FramePose → RGB ControlNet pose map (PIL Image)
├── diffusion_engine.py  SDXL + ControlNet + IP-Adapter inference
├── video_assembler.py   Frames → MP4 / PNG / contact sheet
├── run.py               CLI entry point
└── README.md            This file
```

---

## Installation

```bash
pip install diffusers>=0.27.0 transformers>=4.40.0 accelerate safetensors
pip install opencv-python-headless Pillow imageio[ffmpeg]
pip install xformers          # optional but recommended (halves VRAM)
```

Model weights (~10 GB) are downloaded automatically from HuggingFace on
first run.  To pre-fetch:

```bash
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0
huggingface-cli download thibaud/controlnet-openpose-sdxl-1.0
huggingface-cli download madebyollin/sdxl-vae-fp16-fix
huggingface-cli download h94/IP-Adapter  # only if using identity conditioning
```

---

## Quick start

### Dry run — pose maps only (no GPU, no model weights)

Validates the pose loading and rendering pipeline instantly:

```bash
python avatar_generator/run.py \
    --pose outputs/openpose_json/walking_keypoints \
    --pose-maps-only \
    --out  outputs/avatar_generator/walking_pose_maps/
```

Outputs `pose_000000.png … pose_000149.png` + `pose_grid.png` contact sheet.

### Single image

```bash
python avatar_generator/run.py \
    --pose        outputs/openpose_json/walking_keypoints \
    --out         outputs/avatar_generator/walking_frame0.png \
    --mode        image \
    --frame-index 0
```

### Video clip

```bash
python avatar_generator/run.py \
    --pose outputs/openpose_json/walking_keypoints \
    --out  outputs/avatar_generator/walking.mp4
```

### Video with identity conditioning

```bash
python avatar_generator/run.py \
    --pose      outputs/openpose_json/walking_keypoints \
    --reference outputs/avatar_generator/reference/signer.jpg \
    --out       outputs/avatar_generator/walking_identity.mp4
```

### Quick preview (9-frame contact sheet)

```bash
python avatar_generator/run.py \
    --pose outputs/openpose_json/walking_keypoints \
    --out  outputs/avatar_generator/walking_preview.png \
    --mode contact-sheet
```

### From a .skels file

```bash
python avatar_generator/run.py \
    --pose       data/processed/final_data/train.skels \
    --out        outputs/avatar_generator/train_sample.mp4 \
    --max-frames 30
```

### Batch all procedural motions

```bash
python avatar_generator/run.py \
    --batch-all \
    --reference outputs/avatar_generator/reference/signer.jpg \
    --out       outputs/avatar_generator/
```

---

## Python API

```python
from avatar_generator import AvatarGenerator, GeneratorConfig

# Default config
cfg = GeneratorConfig()

# Or load from YAML
cfg = GeneratorConfig.from_yaml("avatar_generator/config.yaml")

# Override specific values
cfg.num_steps = 30
cfg.width = 1024
cfg.height = 1024

# Single image
with AvatarGenerator(cfg) as gen:
    img = gen.generate_image(
        pose_source="outputs/openpose_json/walking_keypoints",
        frame_index=0,
        reference_image="signer.jpg",   # optional
    )
    img.save("avatar_frame.png")

# Full video
with AvatarGenerator(cfg) as gen:
    gen.generate(
        pose_source="outputs/openpose_json/walking_keypoints",
        output_path="outputs/avatar_generator/walking.mp4",
        reference_image="signer.jpg",   # optional
        max_frames=50,                  # optional: limit for testing
    )

# Batch (weights loaded once)
jobs = [
    {"pose_source": "outputs/openpose_json/walking_keypoints",
     "output_path": "outputs/avatar_generator/walking.mp4"},
    {"pose_source": "outputs/openpose_json/running_keypoints",
     "output_path": "outputs/avatar_generator/running.mp4"},
]
with AvatarGenerator(cfg) as gen:
    gen.generate_batch(jobs, reference_image="signer.jpg")
```

---

## Input formats

### OpenPose JSON directory

Directory of `*_keypoints.json` files in CMU OpenPose v1.3 format.
Already produced by the existing pipeline:
- `outputs/openpose_json/<motion>_keypoints/` — procedural motions
- `mosl/pose/export_openpose_json.py` — from SignLLM predictions

### .skels file

SignLLM compressed pose format.  One frame per line:
```
<time> <x0> <y0> <z0> <x1> <y1> <z1> ... (50 joints × 3 coords)
```
Coords are normalised to approximately `[-1, 1]`.  The loader projects them
to pixel space using `skels_canvas_width` × `skels_canvas_height`.

Joint layout:
- joints 0–7: body (8 joints, COCO subset)
- joints 8–28: left hand (21 joints, full OpenPose hand model)
- joints 29–49: right hand (21 joints)

---

## Configuration

Edit `avatar_generator/config.yaml` or pass overrides at runtime.

Key parameters:

| Parameter | Default | Notes |
|---|---|---|
| `num_steps` | 25 | Increase to 40–50 for higher quality |
| `controlnet_conditioning_scale` | 0.85 | Higher = more pose-faithful |
| `identity_strength` | 0.40 | 0 = no identity; 0.3–0.5 recommended |
| `width` / `height` | 768 | 1024 for higher resolution (needs more VRAM) |
| `batch_size` | 1 | Increase for throughput if VRAM allows |
| `cpu_offload` | false | Enable on GPUs with <16 GB VRAM |

---

## GPU requirements

| Config | VRAM needed |
|---|---|
| SDXL + ControlNet only (`identity_strength=0`) | ~12 GB |
| + IP-Adapter | ~14 GB |
| 1024×1024 resolution | ~18 GB |
| DGX Spark GB10 (128 GB unified) | ✅ all configs |

---

## Relation to avatar/ module

This module (`avatar_generator/`) is a **simpler, standalone** alternative
to the `avatar/` module (on `feat/avatar-pose-conditioning`).

| | `avatar_generator/` | `avatar/` |
|---|---|---|
| Backend | SDXL + ControlNet (single branch) | AnimateDiff + dual ControlNet |
| Temporal consistency | Per-frame (same seed/prompt) | AnimateDiff motion module |
| Identity | IP-Adapter | InstantID + IP-Adapter |
| VRAM | ~12–14 GB | ~28–36 GB |
| Setup complexity | Low | High |
| Best for | Quick generation, single frames, testing | Production sign-language video |

Use `avatar_generator/` to get started and validate the pipeline.
Use `avatar/` for final production renders with full temporal consistency.
