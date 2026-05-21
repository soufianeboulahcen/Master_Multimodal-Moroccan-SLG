# Avatar Rendering Pipeline

Photorealistic avatar video generation built on top of the completed SignLLM
skeleton pipeline.  The SignLLM implementation is untouched; this subsystem
consumes its outputs.

---

## Integration seam

```
Arabic text
    │
    ▼  (existing SignLLM pipeline — unchanged)
SignLLM.generate()  →  (T, 150) pose sequence
    │
    ▼  mosl/pose/export_openpose_json.py  (existing, unchanged)
outputs/openpose_json/<sign>/<frame>_keypoints.json
    │
    ▼  avatar/conditioning/pose_to_controlnet.py  (new)
RGB ControlNet pose maps  (body branch + hand branch)
    │
    ├──► avatar/conditioning/identity_encoder.py  (new)
    │    reference signer image → InstantID / IP-Adapter tensors
    │
    ▼  avatar/backends/animatediff.py  or  mimicmotion.py  (new)
raw video frames  (T, H, W, 3)
    │
    ▼  avatar/interpolation/rife.py  (new)
temporally upsampled frames  (T × scale_factor, H, W, 3)
    │
    ▼  ffmpeg
outputs/avatar/<sign>/<sign>_avatar.mp4
```

The existing `.skels` format, tokenizer, training pipeline, evaluation
scripts, and baseline scripts are not modified.

---

## Repository layout (new files only)

```
avatar/
├── __init__.py
├── config.py                      AvatarConfig + all sub-configs
├── pipeline.py                    Orchestrator: JSON → MP4
├── conditioning/
│   ├── pose_to_controlnet.py      OpenPose JSON → RGB pose maps
│   └── identity_encoder.py       InstantID / IP-Adapter encoding
├── backends/
│   ├── base.py                    Abstract RenderBackend + RenderResult
│   ├── animatediff.py             AnimateDiff + SDXL + dual ControlNet
│   └── mimicmotion.py             MimicMotion (SVD-based full-body)
└── interpolation/
    └── rife.py                    RIFE temporal upsampling

scripts/
└── render_avatar.py               CLI entry point

docs/
└── AVATAR_PIPELINE.md             This file

requirements-avatar.txt            Diffusion stack dependencies
```

---

## Quick start

### Pose maps only (no GPU, no model weights)

Validates the keypoint remapping and ControlNet conditioning images:

```bash
python scripts/render_avatar.py \
    --json-dir outputs/openpose_json/walking_keypoints \
    --pose-maps-only \
    --out outputs/avatar/walking/pose_maps
```

Outputs:
- `outputs/avatar/walking/pose_maps/body/body_XXXXXX.png` — full skeleton maps
- `outputs/avatar/walking/pose_maps/hand/hand_XXXXXX.png` — hand-only maps
- `outputs/avatar/walking/pose_maps/debug_grid.png` — 3×3 visual inspection grid

### Full render (requires GPU + model weights)

```bash
# Install dependencies first
pip install -r requirements-avatar.txt

# Render from existing OpenPose JSON
python scripts/render_avatar.py \
    --json-dir outputs/openpose_json/walking_keypoints \
    --reference outputs/avatar/reference_images/signer.jpg \
    --out outputs/avatar/walking/walking_avatar.mp4

# Generate sign from text, then render
python scripts/render_avatar.py \
    --sign أَنَا \
    --run baseline_mse \
    --reference outputs/avatar/reference_images/signer.jpg \
    --out outputs/avatar/أَنَا/أَنَا_avatar.mp4

# Batch render all procedural motions
python scripts/render_avatar.py \
    --batch-motions \
    --reference outputs/avatar/reference_images/signer.jpg
```

### Python API

```python
from avatar import AvatarPipeline, AvatarConfig, BackendType

cfg = AvatarConfig(backend=BackendType.ANIMATEDIFF)

with AvatarPipeline(cfg) as pipe:
    result = pipe.render(
        json_dir="outputs/openpose_json/أَنَا_keypoints",
        reference_image="outputs/avatar/reference_images/signer.jpg",
        output_path="outputs/avatar/أَنَا/أَنَا_avatar.mp4",
    )

print(f"Generated {result.n_frames} frames at {result.fps} fps")
```

---

## Model weights

Download before first use.  All weights are fetched from HuggingFace Hub
automatically on first run, or manually:

| Model | HuggingFace ID | Size |
|---|---|---|
| SDXL base | `stabilityai/stable-diffusion-xl-base-1.0` | ~7 GB |
| AnimateDiff SDXL adapter | `guoyww/animatediff-motion-adapter-sdxl-beta` | ~1.5 GB |
| ControlNet OpenPose SDXL | `thibaud/controlnet-openpose-sdxl-1.0` | ~1.5 GB |
| IP-Adapter Plus Face SDXL | `h94/IP-Adapter` | ~1 GB |
| InstantID | `InstantX/InstantID` | ~700 MB |
| MimicMotion | `tencent/MimicMotion` | ~8 GB |
| RIFE 4.6 | `AlexWortega/RIFE` or manual | ~200 MB |

Total: ~20 GB.  All stored in HuggingFace cache (`~/.cache/huggingface/`).

---

## Backend selection

| Backend | Best for | VRAM | Speed |
|---|---|---|---|
| `animatediff` (default) | Hand-critical sign language motion | 28–36 GB | ~3s/frame |
| `mimicmotion` | Full-body photorealistic showcase | 20–32 GB | ~5s/frame |

The DGX Spark GB10 (128 GB unified memory) comfortably fits either stack.

Backend selection logic in `AvatarConfig`:
```python
cfg = AvatarConfig(backend=BackendType.ANIMATEDIFF)   # explicit
cfg = AvatarConfig(backend=BackendType.AUTO)           # defaults to AnimateDiff
```

---

## Keypoint remapping

Our OpenPose JSON has:
- **Body**: 25 keypoints (BODY_25) or 18 (COCO-18 from Hzzone)
- **Hands**: 11 keypoints per hand (palm + 2 joints/finger, no fingertips)

ControlNet-OpenPose expects:
- **Body**: 18 keypoints (COCO-18, indices 0–17)
- **Hands**: 21 keypoints per hand (full OpenPose hand model)

`pose_to_controlnet.py` handles both remappings:
1. Body: indices 0–17 are identical between BODY_25 and COCO-18 for
   upper-body joints.  Indices 18–24 (feet, mid-hip) are dropped.
2. Hands: missing distal joints (fingertips + one intermediate per finger)
   are linearly extrapolated from the two detected proximal joints.

---

## Dual ControlNet strategy

Two ControlNet branches are used simultaneously:

| Branch | Input | Weight | Purpose |
|---|---|---|---|
| Body branch | Full skeleton pose map | 0.65 | Body pose context |
| Hand branch | Hands-only pose map (supersampled) | 0.90 | Sign-critical hand detail |

The hand branch has higher weight because hand accuracy is the primary
quality axis for sign language generation.  The supersampling (2× render
then downsample) anti-aliases the thin finger connection lines.

---

## Temporal consistency

Three layers of temporal consistency:

1. **AnimateDiff motion module**: enforces temporal coherence in latent
   space across the 16-frame context window.
2. **Sliding window with overlap**: 4-frame overlap between adjacent
   windows; overlapping frames are blended with a linear alpha ramp.
3. **RIFE interpolation**: optical-flow-based frame insertion after
   diffusion rendering (25fps → 50fps by default).

---

## Identity preservation

InstantID (default) uses two injection points:
- **ArcFace embedding** (512-d): injected via IP-Adapter into UNet
  cross-attention — controls appearance.
- **Face ControlNet**: controls face geometry/structure.

IP-Adapter (fallback) uses CLIP ViT-H/14 image embedding — works for
full-body reference images where face detection may fail.

Identity strength is configurable (`IdentityConfig.identity_strength`,
default 0.40).  Higher values increase identity fidelity at the cost of
pose adherence.  For sign language, 0.35–0.45 is the recommended range.

---

## GPU deployment on DGX Spark

Recommended inference order for a batch of N signs:

```
1. Pre-render all pose maps (CPU, ~0.01s/frame)
2. Encode reference identity once (GPU, ~2s, cached to disk)
3. Load AnimateDiff stack (GPU, ~30s one-time)
4. AnimateDiff: render all N signs sequentially (~3s/frame × T frames × N signs)
5. Unload AnimateDiff stack
6. Load RIFE (GPU, ~5s one-time)
7. RIFE: interpolate all N outputs (~0.1s/frame)
8. Unload RIFE
9. ffmpeg: assemble all videos (CPU, ~1s/video)
```

Batch mode in the CLI:
```bash
python scripts/render_avatar.py --batch-motions --reference signer.jpg
```

---

## Feature branch roadmap

| Branch | Status | Deliverable |
|---|---|---|
| `feat/avatar-pose-conditioning` | ✅ This branch | Pose maps, identity encoding, all backends, RIFE, pipeline, CLI |
| `feat/avatar-animatediff-weights` | Next | Download + verify model weights; end-to-end smoke test on DGX |
| `feat/avatar-hand-quality` | After weights | Hand super-resolution post-processing; ControlNet weight tuning |
| `feat/avatar-mimicmotion-backend` | Parallel | MimicMotion DWPose integration; full-body showcase renders |
| `feat/avatar-batch-inference` | After hand quality | Optimised batch pipeline; queue management for DGX |

---

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Primary backend | AnimateDiff | Better hand detail via dual ControlNet; sign language is hand-critical |
| Identity backend | InstantID | Stronger face preservation than IP-Adapter alone |
| Interpolation | RIFE 4.6 | Best quality/speed; optical flow handles fast hand motion |
| Hand ControlNet weight | 0.90 | Higher than body (0.65) to prioritise finger accuracy |
| Context window | 16 frames | AnimateDiff default; longer windows increase VRAM without quality gain |
| Overlap | 4 frames | Eliminates hard seams at window boundaries |
| Identity strength | 0.40 | Pose must dominate over identity for sign accuracy |
