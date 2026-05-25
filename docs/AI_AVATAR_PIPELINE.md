# AI Avatar Video Generation Pipeline

Upgrade path from the existing OpenPose-based SignLLM system to a
production-ready diffusion-based avatar video generator.

This document is the single source of truth for the integration design.
It does **not** replace any existing code — it layers on top of it.

---

## 0. Quick start — OpenPose video → avatar (new)

The fastest path from an existing skeleton video to a cinematic avatar:

```bash
# Build the diffusion image (once)
docker build -t pfe-avatar:latest -f docker/Dockerfile.diffusion .

# Single video — skeleton MP4 → avatar MP4
docker/run_avatar.sh python scripts/avatar/video_to_avatar.py \
    --input  outputs/videos/skeleton/walking_skeleton.mp4 \
    --reference assets/avatar_reference.jpg \
    --prompt "a Moroccan man performing sign language, studio background, \
              photorealistic, DSLR, cinematic lighting" \
    --out    outputs/avatar/videos/walking_hq.mp4 \
    --interpolate --target-fps 24 \
    --upscale

# From JSON keypoints (cleaner pose maps, preferred)
docker/run_avatar.sh python scripts/avatar/video_to_avatar.py \
    --input  outputs/openpose_json/walking_keypoints/ \
    --source json \
    --reference assets/avatar_reference.jpg \
    --prompt "a Moroccan man performing sign language, photorealistic" \
    --out    outputs/avatar/videos/walking_hq.mp4

# Batch — all skeleton videos at once
docker/run_avatar.sh python scripts/avatar/video_to_avatar.py \
    --input  outputs/videos/skeleton/ \
    --batch \
    --reference assets/avatar_reference.jpg \
    --prompt "a Moroccan signer, photorealistic, studio background" \
    --out-dir outputs/avatar/videos/batch/
```

The pipeline runs three stages automatically:

| Stage | Script | What it does |
|---|---|---|
| 1 | `video_to_pose_maps.py` | Extract ControlNet PNG pose maps from the video |
| 2 | `generate_avatar_video.py` | AnimateDiff + SDXL + ControlNet + IP-Adapter |
| 3 | `postprocess.py` | RIFE interpolation + Real-ESRGAN upscale |

Each stage can be skipped independently (`--skip-pose`, `--skip-generation`,
`--skip-postprocess`) to resume after a partial run.

### Recommended generation settings (per spec)

| Parameter | Value | Flag |
|---|---|---|
| FPS | 24 | `--fps 24` |
| CFG scale | 7.0 | `--cfg 7` |
| Denoise strength | 0.7 | `--denoise-strength 0.7` |
| ControlNet weight | 0.9 | `--controlnet-scale 0.9` |
| IP-Adapter scale | 0.65 | `--ip-scale 0.65` |
| Frame count | 24–48 | `--n-frames 48` |

---

## 1. What already exists (assets we reuse)

| Asset | Location | Format | Role in new pipeline |
|---|---|---|---|
| Per-frame OpenPose JSON | `outputs/openpose_json/` | CMU v1.3 JSON | ControlNet conditioning input |
| Per-clip NPZ keypoints | `data/processed/keypoints_2d/` | `(T,54)+(T,63)×2` float32 | Pose sequence source |
| Skeleton videos | `outputs/videos/skeleton/` | MP4 30fps | Direct video input (new) |
| Procedural skeleton frames | `outputs/frames/*/` | JPEG 1280×720 | ControlNet conditioning (no-GPU path) |
| SignLLM predicted poses | `predictions/*.npz` | `(T,150)` float32 | Generated motion → avatar |
| Skeleton renderer | `scripts/generate_openpose/renderer.py` | Python/OpenCV | Pose-map image generator |
| Rig definition | `scripts/generate_openpose/rig.py` | Python/numpy | Keypoint topology |
| Docker + NGC PyTorch | `docker/` | CUDA 13 / SM_120 | GPU runtime |

**Key insight**: the existing pipeline already produces everything ControlNet
OpenPose needs. The only missing piece is the diffusion model stack on top.

---

## 2. Full system architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                  │
│                                                                     │
│  Arabic text  ──► SignLLM (existing)  ──► pose NPZ (T,150)         │
│                                                                     │
│  OR  procedural motions (motions.py)  ──► keypoints (T,52,2)       │
│                                                                     │
│  OR  real MoSL video  ──► pytorch-openpose  ──► keypoints NPZ      │
│                                                                     │
│  OR  existing skeleton MP4  ──────────────────────────────────────► │
│      outputs/videos/skeleton/*.mp4   (video_to_pose_maps.py)        │
│                                                                     │
│  OR  existing JSON keypoints  ────────────────────────────────────► │
│      outputs/openpose_json/*/        (video_to_pose_maps.py)        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     POSE MAP GENERATION                             │
│                                                                     │
│  scripts/avatar/pose_to_controlnet_map.py  (NEW)                   │
│                                                                     │
│  Input : keypoints (any format above)                               │
│  Output: RGB pose-map images (512×768 or 768×512) per frame        │
│          — body skeleton drawn in OpenPose colour convention        │
│          — hand skeleton overlaid                                   │
│          — face landmarks overlaid (if available)                   │
│                                                                     │
│  Reuses: renderer.py draw logic, rig.py topology                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   IDENTITY REFERENCE FRAME                          │
│                                                                     │
│  One reference portrait image of the target avatar                 │
│  (photo, AI-generated face, or neutral frame from a real clip)     │
│                                                                     │
│  ► IP-Adapter / InstantID encodes face identity embedding          │
│  ► Stored as a 512-dim face latent, reused for every frame         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              FRAME-LEVEL DIFFUSION (per-frame anchor)               │
│                                                                     │
│  Model  : SDXL 1.0 + ControlNet OpenPose (SDXL variant)            │
│  OR       FLUX.1-dev + ControlNet (if VRAM ≥ 24 GB)               │
│                                                                     │
│  Inputs :                                                           │
│    • pose-map image  (ControlNet conditioning, strength 0.7–0.85)  │
│    • text prompt     ("a Moroccan man signing in front of …")      │
│    • face embedding  (IP-Adapter, scale 0.6–0.8)                   │
│                                                                     │
│  Output : anchor frame latents  (one per N frames, e.g. every 8)  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              VIDEO DIFFUSION (temporal coherence)                   │
│                                                                     │
│  Model  : AnimateDiff v3 motion module  (primary choice)           │
│  OR       Stable Video Diffusion XT     (img2vid, 25 frames)       │
│                                                                     │
│  AnimateDiff path:                                                  │
│    • Base UNet = SDXL (same weights as frame-level step above)     │
│    • Motion module injected into temporal attention layers          │
│    • ControlNet OpenPose applied at every frame simultaneously     │
│    • Processes 16–24 frame windows with 4-frame overlap            │
│    • Sliding window stitched with DDIM inversion on overlap        │
│                                                                     │
│  SVD path (alternative for short clips ≤ 25 frames):              │
│    • Condition on anchor frame (from SDXL step above)              │
│    • motion_bucket_id = 100–127  (sign language = controlled motion)│
│    • augmentation_level = 0.02                                     │
│                                                                     │
│  Output : video latent tensor  (B, C, T, H, W)                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    VIDEO DECODER + POST-PROCESS                     │
│                                                                     │
│  VAE decode  → RGB frames  (512×768 or 768×512)                    │
│  RIFE / FILM interpolation → 2× or 4× frame rate upscale          │
│  Real-ESRGAN ×2 → final resolution (1024×1536 or 1280×720)        │
│  ffmpeg encode → H.264 MP4 (matches existing outputs/ convention)  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data format mapping (existing → new)

### 3.1 NPZ keypoints → ControlNet pose map

The existing NPZ layout:
```
pose_keypoints_2d        (T, 54)   18 COCO body × (x, y, confidence)
hand_left_keypoints_2d   (T, 63)   21 hand × (x, y, confidence)
hand_right_keypoints_2d  (T, 63)
```

Mapping to ControlNet OpenPose colour convention (DWPose / OpenPose-v2):
- Body: COCO-18 → draw 17 limbs in standard colour palette
- Hands: 21-point chains → draw 20 finger segments per hand
- Face: not in NPZ → use 5-point simplified face from rig.py or skip

The `pose_to_controlnet_map.py` script handles this conversion.
Output: `(H, W, 3)` uint8 RGB image per frame, saved as PNG.

### 3.2 SignLLM predicted pose → ControlNet pose map

The predicted pose layout:
```
pose  (T, 150)   50 joints × (x, y, z)
  joints 0..7    = 8 upper-body (COCO subset)
  joints 8..28   = 21 left hand
  joints 29..49  = 21 right hand
```

The (x, y) components are in the Prompt2Sign normalized coordinate space
(divided by 9 from pixel space). Denormalization:
```python
xy_pixel = pose[:, :, :2] * 9   # undo /9 normalization
# then scale to target resolution
```

### 3.3 Procedural keypoints → ControlNet pose map

The procedural rig uses normalized [0,1] coords with 52 keypoints.
Direct mapping: `x_pixel = x_norm * W`, `y_pixel = y_norm * H`.
The existing `renderer.py` already does this — we reuse `_to_px()` and
`_draw_skeleton()` but output to a white or neutral background instead of
the cinematic black background.

---

## 4. Recommended models

### Primary stack (VRAM 16–24 GB, DGX Spark GB10)

| Component | Model | HuggingFace ID | VRAM |
|---|---|---|---|
| Base image model | SDXL 1.0 | `stabilityai/stable-diffusion-xl-base-1.0` | 8 GB |
| ControlNet (pose) | ControlNet SDXL OpenPose | `thibaud/controlnet-openpose-sdxl-1.0` | +2 GB |
| Motion module | AnimateDiff v3 | `guoyww/animatediff-motion-adapter-sdxl-beta` | +1 GB |
| Identity | IP-Adapter SDXL Face | `h94/IP-Adapter` → `ip-adapter-plus-face_sdxl_vit-h.bin` | +1 GB |
| Upscaler | Real-ESRGAN x2 | `ai-forever/Real-ESRGAN` | 0.5 GB |
| Frame interpolation | RIFE v4.6 | local weights | 0.3 GB |

**Total peak VRAM**: ~13–15 GB — fits on GB10 (121 GB unified memory).

### Alternative stack (higher quality, VRAM ≥ 24 GB)

| Component | Model | Notes |
|---|---|---|
| Base | FLUX.1-dev | Better anatomy, slower |
| ControlNet | `XLabs-AI/flux-controlnet-openpose-v3` | FLUX ControlNet |
| Video | CogVideoX-5B | Better temporal coherence than AnimateDiff |
| Identity | InstantID | Better face fidelity than IP-Adapter |

### Lightweight stack (no GPU / CPU fallback)

| Component | Model | Notes |
|---|---|---|
| Base | SD 1.5 | `runwayml/stable-diffusion-v1-5` |
| ControlNet | `lllyasviel/control_v11p_sd15_openpose` | Well-tested |
| Motion | AnimateDiff v2 | `guoyww/animatediff-motion-adapter-v1-5-2` |
| Identity | IP-Adapter SD1.5 | `h94/IP-Adapter` → `ip-adapter-plus-face_sd15.bin` |

---

## 5. Optimal generation settings

### AnimateDiff + SDXL

```python
# Frame window
window_size     = 16        # frames per diffusion pass
window_overlap  = 4         # overlap for temporal stitching
fps_target      = 8         # AnimateDiff native fps (interpolate to 25 later)

# Diffusion
num_inference_steps = 25    # DDIM
guidance_scale      = 7.5   # CFG
denoise_strength    = 1.0   # full generation (not img2img)

# ControlNet
controlnet_conditioning_scale = 0.75   # pose adherence vs. realism trade-off
                                        # 0.6 = more creative, 0.9 = strict pose

# IP-Adapter (identity)
ip_adapter_scale = 0.65     # 0.5 = subtle identity, 0.8 = strong identity

# Resolution
width  = 768
height = 512    # landscape for sign language (signer visible head-to-waist)
# OR
width  = 512
height = 768    # portrait (full body)
```

### Stable Video Diffusion (short clips ≤ 25 frames)

```python
num_frames          = 25
motion_bucket_id    = 110   # 100=slow, 127=fast; sign language ≈ 110
fps                 = 7     # SVD native fps
decode_chunk_size   = 8     # VRAM trade-off
noise_aug_strength  = 0.02  # low = more faithful to anchor frame
```

### Post-processing

```python
# RIFE interpolation: 8fps → 25fps (3.125× = 2 passes of 2×)
rife_passes = 2   # 8 → 16 → 32fps, then drop to 25fps

# Real-ESRGAN upscale
upscale_factor = 2   # 512×768 → 1024×1536
tile_size      = 512  # avoid OOM on large frames
```

---

## 6. Identity preservation strategy

### Option A — IP-Adapter (recommended, simpler)

1. Provide one reference portrait image of the avatar.
2. Extract CLIP image embedding via `CLIPImageProcessor`.
3. Inject into UNet cross-attention via IP-Adapter projection layers.
4. Use `ip_adapter_scale=0.65` — high enough to preserve face, low enough
   to not fight the pose conditioning.
5. The same embedding is reused for every frame → consistent identity.

```python
from diffusers import StableDiffusionXLPipeline
pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter-plus-face_sdxl_vit-h.bin",
)
pipeline.set_ip_adapter_scale(0.65)
```

### Option B — InstantID (higher fidelity, requires face detection)

1. Run InsightFace on the reference portrait → 512-dim ArcFace embedding.
2. Also extract facial keypoints → ControlNet IdentityNet conditioning.
3. Two ControlNet inputs simultaneously: OpenPose (body) + IdentityNet (face).
4. Requires `InstantID` pipeline wrapper.

InstantID is preferred when the avatar must be recognizable as a specific
real person. IP-Adapter is sufficient for a generic consistent avatar.

### Temporal identity consistency

The main risk is face drift across frames. Mitigations:
- **Anchor frame injection**: generate frame 0 with full quality, then use
  it as the `image` conditioning for SVD or as the first frame of AnimateDiff.
- **IP-Adapter at every window**: pass the same face embedding to every
  sliding window pass — prevents drift accumulation.
- **Low CFG on subsequent windows**: use `guidance_scale=5.0` for windows
  after the first (less creative = more consistent with anchor).

---

## 7. Temporal consistency strategy

### Problem
AnimateDiff processes 16-frame windows. Naive concatenation produces
visible seams at window boundaries (lighting shift, face drift, motion jump).

### Solution: sliding window with DDIM inversion overlap

```
Window 1: frames  0-15  (full denoise, guidance=7.5)
Window 2: frames 12-27  (4-frame overlap with window 1)
  → frames 12-15: DDIM-invert window-1 output to get noisy latents
  → use as starting point for window 2 denoising
  → blend overlap region: linear crossfade over 4 frames
Window 3: frames 24-39  (same pattern)
```

This is implemented in `scripts/avatar/generate_avatar_video.py` via the
`SlidingWindowAnimateDiff` class.

### Additional temporal stabilizers

1. **Optical flow warping**: between anchor frames, warp the previous frame
   toward the next using RAFT optical flow before diffusion. Reduces
   background flicker.
2. **Latent blending**: in the overlap region, blend latents (not pixels)
   before decoding — smoother than pixel-space crossfade.
3. **Consistent noise seed**: use the same initial noise seed for all windows
   of the same clip. Prevents random background variation.
4. **Background lock**: generate a static background frame once, then
   composite the foreground (signer) onto it using a body segmentation mask
   (SAM or simple depth-based matting).

---

## 8. Integration with existing pipeline (minimal changes)

### What does NOT change

- `mosl/` package — untouched
- `scripts/generate_openpose/` — untouched
- `data/` structure — untouched
- `docker/Dockerfile` — extended, not replaced
- `outputs/` structure — new subdirs added

### What is added

```
scripts/avatar/
├── pose_to_controlnet_map.py   # keypoints (NPZ/procedural) → RGB pose-map images
├── video_to_pose_maps.py       # skeleton MP4 / JSON dir → RGB pose-map images (NEW)
├── generate_avatar_video.py    # AnimateDiff / SVD generation driver
├── identity_encoder.py         # IP-Adapter / InstantID face embedding
├── postprocess.py              # RIFE interpolation + Real-ESRGAN upscale (NEW)
└── video_to_avatar.py          # end-to-end pipeline runner (NEW)

outputs/
└── avatar/
    ├── pose_maps/              # RGB pose-map PNGs (ControlNet input)
    ├── frames/                 # per-frame generated JPEG
    └── videos/                 # final MP4 avatar videos

docker/
└── Dockerfile.diffusion        # extended image with diffusers + RIFE + Real-ESRGAN
```

### New input paths

```
Existing asset                     New script                  Output
──────────────────────────────────────────────────────────────────────────
outputs/videos/skeleton/*.mp4      video_to_pose_maps.py       outputs/avatar/pose_maps/<name>/
outputs/openpose_json/*_keypoints/ video_to_pose_maps.py       outputs/avatar/pose_maps/<name>/
                                          │
                                          ▼
                                   video_to_avatar.py  (orchestrates all 3 stages)
                                          │
                                          ▼
                                   outputs/avatar/videos/<name>_avatar.mp4
```

### Integration points

```
Existing output                    New script                  New output
─────────────────────────────────────────────────────────────────────────
outputs/frames/{motion}/           pose_to_controlnet_map.py   outputs/avatar/pose_maps/{motion}/
data/processed/keypoints_2d/       pose_to_controlnet_map.py   outputs/avatar/pose_maps/{clip}/
predictions/*.npz                  pose_to_controlnet_map.py   outputs/avatar/pose_maps/{text}/
                                          │
                                          ▼
                                   generate_avatar_video.py
                                          │
                                          ▼
                                   outputs/avatar/videos/{name}.mp4
```

---

## 9. Step-by-step integration plan

### Step 1 — Install diffusion stack (1 hour)

Extend `docker/Dockerfile.diffusion`:
```dockerfile
RUN pip install --no-cache-dir \
    diffusers==0.30.0 \
    transformers==4.44.0 \
    accelerate==0.33.0 \
    controlnet-aux==0.0.9 \
    insightface \
    onnxruntime-gpu \
    imageio[ffmpeg] \
    einops \
    omegaconf
```

Download model weights (one-time, ~15 GB):
```bash
python scripts/avatar/download_models.py
```

### Step 2 — Pose map generation (2 hours)

Run `pose_to_controlnet_map.py` on existing outputs:
```bash
# From procedural frames (no GPU needed)
python scripts/avatar/pose_to_controlnet_map.py \
    --source procedural --motion walking

# From real MoSL clip NPZ
python scripts/avatar/pose_to_controlnet_map.py \
    --source npz --clip data/processed/keypoints_2d/Pronouns/أَنَا.npz

# From SignLLM prediction
python scripts/avatar/pose_to_controlnet_map.py \
    --source prediction --npz predictions/baseline_mse_text.npz
```

### Step 3 — Identity reference (30 minutes)

Prepare one reference portrait:
```bash
# Use any portrait image (512×512 minimum)
cp /path/to/avatar_portrait.jpg assets/avatar_reference.jpg

# Or generate one with SDXL (no identity constraint)
python scripts/avatar/generate_avatar_video.py \
    --mode reference-only \
    --prompt "a Moroccan man in his 30s, neutral expression, studio lighting"
```

### Step 4 — First avatar video (test run)

```bash
# Quick test: 16 frames, procedural walking motion
python scripts/avatar/generate_avatar_video.py \
    --pose-maps outputs/avatar/pose_maps/walking/ \
    --reference assets/avatar_reference.jpg \
    --prompt "a Moroccan man performing sign language, studio background" \
    --n-frames 16 \
    --out outputs/avatar/videos/walking_test.mp4
```

### Step 5 — Full MoSL sign generation

```bash
# Generate avatar video for a SignLLM prediction
python scripts/avatar/generate_avatar_video.py \
    --pose-maps outputs/avatar/pose_maps/baseline_mse_ana/ \
    --reference assets/avatar_reference.jpg \
    --prompt "a Moroccan signer performing the sign for أَنَا, clean background" \
    --n-frames 0 \   # 0 = use all available pose maps
    --out outputs/avatar/videos/ana_avatar.mp4
```

---

## 10. Performance optimization for DGX Spark GB10

### Memory

```python
# Enable attention slicing (reduces peak VRAM ~30%)
pipeline.enable_attention_slicing(1)

# xFormers memory-efficient attention (if installed)
pipeline.enable_xformers_memory_efficient_attention()

# VAE tiling for high-res decode
pipeline.enable_vae_tiling()

# CPU offload for identity encoder (runs once, not per frame)
pipeline.enable_model_cpu_offload()
```

### Speed

```python
# Compile UNet with torch.compile (PyTorch 2.x, ~2× speedup after warmup)
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead")

# Use float16 throughout (GB10 has fast fp16 tensor cores)
pipeline = pipeline.to(torch.float16)

# Batch multiple frames through ControlNet simultaneously
# (AnimateDiff already does this for the window)
batch_size = 4   # tune to VRAM
```

### Throughput estimates (DGX Spark GB10)

| Configuration | Frames/sec | Time for 100-frame clip |
|---|---|---|
| SDXL + ControlNet, 25 steps, fp16 | ~0.8 | ~2 min |
| SDXL + AnimateDiff 16-frame window | ~0.5 | ~3.5 min |
| + Real-ESRGAN ×2 upscale | ~0.3 | ~6 min |
| FLUX.1-dev + ControlNet | ~0.15 | ~11 min |

---

## 11. Prompt engineering for MoSL avatar

### Base prompt template

```
"a [GENDER] Moroccan sign language interpreter, [AGE], performing the sign
for '[ARABIC_WORD]', clean studio background, soft diffused lighting,
high detail, photorealistic, 8k"
```

### Negative prompt

```
"blurry, deformed hands, extra fingers, missing fingers, distorted face,
low quality, cartoon, anime, painting, watermark, text, logo,
multiple people, crowd"
```

### Per-motion prompt adjustments

| Motion | Additional positive | Additional negative |
|---|---|---|
| Walking | "full body visible, natural gait" | "sitting, static" |
| Hand signs | "hands in focus, expressive fingers" | "hands behind back" |
| Face expressions | "expressive face, eye contact" | "face obscured" |

---

## 12. Known limitations and mitigations

| Limitation | Mitigation |
|---|---|
| AnimateDiff struggles with fine finger detail | Use ControlNet hand conditioning at scale 0.9 for hand region |
| Face drift across long clips | Anchor frame injection every 16 frames |
| SignLLM poses are mean-like (see RESULTS.md) | Use real NPZ keypoints from `data/processed/` instead of predictions |
| No face keypoints in NPZ | Use 5-point simplified face from rig.py or MediaPipe face mesh |
| 30fps vs 25fps clips | Normalize all pose maps to 25fps before generation |
| SDXL anatomy errors on extreme poses | Increase ControlNet scale to 0.85 for unusual poses |

---

## References

- AnimateDiff: Guo et al. (2023). arXiv:2307.04725
- ControlNet: Zhang et al. (2023). arXiv:2302.05543
- IP-Adapter: Ye et al. (2023). arXiv:2308.06721
- InstantID: Wang et al. (2024). arXiv:2401.07519
- Stable Video Diffusion: Blattmann et al. (2023). arXiv:2311.15127
- FLUX.1: Black Forest Labs (2024)
- Real-ESRGAN: Wang et al. (2021). arXiv:2107.10833
- RIFE: Huang et al. (2022). arXiv:2011.06294
