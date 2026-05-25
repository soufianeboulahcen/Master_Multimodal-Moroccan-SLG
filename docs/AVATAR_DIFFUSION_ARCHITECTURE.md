# Cinematic Avatar Diffusion Architecture

Full design for converting the existing OpenPose motion video dataset into a
production-quality AI avatar video system.

This document is grounded in the actual project state: 7 video types per clip,
460×460 source resolution, 25 fps, COCO-18 body + 21-point hands, MediaPipe
Holistic keypoints, and the existing `scripts/avatar/` pipeline.

---

## 1. What each existing video type actually is

Understanding what each render type encodes determines how it should be used
as a conditioning signal — not all of them are useful for diffusion.

| Video type | What it encodes | Useful for diffusion? | Role |
|---|---|---|---|
| `skeleton` | COCO-18 body + 21-pt hands on black BG, DWPose palette | ✅ Primary | ControlNet OpenPose conditioning |
| `overlay` | Skeleton drawn over synthetic dark background | ✅ Secondary | Depth/structure reference |
| `heatmap` | Per-joint Gaussian confidence maps | ✅ Auxiliary | Soft pose conditioning (ControlNet depth variant) |
| `neon` | Skeleton with glow/bloom effect | ❌ Decorative | Not used — same topology as skeleton, worse signal |
| `mosaic` | Tiled multi-view composite | ❌ Decorative | Not used — spatial layout is wrong for conditioning |
| `slowmo` | Same skeleton at 10 fps playback | ⚠️ Optional | Use for long-clip generation where motion is slow |
| `studio` | Skeleton on animated gradient background | ❌ Decorative | Not used — background bleeds into conditioning |

**Key insight**: only `skeleton` and `heatmap` carry conditioning signal.
`overlay` is useful as a structural reference for the first-frame anchor.
The other four types are visual effects — they add noise to the conditioning.

---

## 2. Full system architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                           INPUT LAYER                                        ║
║                                                                              ║
║  outputs/videos/skeleton/<clip>_skeleton.mp4   (primary motion signal)      ║
║  outputs/videos/heatmap/<clip>_heatmap.mp4     (soft confidence maps)       ║
║  outputs/videos/overlay/<clip>_overlay.mp4     (structural reference)       ║
║                                                                              ║
║  OR  data/processed/keypoints_2d/<cat>/<clip>.npz  (raw keypoints, best)    ║
║  OR  predictions/<run>.npz                         (SignLLM generated pose) ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
         │  skeleton    │  │  heatmap     │  │  overlay     │
         │  frames      │  │  frames      │  │  frame[0]    │
         │  (per frame) │  │  (per frame) │  │  (anchor)    │
         └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                │                 │                  │
                ▼                 ▼                  ▼
    ┌───────────────────┐  ┌────────────┐  ┌─────────────────┐
    │ video_to_pose_    │  │ heatmap_   │  │ identity_       │
    │ maps.py           │  │ to_depth_  │  │ encoder.py      │
    │                   │  │ map.py     │  │ (IP-Adapter or  │
    │ DWPose-palette    │  │            │  │  InstantID)     │
    │ RGB pose PNGs     │  │ grayscale  │  │                 │
    │ 512×768           │  │ depth PNGs │  │ face embedding  │
    └──────────┬────────┘  └─────┬──────┘  └────────┬────────┘
               │                 │                   │
               └────────┬────────┘                   │
                         │                           │
                         ▼                           │
╔════════════════════════════════════════════════════╪════════════════════════╗
║              DUAL-CONTROLNET CONDITIONING          │                        ║
║                                                    │                        ║
║  ControlNet 1: OpenPose (SDXL)                     │                        ║
║    input  : DWPose RGB pose maps                   │                        ║
║    weight : 0.85–0.90  (strict body structure)     │                        ║
║    model  : thibaud/controlnet-openpose-sdxl-1.0   │                        ║
║                                                    │                        ║
║  ControlNet 2: Depth (SDXL)                        │                        ║
║    input  : heatmap-derived depth maps             │                        ║
║    weight : 0.30–0.45  (soft spatial grounding)    │                        ║
║    model  : diffusers/controlnet-depth-sdxl-1.0    │                        ║
║                                                    │                        ║
╚════════════════════════════════════════════════════╪════════════════════════╝
                         │                           │
                         ▼                           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ANIMATEDIFF VIDEO GENERATION                              ║
║                                                                              ║
║  Base UNet  : SDXL 1.0  (stabilityai/stable-diffusion-xl-base-1.0)          ║
║  Motion     : AnimateDiff v3 SDXL beta                                       ║
║               (guoyww/animatediff-motion-adapter-sdxl-beta)                  ║
║  Scheduler  : DDIM (deterministic, reproducible)                             ║
║  Identity   : IP-Adapter SDXL face  (scale 0.60–0.70)                       ║
║               OR InstantID          (scale 0.80, higher fidelity)            ║
║                                                                              ║
║  Sliding window: 16 frames, 4-frame overlap, linear crossfade blend          ║
║  CFG scale  : 7.0                                                            ║
║  Steps      : 25 (DDIM)                                                      ║
║  Resolution : 512×768 (portrait) or 768×512 (landscape)                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                         POST-PROCESSING                                      ║
║                                                                              ║
║  1. EMA flicker reduction   (alpha=0.12, CPU, no deps)                       ║
║  2. RIFE v4.6 interpolation (8 fps → 24 fps, 2× passes)                     ║
║  3. Real-ESRGAN ×2 upscale  (512×768 → 1024×1536)                           ║
║  4. FFmpeg H.264 encode     (yuv420p, CRF 18, preset slow)                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
                    outputs/avatar/videos/<clip>_cinematic.mp4
                    1024×1536 · 24 fps · H.264 · ~15–30 MB/clip
```

---

## 3. Mapping each video type to its conditioning role

### 3.1 Skeleton → ControlNet OpenPose (primary, weight 0.85–0.90)

The skeleton video is the **only** input that carries precise joint topology
in the DWPose colour convention that ControlNet OpenPose was trained on.

Conversion path:
```
skeleton MP4  →  video_to_pose_maps.py --source video
             →  outputs/avatar/pose_maps/<clip>/XXXXXX_pose.png
             →  ControlNet OpenPose conditioning at every frame
```

If per-frame JSON keypoints exist (preferred), use `--source json` instead —
this re-renders clean DWPose maps without JPEG compression artefacts.

### 3.2 Heatmap → ControlNet Depth (auxiliary, weight 0.30–0.45)

The heatmap video encodes per-joint Gaussian confidence blobs. When converted
to a single-channel grayscale image (max across joints), it approximates a
rough depth/saliency map: bright = high-confidence joint region, dark = empty
space. This gives the depth ControlNet a soft spatial prior that reinforces
the body silhouette without over-constraining the appearance.

Conversion path:
```
heatmap MP4  →  multi_controlnet_conditioning.py --extract-depth
             →  outputs/avatar/depth_maps/<clip>/XXXXXX_depth.png
             →  ControlNet Depth conditioning (secondary)
```

Weight must stay below 0.45 — higher values cause the depth map's blob
structure to appear as visible artefacts in the generated skin.

### 3.3 Overlay frame[0] → Identity anchor

The first frame of the overlay video shows the skeleton on a dark background
with the correct body proportions and spatial position. It is used as the
**anchor frame** for the SVD path and as a structural reference for the
IP-Adapter face crop region.

### 3.4 Slowmo → Long-clip generation

For clips where the motion is slow (sign language holds, static poses), the
slowmo version (10 fps playback of 30 fps content) can be used as input to
generate longer avatar clips without repeating frames. The pipeline
automatically detects slowmo input via the `_slowmo` filename suffix.

---

## 4. Model stack recommendation

### Primary stack — best quality (VRAM ≥ 16 GB)

| Component | Model | HuggingFace ID | VRAM |
|---|---|---|---|
| Base UNet | SDXL 1.0 | `stabilityai/stable-diffusion-xl-base-1.0` | 8 GB |
| Motion | AnimateDiff v3 SDXL | `guoyww/animatediff-motion-adapter-sdxl-beta` | +1 GB |
| Pose ControlNet | ControlNet OpenPose SDXL | `thibaud/controlnet-openpose-sdxl-1.0` | +2 GB |
| Depth ControlNet | ControlNet Depth SDXL | `diffusers/controlnet-depth-sdxl-1.0` | +2 GB |
| Identity | IP-Adapter SDXL face | `h94/IP-Adapter` → `ip-adapter-plus-face_sdxl_vit-h.bin` | +1 GB |
| Upscaler | Real-ESRGAN ×2 | auto-downloaded | 0.5 GB |
| Interpolation | RIFE v4.6 | `models/rife/` | 0.3 GB |
| **Total peak** | | | **~15 GB** |

### High-fidelity identity stack (VRAM ≥ 24 GB)

Replace IP-Adapter with InstantID for recognizable real-person identity:

| Component | Model | Notes |
|---|---|---|
| Identity | InstantID | `InstantX/InstantID` — requires InsightFace |
| Face ControlNet | IdentityNet | Bundled with InstantID |
| Base | SDXL 1.0 | Same as primary |
| Motion | AnimateDiff v3 SDXL | Same as primary |
| Pose ControlNet | OpenPose SDXL | Same as primary |

With InstantID, three ControlNets run simultaneously:
- OpenPose (body structure, weight 0.85)
- Depth (spatial grounding, weight 0.35)
- IdentityNet (face fidelity, weight 0.80)

### Lightweight fallback (VRAM < 8 GB or CPU)

| Component | Model | Notes |
|---|---|---|
| Base | SD 1.5 | `runwayml/stable-diffusion-v1-5` |
| Motion | AnimateDiff v2 | `guoyww/animatediff-motion-adapter-v1-5-2` |
| Pose ControlNet | OpenPose SD1.5 | `lllyasviel/control_v11p_sd15_openpose` |
| Identity | IP-Adapter SD1.5 | `h94/IP-Adapter` → `ip-adapter-plus-face_sd15.bin` |

---

## 5. Inference settings

### AnimateDiff + dual ControlNet

```python
# Window
window_size     = 16        # frames per diffusion pass
window_overlap  = 4         # overlap for temporal stitching
fps_native      = 8         # AnimateDiff internal fps (interpolate to 24 after)

# Diffusion
num_inference_steps = 25    # DDIM
guidance_scale      = 7.0   # CFG
denoise_strength    = 0.7   # img2img strength (when using anchor frame)

# ControlNet
controlnet_pose_scale  = 0.85   # primary — strict body structure
controlnet_depth_scale = 0.35   # secondary — soft spatial grounding

# Identity
ip_adapter_scale = 0.65         # IP-Adapter face
# OR
instantid_scale  = 0.80         # InstantID (higher fidelity)

# Resolution
width  = 512
height = 768    # portrait (full body, sign language)
```

### Stable Video Diffusion (short clips ≤ 25 frames)

```python
num_frames          = 25
motion_bucket_id    = 110   # sign language ≈ 110 (100=slow, 127=fast)
fps                 = 7     # SVD native fps
decode_chunk_size   = 8
noise_aug_strength  = 0.02  # low = faithful to anchor frame
```

### Post-processing

```python
# Flicker reduction (before interpolation)
ema_alpha = 0.12            # blend weight of previous frame

# RIFE interpolation
source_fps  = 8             # AnimateDiff native
target_fps  = 24            # output
passes      = 2             # 8→16→24 (two 2× passes, then resample to 24)

# Real-ESRGAN
upscale_factor = 2          # 512×768 → 1024×1536
tile_size      = 512        # VRAM-safe tiling

# FFmpeg final encode
codec   = "libx264"
crf     = 18                # visually lossless
preset  = "slow"            # better compression
pix_fmt = "yuv420p"         # maximum compatibility
```

---

## 6. Temporal consistency strategy

### Problem
AnimateDiff processes 16-frame windows. Naive concatenation produces visible
seams: lighting jumps, face drift, background flicker.

### Solution: three-layer stabilization

**Layer 1 — Consistent noise seed**
Use the same initial noise tensor for all windows of the same clip. This
prevents random background variation between windows.

```python
generator = torch.Generator(device=pipe.device).manual_seed(seed)
# Reuse the same generator for every window — do NOT re-seed between windows
```

**Layer 2 — Sliding window with linear crossfade**
```
Window 1: frames  0–15  (guidance=7.0)
Window 2: frames 12–27  (guidance=5.0, reduced to stay close to window 1)
  overlap region 12–15: pixel-space linear blend
  weight = ramp from 0→1 over the 4-frame overlap
Window 3: frames 24–39  (same pattern)
```
Implemented in `generate_avatar_video.py::generate_animatediff()`.

**Layer 3 — EMA flicker reduction (post-generation)**
```python
# In postprocess.py::reduce_flicker()
blended[t] = prev[t-1] * 0.12 + current[t] * 0.88
```
Applied before RIFE interpolation so the interpolator sees smooth input.

**Layer 4 — Background lock (advanced)**
For clips with a static background, generate the background once and
composite the foreground signer using a body segmentation mask (SAM or
MediaPipe selfie segmentation). Eliminates all background flicker by
construction.

```python
# Planned addition to postprocess.py
background = generate_static_background(prompt, seed)
for frame in frames:
    mask = segment_foreground(frame)   # MediaPipe or SAM
    composite = background * (1 - mask) + frame * mask
```

---

## 7. Identity preservation strategy

### Option A — IP-Adapter (default, simpler)

1. Provide one reference portrait of the target avatar
2. Extract CLIP ViT-L/14 image embedding (257 patch tokens × 1024 dims)
3. Inject into UNet cross-attention via IP-Adapter projection layers
4. Same embedding reused for every frame and every window → consistent identity

```python
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                     weight_name="ip-adapter-plus-face_sdxl_vit-h.bin")
pipe.set_ip_adapter_scale(0.65)
```

Scale tuning:
- `0.50` — subtle identity, more creative freedom
- `0.65` — balanced (recommended)
- `0.80` — strong identity, may fight pose conditioning

### Option B — InstantID (higher fidelity)

1. Run InsightFace ArcFace on reference portrait → 512-dim face embedding
2. Extract 5-point facial keypoints → IdentityNet ControlNet conditioning
3. Three simultaneous ControlNets: OpenPose + Depth + IdentityNet
4. Face embedding injected via IP-Adapter projection (same mechanism)

InstantID is preferred when the avatar must be recognizable as a specific
real person. IP-Adapter is sufficient for a generic consistent avatar.

### Temporal face drift mitigation

The main risk is face drift accumulating across windows. Mitigations:
- Pass the same face embedding to every window (already implemented)
- Reduce CFG on non-first windows (`guidance_scale - 2.0`, min 5.0)
- Use the first generated frame as an anchor for subsequent windows
  (DDIM inversion of frame 0 → noisy latent → condition window 2 start)

---

## 8. GPU efficiency strategy

### Memory

| Technique | VRAM saving | Trade-off |
|---|---|---|
| `fp16` weights | ~50% | Negligible quality loss |
| Attention slicing (`slice_size=1`) | ~20% | ~10% slower |
| VAE tiling | ~30% for large res | Slight seam at tile boundaries |
| xformers memory-efficient attention | ~30% | Requires xformers install |
| CPU offload (`enable_model_cpu_offload`) | ~60% | 3–5× slower |
| Sequential CPU offload | ~80% | 8–10× slower |

Recommended for GB10 (121 GB unified memory): fp16 + attention slicing +
xformers. No CPU offload needed.

### Throughput

| Optimization | Speedup |
|---|---|
| Pre-encode identity embedding once, reuse | 1× per clip saved |
| Batch pose maps (load all before generation) | Eliminates I/O stalls |
| DDIM 25 steps vs 50 | 2× faster, minimal quality loss |
| Compile UNet with `torch.compile` (PyTorch 2.x) | ~20% faster |
| Process clips in parallel (multi-GPU) | Linear scaling |

---

## 9. Step-by-step integration plan

### Step 1 — Install dual-ControlNet stack (30 min)

```bash
docker build -t pfe-avatar:latest -f docker/Dockerfile.diffusion .
docker/run_avatar.sh python scripts/avatar/download_models.py
# Also download depth ControlNet:
docker/run_avatar.sh python -c "
from huggingface_hub import snapshot_download
snapshot_download('diffusers/controlnet-depth-sdxl-1.0')
"
```

### Step 2 — Extract conditioning maps from all video types (10 min)

```bash
# Pose maps from skeleton videos (all clips)
docker/run_avatar.sh python scripts/avatar/video_to_pose_maps.py \
    --source video --input outputs/videos/skeleton/ --batch

# Depth maps from heatmap videos (all clips)
docker/run_avatar.sh python scripts/avatar/multi_controlnet_conditioning.py \
    --extract-depth --input outputs/videos/heatmap/ --batch
```

### Step 3 — Encode identity reference (5 min)

```bash
docker/run_avatar.sh python scripts/avatar/identity_encoder.py \
    assets/avatar_reference.jpg \
    --method ip_adapter \
    --out assets/avatar_reference.embedding.pt
```

### Step 4 — Generate cinematic avatar videos

```bash
# Single clip, full quality
docker/run_avatar.sh python scripts/avatar/generate_cinematic_avatar.py \
    --clip walking \
    --embedding assets/avatar_reference.embedding.pt \
    --prompt "a Moroccan man performing sign language, photorealistic, \
              DSLR 85mm, cinematic lighting, natural skin texture" \
    --out outputs/avatar/videos/walking_cinematic.mp4 \
    --dual-controlnet \
    --interpolate --target-fps 24 \
    --upscale

# Batch all clips
docker/run_avatar.sh python scripts/avatar/generate_cinematic_avatar.py \
    --batch \
    --embedding assets/avatar_reference.embedding.pt \
    --prompt "a Moroccan signer, photorealistic, studio lighting" \
    --out-dir outputs/avatar/videos/cinematic/
```

### Step 5 — Verify output quality

Check for:
- No flickering: compare consecutive frames with `postprocess.py --reduce-flicker`
- Face consistency: crop face region from each frame, compute SSIM across frames
- Pose adherence: overlay generated frames with source skeleton at 50% opacity

---

## 10. Prompt engineering for realism

### Positive prompt structure

```
"[subject], [clothing], [background], [lighting], [camera], [quality tags]"

Example:
"a Moroccan man in his 30s performing sign language,
 wearing a dark blue shirt,
 clean white studio background,
 soft key light from upper left, subtle fill light,
 shot on Canon EOS R5, 85mm f/1.8, shallow depth of field,
 photorealistic, 8K, natural skin texture, detailed hands"
```

### Negative prompt (already in pipeline)

```
"blurry, deformed hands, extra fingers, missing fingers, distorted face,
 low quality, cartoon, anime, painting, watermark, text, logo,
 multiple people, crowd, nsfw, duplicate limbs, temporal artifacts,
 flickering, unrealistic anatomy, fake skin, plastic skin"
```

### Sign-language specific additions

```
# Add to positive prompt for hand clarity:
"detailed hand gestures, clear finger articulation, expressive hands"

# Add to negative prompt for hand quality:
"fused fingers, missing fingers, extra fingers, hand deformation,
 blurry hands, indistinct fingers"
```

---

## 11. File map — what exists vs what is new

```
scripts/avatar/
├── video_to_pose_maps.py          ✅ exists — skeleton/JSON → pose PNGs
├── pose_to_controlnet_map.py      ✅ exists — NPZ/procedural → pose PNGs
├── generate_avatar_video.py       ✅ exists — AnimateDiff + single ControlNet
├── identity_encoder.py            ✅ exists — IP-Adapter / InstantID encoding
├── postprocess.py                 ✅ exists — RIFE + Real-ESRGAN + EMA
├── video_to_avatar.py             ✅ exists — end-to-end orchestrator
├── render_avatar_local.py         ✅ exists — CPU-only stylised renderer
├── multi_controlnet_conditioning.py  🆕 NEW — heatmap→depth + dual-ControlNet prep
└── generate_cinematic_avatar.py      🆕 NEW — dual-ControlNet + InstantID generation
```

---

## References

- AnimateDiff: Guo et al. (2023). arXiv:2307.04725
- ControlNet: Zhang et al. (2023). arXiv:2302.05543
- IP-Adapter: Ye et al. (2023). arXiv:2308.06721
- InstantID: Wang et al. (2024). arXiv:2401.07519
- Stable Video Diffusion: Blattmann et al. (2023). arXiv:2311.15127
- Real-ESRGAN: Wang et al. (2021). arXiv:2107.10833
- RIFE: Huang et al. (2022). arXiv:2011.06294
- DWPose: Yang et al. (2023). arXiv:2307.15880
