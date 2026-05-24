"""Generate a realistic avatar video from OpenPose conditioning maps.

Takes the pose-map PNGs produced by pose_to_controlnet_map.py and a reference
portrait, then runs AnimateDiff + ControlNet OpenPose + IP-Adapter to produce
a temporally coherent avatar video.

Two generation modes:
  --mode animatediff   AnimateDiff v3 + SDXL + ControlNet (default, best quality)
  --mode svd           Stable Video Diffusion XT (faster, ≤ 25 frames)

Usage:
    # Quick test (16 frames, procedural walking)
    python scripts/avatar/generate_avatar_video.py \\
        --pose-maps outputs/avatar/pose_maps/walking/ \\
        --reference assets/avatar_reference.jpg \\
        --prompt "a Moroccan man performing sign language, studio background" \\
        --n-frames 16 \\
        --out outputs/avatar/videos/walking_test.mp4

    # Full MoSL sign from SignLLM prediction
    python scripts/avatar/generate_avatar_video.py \\
        --pose-maps outputs/avatar/pose_maps/baseline_mse_text/ \\
        --reference assets/avatar_reference.jpg \\
        --prompt "a Moroccan signer performing a sign, clean background" \\
        --out outputs/avatar/videos/sign_avatar.mp4

    # Use pre-encoded identity embedding (faster on repeated runs)
    python scripts/avatar/generate_avatar_video.py \\
        --pose-maps outputs/avatar/pose_maps/walking/ \\
        --embedding assets/avatar_reference.embedding.pt \\
        --prompt "..." \\
        --out outputs/avatar/videos/walking.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "avatar"))

# ── Default generation settings (see docs/AI_AVATAR_PIPELINE.md §5) ──────

DEFAULTS = {
    # AnimateDiff
    "window_size":     16,
    "window_overlap":  4,
    "num_steps":       25,
    "guidance_scale":  7.5,
    "controlnet_scale": 0.75,
    "ip_adapter_scale": 0.65,
    "width":           512,
    "height":          768,
    "fps_out":         25,
    # SVD
    "motion_bucket_id": 110,
    "noise_aug":        0.02,
    "decode_chunk":     8,
}

NEGATIVE_PROMPT = (
    "blurry, deformed hands, extra fingers, missing fingers, distorted face, "
    "low quality, cartoon, anime, painting, watermark, text, logo, "
    "multiple people, crowd, nsfw"
)


# ── Pipeline builders ─────────────────────────────────────────────────────

def build_animatediff_pipeline(device: str, dtype: torch.dtype):
    """Build SDXL + AnimateDiff + ControlNet OpenPose + IP-Adapter pipeline."""
    from diffusers import (
        AnimateDiffSDXLPipeline,
        MotionAdapter,
        DDIMScheduler,
        ControlNetModel,
    )

    print("[avatar] loading ControlNet OpenPose (SDXL)...")
    controlnet = ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0",
        torch_dtype=dtype,
    )

    print("[avatar] loading AnimateDiff motion adapter...")
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-sdxl-beta",
        torch_dtype=dtype,
    )

    print("[avatar] loading SDXL base + AnimateDiff pipeline...")
    pipe = AnimateDiffSDXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        motion_adapter=adapter,
        controlnet=controlnet,
        torch_dtype=dtype,
        variant="fp16",
    )

    # Noam scheduler — DDIM for deterministic, reproducible generation
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # IP-Adapter for identity
    print("[avatar] loading IP-Adapter face (SDXL)...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus-face_sdxl_vit-h.bin",
    )

    # Memory optimizations for DGX Spark GB10
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_tiling()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass  # xformers optional

    pipe = pipe.to(device)
    return pipe


def build_svd_pipeline(device: str, dtype: torch.dtype):
    """Build Stable Video Diffusion XT pipeline (short clips ≤ 25 frames)."""
    from diffusers import StableVideoDiffusionPipeline

    print("[avatar] loading Stable Video Diffusion XT...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=dtype,
        variant="fp16",
    )
    pipe.enable_model_cpu_offload()
    return pipe


# ── Pose map loading ──────────────────────────────────────────────────────

def load_pose_maps(pose_dir: Path, n_frames: int, width: int, height: int
                   ) -> list[Image.Image]:
    """Load PNG pose maps from a directory, sorted by filename."""
    pngs = sorted(pose_dir.glob("*_pose.png"))
    if not pngs:
        raise FileNotFoundError(
            f"No *_pose.png files found in {pose_dir}. "
            "Run pose_to_controlnet_map.py first."
        )
    if n_frames > 0:
        pngs = pngs[:n_frames]
    images = []
    for p in pngs:
        img = Image.open(p).convert("RGB").resize((width, height), Image.LANCZOS)
        images.append(img)
    print(f"[avatar] loaded {len(images)} pose maps from {pose_dir}")
    return images


# ── Sliding window AnimateDiff generation ────────────────────────────────

def generate_animatediff(
    pipe,
    pose_maps: list[Image.Image],
    prompt: str,
    negative_prompt: str,
    identity_embedding: Optional[dict],
    cfg: dict,
    seed: int,
) -> list[Image.Image]:
    """Generate frames using AnimateDiff with sliding window for long clips.

    For clips longer than window_size, uses overlapping windows with
    linear crossfade blending in the overlap region.
    """
    W = cfg["window_size"]
    O = cfg["window_overlap"]
    step = W - O
    T = len(pose_maps)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # Set IP-Adapter scale
    if identity_embedding is not None:
        pipe.set_ip_adapter_scale(cfg["ip_adapter_scale"])
        ip_embeds = identity_embedding.get("image_embeds")
        if ip_embeds is not None:
            ip_embeds = ip_embeds.to(pipe.device, dtype=torch.float16)
    else:
        ip_embeds = None

    all_frames: list[Optional[Image.Image]] = [None] * T
    weights = np.zeros(T, dtype=np.float32)

    window_starts = list(range(0, max(T - W + 1, 1), step))
    # Ensure the last window covers the end
    if window_starts[-1] + W < T:
        window_starts.append(T - W)

    for wi, start in enumerate(window_starts):
        end = min(start + W, T)
        window_poses = pose_maps[start:end]
        n = len(window_poses)

        print(f"[avatar] window {wi+1}/{len(window_starts)}: frames {start}–{end-1}")

        # Reduce guidance on non-first windows for consistency
        gs = cfg["guidance_scale"] if wi == 0 else max(cfg["guidance_scale"] - 2.0, 5.0)

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=window_poses,                     # ControlNet conditioning
            num_inference_steps=cfg["num_steps"],
            guidance_scale=gs,
            controlnet_conditioning_scale=cfg["controlnet_scale"],
            generator=generator,
            width=cfg["width"],
            height=cfg["height"],
            num_frames=n,
        )
        if ip_embeds is not None:
            kwargs["ip_adapter_image_embeds"] = [ip_embeds]

        result = pipe(**kwargs)
        window_frames = result.frames[0]            # list of PIL Images

        # Accumulate with linear blend weights
        for local_i, frame in enumerate(window_frames):
            global_i = start + local_i
            if global_i >= T:
                break
            # Weight: ramp up at start of window, ramp down at end
            w_start = min(local_i / max(O, 1), 1.0)
            w_end   = min((n - 1 - local_i) / max(O, 1), 1.0)
            w = min(w_start, w_end)

            if all_frames[global_i] is None:
                all_frames[global_i] = frame
                weights[global_i] = w
            else:
                # Blend with existing frame
                existing = np.array(all_frames[global_i], dtype=np.float32)
                new_f    = np.array(frame, dtype=np.float32)
                prev_w   = weights[global_i]
                blended  = (existing * prev_w + new_f * w) / (prev_w + w + 1e-8)
                all_frames[global_i] = Image.fromarray(blended.clip(0, 255).astype(np.uint8))
                weights[global_i] = prev_w + w

    # Fill any remaining None frames (shouldn't happen, but defensive)
    for i in range(T):
        if all_frames[i] is None:
            all_frames[i] = pose_maps[i].convert("RGB")

    return all_frames  # type: ignore[return-value]


def generate_svd(
    pipe,
    anchor_frame: Image.Image,
    n_frames: int,
    cfg: dict,
    seed: int,
) -> list[Image.Image]:
    """Generate frames using Stable Video Diffusion from an anchor frame."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        image=anchor_frame,
        num_frames=min(n_frames, 25),
        motion_bucket_id=cfg["motion_bucket_id"],
        noise_aug_strength=cfg["noise_aug"],
        decode_chunk_size=cfg["decode_chunk"],
        generator=generator,
    )
    return result.frames[0]


# ── Video export ──────────────────────────────────────────────────────────

def frames_to_video(frames: list[Image.Image], out_path: Path, fps: int) -> None:
    """Write PIL frames to MP4 using imageio (ffmpeg backend)."""
    try:
        import imageio
    except ImportError:
        raise ImportError("imageio required: pip install imageio[ffmpeg]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p")
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()
    print(f"[avatar] wrote {len(frames)} frames @ {fps}fps → {out_path}")


# ── Optional post-processing ──────────────────────────────────────────────

def upscale_frames(frames: list[Image.Image], scale: int = 2) -> list[Image.Image]:
    """Upscale frames with Real-ESRGAN (if available)."""
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        print("[avatar] Real-ESRGAN not installed — skipping upscale")
        return frames

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=scale)
    upsampler = RealESRGANer(
        scale=scale,
        model_path=f"https://github.com/xinntao/Real-ESRGAN/releases/download/"
                   f"v0.1.0/RealESRGAN_x{scale}plus.pth",
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=True,
    )
    import cv2
    out = []
    for i, frame in enumerate(frames):
        img_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        enhanced, _ = upsampler.enhance(img_bgr, outscale=scale)
        out.append(Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)))
        if (i + 1) % 10 == 0:
            print(f"[avatar] upscaled {i+1}/{len(frames)}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pose-maps", required=True,
                   help="Directory of *_pose.png files from pose_to_controlnet_map.py")
    p.add_argument("--reference", default=None,
                   help="Reference portrait image for identity (JPG/PNG)")
    p.add_argument("--embedding", default=None,
                   help="Pre-encoded identity embedding (.pt or .npz)")
    p.add_argument("--prompt", required=True,
                   help="Text prompt describing the avatar and scene")
    p.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    p.add_argument("--out", required=True, help="Output MP4 path")
    p.add_argument("--mode", default="animatediff",
                   choices=["animatediff", "svd"],
                   help="Generation mode")
    p.add_argument("--n-frames", type=int, default=0,
                   help="Number of frames to generate (0 = all pose maps)")
    p.add_argument("--width",  type=int, default=DEFAULTS["width"])
    p.add_argument("--height", type=int, default=DEFAULTS["height"])
    p.add_argument("--fps",    type=int, default=DEFAULTS["fps_out"])
    p.add_argument("--steps",  type=int, default=DEFAULTS["num_steps"])
    p.add_argument("--cfg",    type=float, default=DEFAULTS["guidance_scale"],
                   dest="guidance_scale")
    p.add_argument("--controlnet-scale", type=float,
                   default=DEFAULTS["controlnet_scale"])
    p.add_argument("--ip-scale", type=float, default=DEFAULTS["ip_adapter_scale"])
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--upscale", action="store_true",
                   help="Apply Real-ESRGAN ×2 upscale after generation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype",  default="float16", choices=["float16", "bfloat16", "float32"])
    args = p.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    cfg = {
        "window_size":     DEFAULTS["window_size"],
        "window_overlap":  DEFAULTS["window_overlap"],
        "num_steps":       args.steps,
        "guidance_scale":  args.guidance_scale,
        "controlnet_scale": args.controlnet_scale,
        "ip_adapter_scale": args.ip_scale,
        "width":           args.width,
        "height":          args.height,
        "motion_bucket_id": DEFAULTS["motion_bucket_id"],
        "noise_aug":        DEFAULTS["noise_aug"],
        "decode_chunk":     DEFAULTS["decode_chunk"],
    }

    # Load pose maps
    pose_maps = load_pose_maps(
        Path(args.pose_maps), args.n_frames, args.width, args.height
    )

    # Load identity embedding
    identity_embedding = None
    if args.embedding:
        from identity_encoder import IdentityEncoder
        identity_embedding = IdentityEncoder.load(args.embedding)
        print(f"[avatar] loaded embedding from {args.embedding}")
    elif args.reference:
        from identity_encoder import IdentityEncoder
        enc = IdentityEncoder(method="ip_adapter", device=args.device)
        identity_embedding = enc.encode(args.reference)
        print(f"[avatar] encoded identity from {args.reference}")
    else:
        print("[avatar] warning: no --reference or --embedding provided — "
              "identity will not be preserved across frames")

    # Generate
    if args.mode == "animatediff":
        print(f"[avatar] building AnimateDiff pipeline on {args.device} ({args.dtype})...")
        pipe = build_animatediff_pipeline(args.device, dtype)
        frames = generate_animatediff(
            pipe, pose_maps, args.prompt, args.negative_prompt,
            identity_embedding, cfg, args.seed,
        )
    else:  # svd
        print(f"[avatar] building SVD pipeline on {args.device} ({args.dtype})...")
        pipe = build_svd_pipeline(args.device, dtype)
        # SVD needs an anchor frame — use the first pose map rendered through SDXL
        # For simplicity, use the first pose map as the anchor directly
        anchor = pose_maps[0]
        frames = generate_svd(pipe, anchor, len(pose_maps), cfg, args.seed)

    # Optional upscale
    if args.upscale:
        print("[avatar] upscaling with Real-ESRGAN ×2...")
        frames = upscale_frames(frames, scale=2)

    # Export
    frames_to_video(frames, Path(args.out), args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
