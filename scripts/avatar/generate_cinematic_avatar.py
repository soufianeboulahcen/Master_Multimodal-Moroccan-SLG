"""Cinematic avatar video generation — dual ControlNet + InstantID/IP-Adapter.

Upgrades generate_avatar_video.py with:
  - Dual ControlNet: OpenPose (primary, weight 0.85) + Depth (auxiliary, 0.35)
  - InstantID support alongside IP-Adapter
  - Background lock via luminance-based compositing
  - Batch mode over all clips with available pose maps

See docs/AVATAR_DIFFUSION_ARCHITECTURE.md for full design rationale.

Usage:
    # Single clip — dual ControlNet + IP-Adapter
    python scripts/avatar/generate_cinematic_avatar.py \\
        --clip walking \\
        --reference assets/avatar_reference.jpg \\
        --prompt "a Moroccan man performing sign language, photorealistic, \\
                  DSLR 85mm, cinematic lighting, natural skin texture" \\
        --out outputs/avatar/videos/walking_cinematic.mp4 \\
        --dual-controlnet --interpolate --upscale

    # InstantID (higher face fidelity, requires insightface)
    python scripts/avatar/generate_cinematic_avatar.py \\
        --clip walking \\
        --reference assets/avatar_reference.jpg \\
        --prompt "a Moroccan man performing sign language, photorealistic" \\
        --out outputs/avatar/videos/walking_instantid.mp4 \\
        --identity-method instantid --dual-controlnet

    # Batch all clips
    python scripts/avatar/generate_cinematic_avatar.py \\
        --batch \\
        --reference assets/avatar_reference.jpg \\
        --prompt "a Moroccan signer, photorealistic, studio lighting" \\
        --out-dir outputs/avatar/videos/cinematic/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "avatar"))

DEFAULTS = {
    "width":            512,
    "height":           768,
    "fps_out":          24,
    "window_size":      16,
    "window_overlap":   4,
    "num_steps":        25,
    "guidance_scale":   7.0,
    "denoise_strength": 0.7,
    "pose_scale":       0.85,
    "depth_scale":      0.35,
    "ip_adapter_scale": 0.65,
    "instantid_scale":  0.80,
    "motion_bucket_id": 110,
    "noise_aug":        0.02,
    "decode_chunk":     8,
}

NEGATIVE_PROMPT = (
    "blurry, deformed hands, extra fingers, missing fingers, distorted face, "
    "low quality, cartoon, anime, painting, watermark, text, logo, "
    "multiple people, crowd, nsfw, duplicate limbs, temporal artifacts, "
    "flickering, unrealistic anatomy, fake skin, plastic skin, "
    "fused fingers, indistinct fingers, hand deformation"
)


# ── Pipeline builders ─────────────────────────────────────────────────────

def build_dual_controlnet_pipeline(device: str, dtype: torch.dtype):
    """SDXL + AnimateDiff + dual ControlNet (OpenPose + Depth) + IP-Adapter."""
    from diffusers import (
        AnimateDiffSDXLPipeline, MotionAdapter,
        DDIMScheduler, ControlNetModel,
    )
    print("[cinematic] loading ControlNet OpenPose (SDXL)...")
    cn_pose = ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=dtype)
    print("[cinematic] loading ControlNet Depth (SDXL)...")
    cn_depth = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=dtype)
    print("[cinematic] loading AnimateDiff motion adapter...")
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-sdxl-beta", torch_dtype=dtype)
    print("[cinematic] loading SDXL + AnimateDiff pipeline...")
    pipe = AnimateDiffSDXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        motion_adapter=adapter, controlnet=[cn_pose, cn_depth],
        torch_dtype=dtype, variant="fp16",
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    print("[cinematic] loading IP-Adapter face (SDXL)...")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                         weight_name="ip-adapter-plus-face_sdxl_vit-h.bin")
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_tiling()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe.to(device)


def build_single_controlnet_pipeline(device: str, dtype: torch.dtype):
    """Fallback: single ControlNet OpenPose (when depth maps unavailable)."""
    from generate_avatar_video import build_animatediff_pipeline
    return build_animatediff_pipeline(device, dtype)


def build_instantid_pipeline(device: str, dtype: torch.dtype):
    """SDXL + AnimateDiff + OpenPose ControlNet + InstantID IdentityNet."""
    from diffusers import (
        AnimateDiffSDXLPipeline, MotionAdapter,
        DDIMScheduler, ControlNetModel,
    )
    try:
        import insightface  # noqa: F401
    except ImportError:
        raise ImportError(
            "InstantID requires insightface: "
            "pip install insightface onnxruntime-gpu"
        )
    print("[cinematic] loading ControlNet OpenPose (SDXL)...")
    cn_pose = ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=dtype)
    print("[cinematic] loading InstantID IdentityNet...")
    cn_identity = ControlNetModel.from_pretrained(
        "InstantX/InstantID", subfolder="ControlNetModel", torch_dtype=dtype)
    print("[cinematic] loading AnimateDiff motion adapter...")
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-sdxl-beta", torch_dtype=dtype)
    print("[cinematic] loading SDXL + AnimateDiff pipeline...")
    pipe = AnimateDiffSDXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        motion_adapter=adapter, controlnet=[cn_pose, cn_identity],
        torch_dtype=dtype, variant="fp16",
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter("InstantX/InstantID", subfolder="",
                         weight_name="ip-adapter.bin")
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_tiling()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe.to(device)


# ── Identity loading ──────────────────────────────────────────────────────

def load_identity(
    reference: Optional[str],
    embedding: Optional[str],
    method: str,
    device: str,
) -> Optional[dict]:
    if embedding:
        from identity_encoder import IdentityEncoder
        emb = IdentityEncoder.load(embedding)
        print(f"[cinematic] loaded {method} embedding from {embedding}")
        return emb
    if reference:
        from identity_encoder import IdentityEncoder
        enc = IdentityEncoder(method=method, device=device)
        emb = enc.encode(reference)
        print(f"[cinematic] encoded identity ({method}) from {reference}")
        return emb
    print("[cinematic] warning: no --reference or --embedding — "
          "identity will not be preserved")
    return None


# ── Background lock ───────────────────────────────────────────────────────

def apply_background_lock(
    frames: list[Image.Image],
    background: Image.Image,
) -> list[Image.Image]:
    """Composite frames onto a static background to eliminate BG flicker."""
    import cv2
    bg_arr = np.array(background.resize(frames[0].size, Image.LANCZOS))
    out: list[Image.Image] = []
    for frame in frames:
        frame_arr = np.array(frame)
        diff = np.abs(
            frame_arr.astype(np.float32) - bg_arr.astype(np.float32)
        ).max(axis=2)
        mask = (diff > 20).astype(np.float32)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        mask = cv2.GaussianBlur(mask, (11, 11), 3)[..., np.newaxis]
        composite = (frame_arr * mask + bg_arr * (1 - mask)).clip(0, 255)
        out.append(Image.fromarray(composite.astype(np.uint8)))
    return out


# ── Core generation ───────────────────────────────────────────────────────

def generate_with_dual_controlnet(
    pipe,
    pose_maps: list[Image.Image],
    depth_maps: Optional[list[Image.Image]],
    prompt: str,
    negative_prompt: str,
    identity_embedding: Optional[dict],
    cfg: dict,
    seed: int,
) -> list[Image.Image]:
    """Sliding-window AnimateDiff with dual ControlNet and temporal consistency.

    Three-layer temporal stability (AVATAR_DIFFUSION_ARCHITECTURE.md §6):
      1. Single consistent noise generator across all windows
      2. Reduced CFG on non-first windows (prevents creative drift)
      3. Linear crossfade blend in overlap regions
    """
    W = cfg["window_size"]
    O = cfg["window_overlap"]
    step = W - O
    T = len(pose_maps)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    if identity_embedding is not None:
        pipe.set_ip_adapter_scale(cfg["ip_adapter_scale"])
        ip_embeds = identity_embedding.get("image_embeds")
        if ip_embeds is not None:
            ip_embeds = ip_embeds.to(pipe.device, dtype=torch.float16)
    else:
        ip_embeds = None

    dual = depth_maps is not None
    all_frames: list[Optional[Image.Image]] = [None] * T
    weights = np.zeros(T, dtype=np.float32)

    window_starts = list(range(0, max(T - W + 1, 1), step))
    if window_starts[-1] + W < T:
        window_starts.append(T - W)

    for wi, start in enumerate(window_starts):
        end = min(start + W, T)
        w_pose = pose_maps[start:end]
        n = len(w_pose)
        gs = cfg["guidance_scale"] if wi == 0 else max(
            cfg["guidance_scale"] - 2.0, 5.0)

        print(f"[cinematic] window {wi+1}/{len(window_starts)}: "
              f"frames {start}–{end-1}  cfg={gs:.1f}  "
              f"{'dual-CN' if dual else 'single-CN'}")

        if dual:
            cn_images = [w_pose, depth_maps[start:end]]
            cn_scales = [cfg["pose_scale"], cfg["depth_scale"]]
        else:
            cn_images = w_pose
            cn_scales = cfg["pose_scale"]

        kwargs: dict = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=cn_images,
            controlnet_conditioning_scale=cn_scales,
            num_inference_steps=cfg["num_steps"],
            guidance_scale=gs,
            generator=generator,
            width=cfg["width"],
            height=cfg["height"],
            num_frames=n,
        )
        if ip_embeds is not None:
            kwargs["ip_adapter_image_embeds"] = [ip_embeds]

        result = pipe(**kwargs)
        window_frames = result.frames[0]

        for local_i, frame in enumerate(window_frames):
            global_i = start + local_i
            if global_i >= T:
                break
            w_start = min(local_i / max(O, 1), 1.0)
            w_end   = min((n - 1 - local_i) / max(O, 1), 1.0)
            w = min(w_start, w_end)
            if all_frames[global_i] is None:
                all_frames[global_i] = frame
                weights[global_i] = w
            else:
                existing = np.array(all_frames[global_i], dtype=np.float32)
                new_f    = np.array(frame, dtype=np.float32)
                prev_w   = weights[global_i]
                blended  = (existing * prev_w + new_f * w) / (prev_w + w + 1e-8)
                all_frames[global_i] = Image.fromarray(
                    blended.clip(0, 255).astype(np.uint8))
                weights[global_i] = prev_w + w

    for i in range(T):
        if all_frames[i] is None:
            all_frames[i] = pose_maps[i].convert("RGB")

    return all_frames  # type: ignore[return-value]


# ── Video export ──────────────────────────────────────────────────────────

def frames_to_video(frames: list[Image.Image], out_path: Path, fps: float) -> None:
    try:
        import imageio
    except ImportError:
        raise ImportError("imageio required: pip install imageio[ffmpeg]")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                quality=9, pixelformat="yuv420p")
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()
    print(f"[cinematic] wrote {len(frames)} frames @ {fps}fps → {out_path}")


# ── Single-clip pipeline ──────────────────────────────────────────────────

def run_clip(
    clip_name: str,
    reference: Optional[str],
    embedding: Optional[str],
    prompt: str,
    negative_prompt: str,
    out_path: Path,
    identity_method: str,
    dual_controlnet: bool,
    n_frames: int,
    cfg: dict,
    post_cfg: dict,
    device: str,
    dtype: torch.dtype,
    lock_background: bool,
) -> None:
    from multi_controlnet_conditioning import ConditioningMaps
    from postprocess import interpolate_frames, upscale_frames, reduce_flicker

    t0 = time.perf_counter()
    cond = ConditioningMaps(clip_name, width=cfg["width"],
                            height=cfg["height"], n_frames=n_frames)
    pose_maps, depth_maps = cond.load_paired()

    if not dual_controlnet or depth_maps is None:
        if dual_controlnet:
            print("[cinematic] depth maps unavailable — "
                  "falling back to single-ControlNet")
        depth_maps = None

    identity = load_identity(reference, embedding, identity_method, device)

    if identity_method == "instantid":
        pipe = build_instantid_pipeline(device, dtype)
    elif dual_controlnet and depth_maps is not None:
        pipe = build_dual_controlnet_pipeline(device, dtype)
    else:
        pipe = build_single_controlnet_pipeline(device, dtype)

    raw_frames = generate_with_dual_controlnet(
        pipe, pose_maps, depth_maps,
        prompt, negative_prompt, identity, cfg, cfg["seed"],
    )

    if lock_background and raw_frames:
        raw_frames = apply_background_lock(raw_frames, raw_frames[0].copy())

    frames = raw_frames
    out_fps: float = float(cfg["fps_out"])

    if post_cfg.get("reduce_flicker"):
        frames = reduce_flicker(frames, blend_alpha=post_cfg["flicker_alpha"])
    if post_cfg.get("interpolate"):
        frames = interpolate_frames(frames, target_fps=post_cfg["target_fps"],
                                    source_fps=out_fps, device=device)
        out_fps = post_cfg["target_fps"]
    if post_cfg.get("upscale"):
        frames = upscale_frames(frames, scale=post_cfg["upscale_scale"])

    frames_to_video(frames, out_path, out_fps)
    print(f"[cinematic] '{clip_name}' done in "
          f"{time.perf_counter()-t0:.1f}s → {out_path}")


# ── Batch mode ────────────────────────────────────────────────────────────

def _discover_clips() -> list[str]:
    pose_base = ROOT / "outputs" / "avatar" / "pose_maps"
    if not pose_base.exists():
        return []
    return sorted(d.name for d in pose_base.iterdir()
                  if d.is_dir() and list(d.glob("*_pose.png")))


def run_batch(
    reference: Optional[str], embedding: Optional[str],
    prompt: str, negative_prompt: str, out_dir: Path,
    identity_method: str, dual_controlnet: bool, n_frames: int,
    cfg: dict, post_cfg: dict, device: str, dtype: torch.dtype,
    lock_background: bool,
) -> None:
    clips = _discover_clips()
    if not clips:
        raise RuntimeError(
            "No pose maps found. Run multi_controlnet_conditioning.py "
            "--extract-pose first.")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cinematic] batch: {len(clips)} clips → {out_dir}")
    for clip in clips:
        out_path = out_dir / f"{clip}_cinematic.mp4"
        print(f"\n{'='*60}\n[cinematic] {clip}\n{'='*60}")
        try:
            run_clip(clip, reference, embedding, prompt, negative_prompt,
                     out_path, identity_method, dual_controlnet, n_frames,
                     cfg, post_cfg, device, dtype, lock_background)
        except Exception as e:
            import traceback
            print(f"[cinematic] ✗ FAILED: {clip}: {e}", file=sys.stderr)
            traceback.print_exc()


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip",  default=None)
    p.add_argument("--batch", action="store_true")
    p.add_argument("--reference",       default=None)
    p.add_argument("--embedding",       default=None)
    p.add_argument("--identity-method", default="ip_adapter",
                   choices=["ip_adapter", "instantid"])
    p.add_argument("--prompt",          required=True)
    p.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    p.add_argument("--out",     default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--dual-controlnet", action="store_true")
    p.add_argument("--n-frames",    type=int,   default=0)
    p.add_argument("--width",       type=int,   default=DEFAULTS["width"])
    p.add_argument("--height",      type=int,   default=DEFAULTS["height"])
    p.add_argument("--fps",         type=int,   default=DEFAULTS["fps_out"])
    p.add_argument("--steps",       type=int,   default=DEFAULTS["num_steps"])
    p.add_argument("--cfg",         type=float, default=DEFAULTS["guidance_scale"],
                   dest="guidance_scale")
    p.add_argument("--pose-scale",  type=float, default=DEFAULTS["pose_scale"])
    p.add_argument("--depth-scale", type=float, default=DEFAULTS["depth_scale"])
    p.add_argument("--ip-scale",    type=float, default=DEFAULTS["ip_adapter_scale"])
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      default=None)
    p.add_argument("--dtype",       default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--interpolate",     action="store_true")
    p.add_argument("--target-fps",      type=float, default=24.0)
    p.add_argument("--upscale",         action="store_true")
    p.add_argument("--upscale-scale",   type=int,   default=2, choices=[2, 4])
    p.add_argument("--reduce-flicker",  action="store_true")
    p.add_argument("--flicker-alpha",   type=float, default=0.12)
    p.add_argument("--lock-background", action="store_true")
    args = p.parse_args()

    if not args.clip and not args.batch:
        print("[error] specify --clip <name> or --batch", file=sys.stderr)
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cinematic] device={device}  dtype={args.dtype}  "
          f"identity={args.identity_method}  dual-CN={args.dual_controlnet}")

    dtype_map = {"float16": torch.float16,
                 "bfloat16": torch.bfloat16,
                 "float32": torch.float32}

    cfg = {
        "window_size":      DEFAULTS["window_size"],
        "window_overlap":   DEFAULTS["window_overlap"],
        "num_steps":        args.steps,
        "guidance_scale":   args.guidance_scale,
        "denoise_strength": DEFAULTS["denoise_strength"],
        "pose_scale":       args.pose_scale,
        "depth_scale":      args.depth_scale,
        "ip_adapter_scale": args.ip_scale,
        "width":            args.width,
        "height":           args.height,
        "fps_out":          args.fps,
        "seed":             args.seed,
        "motion_bucket_id": DEFAULTS["motion_bucket_id"],
        "noise_aug":        DEFAULTS["noise_aug"],
        "decode_chunk":     DEFAULTS["decode_chunk"],
    }
    post_cfg = {
        "interpolate":    args.interpolate,
        "target_fps":     args.target_fps,
        "upscale":        args.upscale,
        "upscale_scale":  args.upscale_scale,
        "reduce_flicker": args.reduce_flicker,
        "flicker_alpha":  args.flicker_alpha,
    }

    if args.batch:
        out_dir = (Path(args.out_dir) if args.out_dir
                   else ROOT / "outputs" / "avatar" / "videos" / "cinematic")
        run_batch(args.reference, args.embedding, args.prompt,
                  args.negative_prompt, out_dir, args.identity_method,
                  args.dual_controlnet, args.n_frames, cfg, post_cfg,
                  device, dtype_map[args.dtype], args.lock_background)
    else:
        out_path = (Path(args.out) if args.out
                    else ROOT / "outputs" / "avatar" / "videos" /
                         f"{args.clip}_cinematic.mp4")
        run_clip(args.clip, args.reference, args.embedding, args.prompt,
                 args.negative_prompt, out_path, args.identity_method,
                 args.dual_controlnet, args.n_frames, cfg, post_cfg,
                 device, dtype_map[args.dtype], args.lock_background)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
