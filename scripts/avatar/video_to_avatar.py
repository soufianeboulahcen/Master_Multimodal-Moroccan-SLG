"""End-to-end pipeline: OpenPose video → realistic avatar video.

Takes an existing OpenPose skeleton video (or keypoint JSON directory) and
produces a cinematic, photorealistic avatar video using:

  1. video_to_pose_maps.py  — extract ControlNet pose-map PNGs from the video
  2. generate_avatar_video.py — AnimateDiff + SDXL + ControlNet + IP-Adapter
  3. postprocess.py          — RIFE interpolation + Real-ESRGAN upscale

All three stages are run in sequence.  Any stage can be skipped with the
corresponding --skip-* flag (useful for resuming after a partial run).

Usage:
    # Minimal — skeleton video + reference portrait
    python scripts/avatar/video_to_avatar.py \\
        --input  outputs/videos/skeleton/walking_skeleton.mp4 \\
        --reference assets/avatar_reference.jpg \\
        --prompt "a Moroccan man performing sign language, studio background, \\
                  photorealistic, DSLR, cinematic lighting" \\
        --out    outputs/avatar/videos/walking_hq.mp4

    # From JSON keypoints (cleaner pose maps)
    python scripts/avatar/video_to_avatar.py \\
        --input  outputs/openpose_json/walking_keypoints/ \\
        --source json \\
        --reference assets/avatar_reference.jpg \\
        --prompt "a Moroccan signer, studio background, photorealistic" \\
        --out    outputs/avatar/videos/walking_hq.mp4

    # Full quality: AnimateDiff + upscale + interpolation to 24 fps
    python scripts/avatar/video_to_avatar.py \\
        --input  outputs/videos/skeleton/waving_skeleton.mp4 \\
        --reference assets/avatar_reference.jpg \\
        --prompt "a Moroccan woman signing, clean white background, \\
                  photorealistic, natural lighting" \\
        --out    outputs/avatar/videos/waving_hq.mp4 \\
        --upscale --interpolate --target-fps 24

    # SVD mode (faster, ≤ 25 frames)
    python scripts/avatar/video_to_avatar.py \\
        --input  outputs/videos/skeleton/hand_face_skeleton.mp4 \\
        --reference assets/avatar_reference.jpg \\
        --prompt "a Moroccan signer, photorealistic" \\
        --out    outputs/avatar/videos/hand_face_svd.mp4 \\
        --mode svd --n-frames 25

    # Batch: all skeleton videos
    python scripts/avatar/video_to_avatar.py \\
        --input  outputs/videos/skeleton/ \\
        --batch \\
        --reference assets/avatar_reference.jpg \\
        --prompt "a Moroccan man performing sign language, photorealistic" \\
        --out-dir outputs/avatar/videos/batch/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "avatar"))

# ── Default generation settings ───────────────────────────────────────────
# These match the recommended values from docs/AI_AVATAR_PIPELINE.md §5
# and the user's specification (CFG=7, denoise=0.7, controlnet=0.9, fps=24).

DEFAULTS = {
    "width":             512,
    "height":            768,
    "fps":               24,
    "n_frames":          0,       # 0 = all frames from the video
    "mode":              "animatediff",
    "steps":             25,
    "cfg":               7.0,
    "controlnet_scale":  0.9,
    "ip_scale":          0.65,
    "seed":              42,
    "target_fps":        24.0,
    "upscale_scale":     2,
    "flicker_alpha":     0.12,
}

NEGATIVE_PROMPT = (
    "blurry, deformed hands, extra fingers, missing fingers, distorted face, "
    "low quality, cartoon, anime, painting, watermark, text, logo, "
    "multiple people, crowd, nsfw, duplicate limbs, temporal artifacts, "
    "flickering, unrealistic anatomy, fake skin, plastic skin"
)


# ── Stage runners ─────────────────────────────────────────────────────────

def run_pose_extraction(
    input_path: Path,
    source: str,
    pose_maps_dir: Path,
    width: int,
    height: int,
    n_frames: int,
) -> Path:
    """Stage 1: extract ControlNet pose-map PNGs from the input."""
    from video_to_pose_maps import (
        extract_frames_from_video,
        save_video_frames_as_pose_maps,
        render_json_keypoints_as_pose_maps,
        _detect_source_resolution,
    )

    print(f"\n{'='*60}")
    print(f"[pipeline] Stage 1 — pose extraction ({source})")
    print(f"  input : {input_path}")
    print(f"  output: {pose_maps_dir}")
    print(f"{'='*60}")

    if source == "video":
        frames = extract_frames_from_video(
            input_path, width, height, max_frames=n_frames
        )
        save_video_frames_as_pose_maps(frames, pose_maps_dir)
    else:  # json
        src_w, src_h = _detect_source_resolution(input_path)
        render_json_keypoints_as_pose_maps(
            input_path, pose_maps_dir,
            width, height, src_w, src_h,
            max_frames=n_frames,
        )

    return pose_maps_dir


def run_generation(
    pose_maps_dir: Path,
    reference: str | None,
    embedding: str | None,
    prompt: str,
    negative_prompt: str,
    out_path: Path,
    mode: str,
    cfg: dict,
    device: str,
    dtype: str,
) -> Path:
    """Stage 2: AnimateDiff / SVD generation."""
    import torch
    from generate_avatar_video import (
        build_animatediff_pipeline,
        build_svd_pipeline,
        load_pose_maps,
        generate_animatediff,
        generate_svd,
        frames_to_video,
    )

    print(f"\n{'='*60}")
    print(f"[pipeline] Stage 2 — diffusion generation ({mode})")
    print(f"  pose maps : {pose_maps_dir}")
    print(f"  output    : {out_path}")
    print(f"  cfg scale : {cfg['guidance_scale']}  "
          f"controlnet: {cfg['controlnet_scale']}  "
          f"ip-adapter: {cfg['ip_adapter_scale']}")
    print(f"{'='*60}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]

    pose_maps = load_pose_maps(
        pose_maps_dir, cfg.get("n_frames", 0), cfg["width"], cfg["height"]
    )

    # Identity embedding
    identity_embedding = None
    if embedding:
        from identity_encoder import IdentityEncoder
        identity_embedding = IdentityEncoder.load(embedding)
        print(f"[pipeline] loaded embedding from {embedding}")
    elif reference:
        from identity_encoder import IdentityEncoder
        enc = IdentityEncoder(method="ip_adapter", device=device)
        identity_embedding = enc.encode(reference)
        print(f"[pipeline] encoded identity from {reference}")
    else:
        print("[pipeline] warning: no --reference or --embedding — "
              "identity will not be preserved")

    if mode == "animatediff":
        pipe = build_animatediff_pipeline(device, torch_dtype)
        frames = generate_animatediff(
            pipe, pose_maps, prompt, negative_prompt,
            identity_embedding, cfg, cfg["seed"],
        )
    else:  # svd
        pipe = build_svd_pipeline(device, torch_dtype)
        frames = generate_svd(pipe, pose_maps[0], len(pose_maps), cfg, cfg["seed"])

    frames_to_video(frames, out_path, cfg["fps_out"])
    return out_path


def run_postprocess(
    in_path: Path,
    out_path: Path,
    src_fps: float,
    interpolate: bool,
    target_fps: float,
    upscale: bool,
    upscale_scale: int,
    reduce_flicker: bool,
    flicker_alpha: float,
    device: str | None,
) -> Path:
    """Stage 3: RIFE interpolation + Real-ESRGAN upscale."""
    from postprocess import (
        read_video_frames,
        write_video_frames,
        interpolate_frames,
        upscale_frames,
        reduce_flicker as do_reduce_flicker,
    )

    if not interpolate and not upscale and not reduce_flicker:
        return in_path  # nothing to do

    print(f"\n{'='*60}")
    print(f"[pipeline] Stage 3 — post-processing")
    print(f"  input : {in_path}")
    print(f"  output: {out_path}")
    print(f"  interpolate={interpolate}  upscale={upscale}  "
          f"flicker={reduce_flicker}")
    print(f"{'='*60}")

    frames, fps = read_video_frames(in_path)
    out_fps = fps

    if reduce_flicker:
        frames = do_reduce_flicker(frames, blend_alpha=flicker_alpha)

    if interpolate:
        frames = interpolate_frames(
            frames,
            target_fps=target_fps,
            source_fps=fps,
            device=device,
        )
        out_fps = target_fps

    if upscale:
        frames = upscale_frames(frames, scale=upscale_scale)

    write_video_frames(frames, out_path, out_fps)
    return out_path


# ── Single-video pipeline ─────────────────────────────────────────────────

def run_pipeline(
    input_path: Path,
    source: str,
    reference: str | None,
    embedding: str | None,
    prompt: str,
    negative_prompt: str,
    out_path: Path,
    mode: str,
    gen_cfg: dict,
    post_cfg: dict,
    device: str,
    dtype: str,
    skip_pose: bool,
    skip_generation: bool,
    skip_postprocess: bool,
    pose_maps_dir: Path | None,
) -> None:
    t0 = time.perf_counter()

    # Derive pose_maps_dir from output path if not given
    if pose_maps_dir is None:
        pose_maps_dir = (
            ROOT / "outputs" / "avatar" / "pose_maps" / out_path.stem
        )

    # Stage 1
    if not skip_pose:
        run_pose_extraction(
            input_path, source, pose_maps_dir,
            gen_cfg["width"], gen_cfg["height"], gen_cfg.get("n_frames", 0),
        )
    else:
        print(f"[pipeline] skipping pose extraction — using {pose_maps_dir}")

    # Stage 2
    raw_out = out_path.with_stem(out_path.stem + "_raw") if (
        post_cfg["interpolate"] or post_cfg["upscale"] or post_cfg["reduce_flicker"]
    ) else out_path

    if not skip_generation:
        run_generation(
            pose_maps_dir, reference, embedding,
            prompt, negative_prompt, raw_out,
            mode, gen_cfg, device, dtype,
        )
    else:
        print(f"[pipeline] skipping generation — using {raw_out}")

    # Stage 3
    if not skip_postprocess:
        final = run_postprocess(
            raw_out, out_path,
            src_fps=gen_cfg["fps_out"],
            **post_cfg,
            device=device,
        )
        # Remove the intermediate raw file if post-processing produced a new one
        if final != raw_out and raw_out.exists() and raw_out != out_path:
            raw_out.unlink()
    else:
        print("[pipeline] skipping post-processing")

    elapsed = time.perf_counter() - t0
    print(f"\n[pipeline] done in {elapsed:.1f}s → {out_path}")


# ── Batch mode ────────────────────────────────────────────────────────────

def run_batch(
    input_dir: Path,
    source: str,
    out_dir: Path,
    reference: str | None,
    embedding: str | None,
    prompt: str,
    negative_prompt: str,
    mode: str,
    gen_cfg: dict,
    post_cfg: dict,
    device: str,
    dtype: str,
) -> None:
    if source == "video":
        inputs = sorted(input_dir.glob("*_skeleton.mp4"))
        if not inputs:
            raise FileNotFoundError(f"No *_skeleton.mp4 in {input_dir}")
    else:
        inputs = sorted(
            d for d in input_dir.iterdir()
            if d.is_dir() and list(d.glob("*_keypoints.json"))
        )
        if not inputs:
            raise FileNotFoundError(f"No keypoint dirs in {input_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] batch: {len(inputs)} inputs → {out_dir}")

    for inp in inputs:
        stem = inp.stem.replace("_skeleton", "").replace("_keypoints", "")
        out_path = out_dir / f"{stem}_avatar.mp4"
        print(f"\n{'#'*60}")
        print(f"[pipeline] processing: {inp.name}")
        print(f"{'#'*60}")
        try:
            run_pipeline(
                input_path=inp,
                source=source,
                reference=reference,
                embedding=embedding,
                prompt=prompt,
                negative_prompt=negative_prompt,
                out_path=out_path,
                mode=mode,
                gen_cfg=gen_cfg,
                post_cfg=post_cfg,
                device=device,
                dtype=dtype,
                skip_pose=False,
                skip_generation=False,
                skip_postprocess=False,
                pose_maps_dir=None,
            )
        except Exception as e:
            import traceback
            print(f"[pipeline] ✗ FAILED: {inp.name}: {e}", file=sys.stderr)
            traceback.print_exc()


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    p.add_argument("--input", required=True,
                   help="Skeleton MP4, keypoints dir, or parent dir (--batch)")
    p.add_argument("--source", default="video", choices=["video", "json"],
                   help="Input source type (default: video)")
    p.add_argument("--batch", action="store_true",
                   help="Process all videos/JSON dirs under --input")

    # Identity
    p.add_argument("--reference", default=None,
                   help="Reference portrait image for identity preservation")
    p.add_argument("--embedding", default=None,
                   help="Pre-encoded identity embedding (.pt)")

    # Prompt
    p.add_argument("--prompt", required=True,
                   help="Text prompt describing the avatar and scene")
    p.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)

    # Output
    p.add_argument("--out", default=None,
                   help="Output MP4 path (single mode)")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (batch mode)")

    # Generation
    p.add_argument("--mode", default=DEFAULTS["mode"],
                   choices=["animatediff", "svd"])
    p.add_argument("--n-frames", type=int, default=DEFAULTS["n_frames"],
                   help="Max frames to process (0 = all)")
    p.add_argument("--width",  type=int, default=DEFAULTS["width"])
    p.add_argument("--height", type=int, default=DEFAULTS["height"])
    p.add_argument("--fps",    type=int, default=DEFAULTS["fps"])
    p.add_argument("--steps",  type=int, default=DEFAULTS["steps"])
    p.add_argument("--cfg",    type=float, default=DEFAULTS["cfg"],
                   dest="guidance_scale")
    p.add_argument("--controlnet-scale", type=float,
                   default=DEFAULTS["controlnet_scale"])
    p.add_argument("--ip-scale", type=float, default=DEFAULTS["ip_scale"])
    p.add_argument("--seed",   type=int, default=DEFAULTS["seed"])
    p.add_argument("--device", default=None,
                   help="Torch device (default: auto-detect)")
    p.add_argument("--dtype",  default="float16",
                   choices=["float16", "bfloat16", "float32"])

    # Post-processing
    p.add_argument("--interpolate", action="store_true",
                   help="Apply RIFE frame interpolation after generation")
    p.add_argument("--target-fps", type=float, default=DEFAULTS["target_fps"],
                   help="Target FPS after interpolation (default: 24)")
    p.add_argument("--upscale", action="store_true",
                   help="Apply Real-ESRGAN ×2 upscale after generation")
    p.add_argument("--upscale-scale", type=int, default=DEFAULTS["upscale_scale"],
                   choices=[2, 4])
    p.add_argument("--reduce-flicker", action="store_true",
                   help="Apply EMA flicker reduction before other post-processing")
    p.add_argument("--flicker-alpha", type=float, default=DEFAULTS["flicker_alpha"])

    # Skip flags (for resuming)
    p.add_argument("--skip-pose",        action="store_true")
    p.add_argument("--skip-generation",  action="store_true")
    p.add_argument("--skip-postprocess", action="store_true")
    p.add_argument("--pose-maps-dir", default=None,
                   help="Existing pose-maps dir (implies --skip-pose)")

    args = p.parse_args()

    # Auto-detect device
    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"[pipeline] device: {device}  dtype: {args.dtype}")

    # Build config dicts
    gen_cfg = {
        "window_size":      16,
        "window_overlap":   4,
        "num_steps":        args.steps,
        "guidance_scale":   args.guidance_scale,
        "controlnet_scale": args.controlnet_scale,
        "ip_adapter_scale": args.ip_scale,
        "width":            args.width,
        "height":           args.height,
        "fps_out":          args.fps,
        "n_frames":         args.n_frames,
        "seed":             args.seed,
        # SVD-specific
        "motion_bucket_id": 110,
        "noise_aug":        0.02,
        "decode_chunk":     8,
    }

    post_cfg = {
        "interpolate":     args.interpolate,
        "target_fps":      args.target_fps,
        "upscale":         args.upscale,
        "upscale_scale":   args.upscale_scale,
        "reduce_flicker":  args.reduce_flicker,
        "flicker_alpha":   args.flicker_alpha,
    }

    inp = Path(args.input)
    if not inp.exists():
        print(f"[error] not found: {inp}", file=sys.stderr)
        return 1

    if args.batch:
        out_dir = (
            Path(args.out_dir) if args.out_dir
            else ROOT / "outputs" / "avatar" / "videos" / "batch"
        )
        run_batch(
            input_dir=inp,
            source=args.source,
            out_dir=out_dir,
            reference=args.reference,
            embedding=args.embedding,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            mode=args.mode,
            gen_cfg=gen_cfg,
            post_cfg=post_cfg,
            device=device,
            dtype=args.dtype,
        )
    else:
        if not args.out:
            stem = inp.stem.replace("_skeleton", "").replace("_keypoints", "")
            out_path = ROOT / "outputs" / "avatar" / "videos" / f"{stem}_avatar.mp4"
        else:
            out_path = Path(args.out)

        pose_maps_dir = Path(args.pose_maps_dir) if args.pose_maps_dir else None
        skip_pose = args.skip_pose or (pose_maps_dir is not None)

        run_pipeline(
            input_path=inp,
            source=args.source,
            reference=args.reference,
            embedding=args.embedding,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            out_path=out_path,
            mode=args.mode,
            gen_cfg=gen_cfg,
            post_cfg=post_cfg,
            device=device,
            dtype=args.dtype,
            skip_pose=skip_pose,
            skip_generation=args.skip_generation,
            skip_postprocess=args.skip_postprocess,
            pose_maps_dir=pose_maps_dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
