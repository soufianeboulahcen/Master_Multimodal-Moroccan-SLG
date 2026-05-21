"""CLI entry point for avatar video generation.

Converts a SignLLM skeleton output (OpenPose JSON directory) into a
photorealistic avatar video using AnimateDiff or MimicMotion.

Two input modes:

    --json-dir   : use an existing OpenPose JSON directory directly
                   (produced by mosl/pose/export_openpose_json.py or
                   scripts/generate_openpose/generate.py)

    --sign       : generate pose from a trained SignLLM checkpoint first,
                   then render the avatar.  Requires --run and --vocab.

Usage examples:

    # Render from existing JSON (fastest — no SignLLM inference needed)
    python scripts/render_avatar.py \\
        --json-dir outputs/openpose_json/walking_keypoints \\
        --reference outputs/avatar/reference_images/signer.jpg \\
        --out outputs/avatar/walking/walking_avatar.mp4

    # Generate sign from text, then render
    python scripts/render_avatar.py \\
        --sign أَنَا \\
        --run baseline_mse \\
        --reference outputs/avatar/reference_images/signer.jpg \\
        --out outputs/avatar/أَنَا/أَنَا_avatar.mp4

    # Use MimicMotion backend
    python scripts/render_avatar.py \\
        --json-dir outputs/openpose_json/dancing_keypoints \\
        --reference signer.jpg \\
        --backend mimicmotion \\
        --out outputs/avatar/dancing/dancing_avatar.mp4

    # Batch render all procedural motions
    python scripts/render_avatar.py \\
        --batch-motions \\
        --reference outputs/avatar/reference_images/signer.jpg

    # Dry run: render pose maps only (no diffusion, no GPU required)
    python scripts/render_avatar.py \\
        --json-dir outputs/openpose_json/walking_keypoints \\
        --pose-maps-only \\
        --out outputs/avatar/walking/pose_maps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _generate_sign_json(sign_text: str, run_name: str, vocab_path: str) -> Path:
    """Run SignLLM inference and export OpenPose JSON.  Returns the JSON dir."""
    import torch
    from mosl.model.signllm import SignLLM, SignLLMConfig
    from mosl.text.tokenizer import WordTokenizer
    from mosl.pose.export_openpose_json import emit_json_for_clip
    import numpy as np

    tok = WordTokenizer.load(vocab_path)
    ckpt_path = ROOT / "runs" / run_name / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = SignLLMConfig(**ckpt["model_cfg"])
    model = SignLLM(model_cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    ids = tok.encode(sign_text)
    text_ids = torch.tensor([ids], dtype=torch.long, device=device)
    text_mask = torch.ones_like(text_ids, dtype=torch.bool)

    with torch.no_grad():
        gen = model.generate(text_ids, text_mask)

    T_pred = int(gen["lengths"][0].item())
    pose = gen["pose"][0, :T_pred].cpu().numpy()   # (T, 150)

    # Build NPZ in the format expected by emit_json_for_clip.
    # emit_json_for_clip expects arrays: pose_keypoints_2d (T,54),
    # hand_left_keypoints_2d (T,63), hand_right_keypoints_2d (T,63).
    # Our pose is (T, 150) = 50 joints × xyz.
    # Layout: joints 0–7 = body (8 joints), 8–28 = left hand (21), 29–49 = right hand (21).
    # We reconstruct approximate 2D projections (x, y from xyz, confidence=1.0).

    T = pose.shape[0]
    # Body: joints 0–7 → 8 joints → pad to 18 for COCO-18 (zeros for missing)
    body_2d = np.zeros((T, 54), dtype=np.float32)   # 18 * 3
    for j in range(8):
        body_2d[:, j * 3] = pose[:, j * 3] * 1280       # x (denormalize approx)
        body_2d[:, j * 3 + 1] = pose[:, j * 3 + 1] * 720  # y
        body_2d[:, j * 3 + 2] = 1.0                       # confidence

    # Hands: joints 8–28 = left (21 joints), 29–49 = right (21 joints)
    lhand_2d = np.zeros((T, 63), dtype=np.float32)   # 21 * 3
    rhand_2d = np.zeros((T, 63), dtype=np.float32)
    for j in range(21):
        lj = 8 + j
        rj = 29 + j
        lhand_2d[:, j * 3] = pose[:, lj * 3] * 1280
        lhand_2d[:, j * 3 + 1] = pose[:, lj * 3 + 1] * 720
        lhand_2d[:, j * 3 + 2] = 1.0
        rhand_2d[:, j * 3] = pose[:, rj * 3] * 1280
        rhand_2d[:, j * 3 + 1] = pose[:, rj * 3 + 1] * 720
        rhand_2d[:, j * 3 + 2] = 1.0

    # Save temporary NPZ.
    safe_sign = "".join(c for c in sign_text if c.isalnum() or c in "-_") or "sign"
    npz_dir = ROOT / "outputs" / "avatar" / "_tmp_npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz_path = npz_dir / f"{safe_sign}.npz"
    np.savez(
        npz_path,
        pose_keypoints_2d=body_2d,
        hand_left_keypoints_2d=lhand_2d,
        hand_right_keypoints_2d=rhand_2d,
    )

    # Export to OpenPose JSON.
    json_dir = ROOT / "outputs" / "avatar" / f"{safe_sign}_keypoints"
    emit_json_for_clip(npz_path, json_dir)
    print(f"[render_avatar] Generated {T_pred} frames → {json_dir}")
    return json_dir


def _render_pose_maps_only(json_dir: Path, out_dir: Path) -> None:
    """Dry run: render pose maps without diffusion."""
    from avatar.conditioning.pose_to_controlnet import PoseMapRenderer
    from avatar.config import PoseMapConfig

    renderer = PoseMapRenderer(PoseMapConfig())
    body_paths, hand_paths = renderer.render_and_save(
        json_dir,
        out_dir / "body",
        out_dir / "hand",
        verbose=True,
    )
    renderer.render_debug_grid(json_dir, out_dir / "debug_grid.png")
    print(f"[render_avatar] Pose maps saved to {out_dir}")
    print(f"  body: {len(body_paths)} frames")
    print(f"  hand: {len(hand_paths)} frames")
    print(f"  debug grid: {out_dir / 'debug_grid.png'}")


def _batch_motions(reference: Path, backend: str, out_root: Path) -> None:
    """Render all procedural motions from outputs/openpose_json/."""
    from avatar.config import AvatarConfig, BackendType
    from avatar.pipeline import AvatarPipeline

    json_root = ROOT / "outputs" / "openpose_json"
    motion_dirs = sorted(d for d in json_root.iterdir() if d.is_dir())
    if not motion_dirs:
        print(f"[render_avatar] No motion directories found in {json_root}")
        return

    cfg = AvatarConfig(backend=BackendType(backend))
    jobs = [
        {
            "json_dir": str(d),
            "reference_image": str(reference),
            "output_path": str(out_root / d.name.replace("_keypoints", "") / f"{d.name.replace('_keypoints', '')}_avatar.mp4"),
        }
        for d in motion_dirs
    ]

    print(f"[render_avatar] Batch rendering {len(jobs)} motions")
    with AvatarPipeline(cfg) as pipe:
        results = pipe.render_batch(jobs)

    for job, result in zip(jobs, results):
        if result.n_frames > 0:
            print(f"  ✅ {Path(job['output_path']).name}: {result.n_frames} frames")
        else:
            print(f"  ❌ {Path(job['output_path']).name}: failed — {result.metadata.get('error', '?')}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input source (mutually exclusive).
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--json-dir",
        help="Directory of *_keypoints.json files (existing OpenPose output).",
    )
    src.add_argument(
        "--sign",
        help="Arabic sign text to generate via SignLLM, then render.",
    )
    src.add_argument(
        "--batch-motions",
        action="store_true",
        help="Render all procedural motions from outputs/openpose_json/.",
    )

    # SignLLM options (only used with --sign).
    p.add_argument("--run", default="baseline_mse",
                   help="SignLLM run name under runs/ (default: baseline_mse).")
    p.add_argument("--vocab", default=str(ROOT / "data" / "processed" / "vocab.json"),
                   help="Path to vocab.json.")

    # Identity conditioning.
    p.add_argument(
        "--reference",
        help="Reference signer image for identity conditioning.",
    )
    p.add_argument(
        "--identity-backend",
        choices=["instantid", "ip_adapter", "none"],
        default="instantid",
        help="Identity conditioning backend (default: instantid).",
    )

    # Rendering options.
    p.add_argument(
        "--backend",
        choices=["animatediff", "mimicmotion", "auto"],
        default="animatediff",
        help="Diffusion rendering backend (default: animatediff).",
    )
    p.add_argument("--prompt", default=None, help="Custom positive text prompt.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-interpolate", action="store_true",
                   help="Skip RIFE temporal interpolation.")
    p.add_argument("--rife-scale", type=int, default=2, choices=[2, 4],
                   help="RIFE upsampling factor (default: 2).")

    # Output.
    p.add_argument("--out", default=None,
                   help="Output path (.mp4 for video, directory for --pose-maps-only).")

    # Dry run.
    p.add_argument("--pose-maps-only", action="store_true",
                   help="Render ControlNet pose maps only (no diffusion, no GPU required).")

    # Config.
    p.add_argument("--config", default=None,
                   help="Path to AvatarConfig JSON file (overrides CLI flags).")

    args = p.parse_args()

    # ------------------------------------------------------------------
    # Batch motions shortcut
    # ------------------------------------------------------------------
    if args.batch_motions:
        if not args.reference:
            print("Error: --reference is required for --batch-motions")
            return 1
        out_root = Path(args.out) if args.out else ROOT / "outputs" / "avatar"
        _batch_motions(Path(args.reference), args.backend, out_root)
        return 0

    # ------------------------------------------------------------------
    # Resolve JSON directory
    # ------------------------------------------------------------------
    if args.json_dir:
        json_dir = Path(args.json_dir)
        if not json_dir.exists():
            print(f"Error: --json-dir not found: {json_dir}")
            return 1
    elif args.sign:
        print(f"[render_avatar] Generating sign: {args.sign!r}")
        json_dir = _generate_sign_json(args.sign, args.run, args.vocab)
    else:
        print("Error: one of --json-dir, --sign, or --batch-motions is required")
        p.print_help()
        return 1

    # ------------------------------------------------------------------
    # Pose-maps-only dry run
    # ------------------------------------------------------------------
    if args.pose_maps_only:
        out_dir = Path(args.out) if args.out else ROOT / "outputs" / "avatar" / "pose_maps_debug"
        _render_pose_maps_only(json_dir, out_dir)
        return 0

    # ------------------------------------------------------------------
    # Full render
    # ------------------------------------------------------------------
    if not args.reference:
        print("Error: --reference is required for full rendering")
        return 1

    # Build config.
    if args.config:
        from avatar.config import AvatarConfig
        cfg = AvatarConfig.from_json(args.config)
    else:
        from avatar.config import AvatarConfig, BackendType, IdentityConfig, IdentityBackend, RIFEConfig
        cfg = AvatarConfig(
            backend=BackendType(args.backend),
            seed=args.seed,
            interpolate=not args.no_interpolate,
        )
        cfg.identity = IdentityConfig(backend=IdentityBackend(args.identity_backend))
        cfg.rife = RIFEConfig(scale_factor=args.rife_scale)

    # Resolve output path.
    if args.out:
        out_path = Path(args.out)
    else:
        sign_name = args.sign or json_dir.name.replace("_keypoints", "")
        safe = "".join(c for c in sign_name if c.isalnum() or c in "-_") or "output"
        out_path = ROOT / "outputs" / "avatar" / safe / f"{safe}_avatar.mp4"

    # Run pipeline.
    from avatar.pipeline import AvatarPipeline
    with AvatarPipeline(cfg) as pipe:
        result = pipe.render(
            json_dir=json_dir,
            reference_image=args.reference,
            output_path=out_path,
            prompt=args.prompt,
            seed=args.seed,
        )

    print(f"\n✅ Avatar video: {out_path}")
    print(f"   Frames: {result.n_frames}  FPS: {result.fps}")
    print(f"   Backend: {result.metadata.get('backend', '?')}")
    print(f"   Total time: {result.metadata.get('pipeline_total_time_s', 0):.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
