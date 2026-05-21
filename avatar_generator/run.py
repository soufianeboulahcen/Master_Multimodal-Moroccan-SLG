"""CLI entry point for avatar_generator.

Examples
--------
# Video from OpenPose JSON directory (no reference image)
python avatar_generator/run.py \\
    --pose outputs/openpose_json/walking_keypoints \\
    --out  outputs/avatar_generator/walking.mp4

# Video with identity conditioning
python avatar_generator/run.py \\
    --pose      outputs/openpose_json/walking_keypoints \\
    --reference outputs/avatar_generator/reference/signer.jpg \\
    --out       outputs/avatar_generator/walking_identity.mp4

# Video from .skels file
python avatar_generator/run.py \\
    --pose data/processed/final_data/train.skels \\
    --out  outputs/avatar_generator/train_sample.mp4 \\
    --max-frames 30

# Single image (frame 0)
python avatar_generator/run.py \\
    --pose  outputs/openpose_json/walking_keypoints \\
    --out   outputs/avatar_generator/walking_frame0.png \\
    --mode  image

# Contact sheet (9-frame preview grid, fast)
python avatar_generator/run.py \\
    --pose outputs/openpose_json/walking_keypoints \\
    --out  outputs/avatar_generator/walking_preview.png \\
    --mode contact-sheet \\
    --max-frames 9

# Batch: all procedural motions
python avatar_generator/run.py \\
    --batch-all \\
    --reference outputs/avatar_generator/reference/signer.jpg \\
    --out outputs/avatar_generator/

# Dry run: render pose maps only (no GPU, no model weights)
python avatar_generator/run.py \\
    --pose outputs/openpose_json/walking_keypoints \\
    --pose-maps-only \\
    --out  outputs/avatar_generator/walking_pose_maps/

# Custom config
python avatar_generator/run.py \\
    --pose   outputs/openpose_json/walking_keypoints \\
    --out    outputs/avatar_generator/walking.mp4 \\
    --config avatar_generator/config.yaml \\
    --steps  30 \\
    --width  1024 \\
    --height 1024
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _pose_maps_only(pose_source: Path, out_dir: Path) -> None:
    """Render ControlNet pose maps without running diffusion."""
    from avatar_generator.config import GeneratorConfig
    from avatar_generator.pose_loader import PoseLoader
    from avatar_generator.pose_renderer import PoseMapRenderer

    cfg = GeneratorConfig()
    frames = PoseLoader(cfg).load(pose_source)
    renderer = PoseMapRenderer(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {len(frames)} pose maps → {out_dir}")
    for i, frame in enumerate(frames):
        img = renderer.render(frame)
        img.save(str(out_dir / f"pose_{i:06d}.png"))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(frames)}")

    # Contact sheet for quick inspection.
    import numpy as np
    from PIL import Image
    indices = np.linspace(0, len(frames) - 1, min(9, len(frames)), dtype=int)
    thumbs = [renderer.render(frames[i]).resize((256, 256)) for i in indices]
    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    grid = Image.new("RGB", (cols * 256, rows * 256), (20, 20, 20))
    for j, t in enumerate(thumbs):
        r, c = divmod(j, cols)
        grid.paste(t, (c * 256, r * 256))
    grid.save(str(out_dir / "pose_grid.png"))
    print(f"Done. Pose grid → {out_dir / 'pose_grid.png'}")


def _batch_all(reference: Path | None, out_root: Path, cfg, batch_size: int) -> None:
    """Render all procedural motions from outputs/openpose_json/."""
    from avatar_generator.generator import AvatarGenerator

    json_root = ROOT / "outputs" / "openpose_json"
    motion_dirs = sorted(d for d in json_root.iterdir() if d.is_dir())
    if not motion_dirs:
        print(f"No motion directories found in {json_root}")
        return

    jobs = [
        {
            "pose_source": str(d),
            "output_path": str(out_root / d.name.replace("_keypoints", "") / f"{d.name.replace('_keypoints', '')}.mp4"),
            "reference_image": str(reference) if reference else None,
        }
        for d in motion_dirs
    ]

    print(f"Batch rendering {len(jobs)} motions")
    with AvatarGenerator(cfg) as gen:
        results = gen.generate_batch(jobs, batch_size=batch_size)

    for job, result in zip(jobs, results):
        status = "✅" if result else "❌"
        print(f"  {status} {Path(job['output_path']).name}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    p.add_argument("--pose", help=".skels file or OpenPose JSON directory.")
    p.add_argument("--reference", default=None, help="Reference signer image (optional).")
    p.add_argument("--batch-all", action="store_true",
                   help="Render all procedural motions from outputs/openpose_json/.")

    # Output
    p.add_argument("--out", required=True, help="Output .mp4, .png, or directory.")
    p.add_argument("--mode", choices=["video", "frames", "image", "contact-sheet"],
                   default="video", help="Output mode (default: video).")

    # Generation options
    p.add_argument("--prompt", default=None, help="Custom positive text prompt.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Limit frames rendered (useful for quick tests).")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Frames per diffusion call (default: 1).")
    p.add_argument("--frame-index", type=int, default=0,
                   help="Frame index for --mode image (default: 0).")

    # Config overrides
    p.add_argument("--config", default=None, help="Path to config.yaml.")
    p.add_argument("--steps", type=int, default=None, help="Override num_steps.")
    p.add_argument("--width", type=int, default=None, help="Override output width.")
    p.add_argument("--height", type=int, default=None, help="Override output height.")
    p.add_argument("--seed", type=int, default=None, help="Override random seed.")
    p.add_argument("--no-identity", action="store_true",
                   help="Disable identity conditioning (ignore --reference).")

    # Dry run
    p.add_argument("--pose-maps-only", action="store_true",
                   help="Render pose maps only — no diffusion, no GPU required.")

    args = p.parse_args()

    # Build config
    from avatar_generator.config import GeneratorConfig
    if args.config:
        cfg = GeneratorConfig.from_yaml(args.config)
    else:
        cfg = GeneratorConfig()

    # Apply CLI overrides
    if args.steps:
        cfg.num_steps = args.steps
    if args.width:
        cfg.width = args.width
    if args.height:
        cfg.height = args.height
    if args.seed is not None:
        cfg.seed = args.seed
    if args.no_identity:
        cfg.identity_strength = 0.0

    out_path = Path(args.out)

    # ── Batch all motions ──────────────────────────────────────────────
    if args.batch_all:
        ref = Path(args.reference) if args.reference else None
        _batch_all(ref, out_path, cfg, args.batch_size)
        return 0

    # ── Require --pose for all other modes ─────────────────────────────
    if not args.pose:
        print("Error: --pose is required (unless using --batch-all)")
        p.print_help()
        return 1

    pose_source = Path(args.pose)
    if not pose_source.exists():
        print(f"Error: pose source not found: {pose_source}")
        return 1

    # ── Pose maps only (dry run) ───────────────────────────────────────
    if args.pose_maps_only:
        _pose_maps_only(pose_source, out_path)
        return 0

    # ── Full generation ────────────────────────────────────────────────
    from avatar_generator.generator import AvatarGenerator

    reference = Path(args.reference) if args.reference else None

    with AvatarGenerator(cfg) as gen:
        if args.mode == "image":
            img = gen.generate_image(
                pose_source=pose_source,
                frame_index=args.frame_index,
                reference_image=reference,
                prompt=args.prompt,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(out_path))
            print(f"✅ Saved: {out_path}")

        elif args.mode == "contact-sheet":
            gen.generate_contact_sheet(
                pose_source=pose_source,
                output_path=out_path,
                n_frames=args.max_frames or 9,
                reference_image=reference,
                prompt=args.prompt,
            )
            print(f"✅ Contact sheet: {out_path}")

        else:
            gen.generate(
                pose_source=pose_source,
                output_path=out_path,
                reference_image=reference,
                prompt=args.prompt,
                max_frames=args.max_frames,
                batch_size=args.batch_size,
                mode=args.mode,
            )
            print(f"✅ Output: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
