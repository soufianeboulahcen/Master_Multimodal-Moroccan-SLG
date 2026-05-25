"""Post-processing for generated avatar frames.

Two stages, both optional and independently usable:

  1. Frame interpolation (RIFE v4.6)
     Doubles the frame rate: 8 fps → 16 fps → 24/25 fps.
     Eliminates the choppy motion that AnimateDiff produces at its native
     8 fps.  Requires the RIFE weights in models/rife/ (downloaded once).

  2. Super-resolution upscale (Real-ESRGAN ×2 or ×4)
     Upscales 512×768 → 1024×1536 (×2) or 2048×3072 (×4).
     Recovers fine skin texture, hair detail, and sharpens edges that the
     VAE decoder softens.

Both stages operate on a list of PIL Images or on an MP4 file.

Usage (as a library):
    from scripts.avatar.postprocess import interpolate_frames, upscale_frames

    frames = [...]  # list of PIL Images at 8 fps
    frames = interpolate_frames(frames, target_fps=24, source_fps=8)
    frames = upscale_frames(frames, scale=2)

Usage (CLI):
    # Interpolate + upscale an existing avatar MP4
    python scripts/avatar/postprocess.py \\
        --input  outputs/avatar/videos/walking_avatar.mp4 \\
        --output outputs/avatar/videos/walking_avatar_hq.mp4 \\
        --interpolate --target-fps 24 \\
        --upscale --scale 2

    # Upscale only
    python scripts/avatar/postprocess.py \\
        --input  outputs/avatar/videos/sign_avatar.mp4 \\
        --output outputs/avatar/videos/sign_avatar_4k.mp4 \\
        --upscale --scale 4

    # Interpolate only (no upscale)
    python scripts/avatar/postprocess.py \\
        --input  outputs/avatar/videos/waving_avatar.mp4 \\
        --output outputs/avatar/videos/waving_avatar_smooth.mp4 \\
        --interpolate --target-fps 24
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

# ── Video I/O helpers ─────────────────────────────────────────────────────

def read_video_frames(video_path: Path) -> tuple[list[Image.Image], float]:
    """Read all frames from an MP4. Returns (frames, fps)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[Image.Image] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(bgr[:, :, ::-1]))  # BGR → RGB
    cap.release()
    print(f"[postprocess] read {len(frames)} frames @ {fps:.1f} fps from {video_path.name}")
    return frames, fps


def write_video_frames(
    frames: list[Image.Image],
    out_path: Path,
    fps: float,
) -> None:
    """Write PIL frames to MP4 (H.264, yuv420p)."""
    try:
        import imageio
    except ImportError:
        raise ImportError("imageio required: pip install imageio[ffmpeg]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264",
        quality=9, pixelformat="yuv420p",
    )
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()
    print(f"[postprocess] wrote {len(frames)} frames @ {fps:.1f} fps → {out_path}")


# ── Stage 1: RIFE frame interpolation ────────────────────────────────────

def _download_rife_weights(model_dir: Path) -> Path:
    """Download RIFE v4.6 weights if not already present."""
    weights_path = model_dir / "flownet.pkl"
    if weights_path.exists():
        return model_dir

    print("[postprocess] downloading RIFE v4.6 weights...")
    import urllib.request
    import zipfile

    model_dir.mkdir(parents=True, exist_ok=True)
    url = (
        "https://github.com/hzwer/Practical-RIFE/releases/download/"
        "v4.6/rife46.zip"
    )
    zip_path = model_dir / "rife46.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(model_dir)
    zip_path.unlink()
    print(f"[postprocess] RIFE weights → {model_dir}")
    return model_dir


def _rife_interpolate_pair(
    model,
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    n_midframes: int,
    device: str,
) -> list[np.ndarray]:
    """Generate n_midframes intermediate frames between frame_a and frame_b.

    Returns [mid_1, mid_2, ..., mid_n] as uint8 RGB arrays.
    """
    import torch

    def to_tensor(img: np.ndarray) -> "torch.Tensor":
        t = torch.from_numpy(img).float() / 255.0
        return t.permute(2, 0, 1).unsqueeze(0).to(device)

    ta = to_tensor(frame_a)
    tb = to_tensor(frame_b)

    mids = []
    # Recursively bisect for n_midframes = 2^k - 1 (e.g. 1, 3, 7)
    # For arbitrary counts, use evenly-spaced timesteps
    timesteps = [(i + 1) / (n_midframes + 1) for i in range(n_midframes)]
    for t in timesteps:
        with torch.no_grad():
            mid = model.inference(ta, tb, timestep=t)
        mid_np = (mid[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        mids.append(mid_np)
    return mids


def interpolate_frames(
    frames: list[Image.Image],
    target_fps: float = 24.0,
    source_fps: float = 8.0,
    device: Optional[str] = None,
    model_dir: Optional[Path] = None,
) -> list[Image.Image]:
    """Interpolate frames from source_fps to target_fps using RIFE.

    For each consecutive pair of input frames, inserts
    ceil(target_fps / source_fps) - 1 intermediate frames.

    Falls back to simple frame duplication if RIFE is unavailable.
    """
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if model_dir is None:
        model_dir = ROOT / "models" / "rife"

    ratio = target_fps / source_fps
    n_mid = max(int(round(ratio)) - 1, 0)

    if n_mid == 0:
        print("[postprocess] source_fps >= target_fps — no interpolation needed")
        return frames

    # Try RIFE
    try:
        import torch
        sys.path.insert(0, str(ROOT / "models" / "rife"))
        from model.RIFE import Model  # type: ignore[import]

        weights_dir = _download_rife_weights(model_dir)
        model = Model()
        model.load_model(str(weights_dir), -1)
        model.eval()
        model.device()

        print(f"[postprocess] RIFE interpolation: {len(frames)} frames "
              f"@ {source_fps} fps → ~{len(frames) * (n_mid + 1)} frames "
              f"@ {target_fps} fps")

        np_frames = [np.array(f) for f in frames]
        out: list[np.ndarray] = []
        for i in range(len(np_frames) - 1):
            out.append(np_frames[i])
            mids = _rife_interpolate_pair(model, np_frames[i], np_frames[i + 1],
                                          n_mid, device)
            out.extend(mids)
        out.append(np_frames[-1])

        result = [Image.fromarray(f) for f in out]
        print(f"[postprocess] interpolated → {len(result)} frames")
        return result

    except Exception as e:
        print(f"[postprocess] RIFE unavailable ({e}) — falling back to frame duplication")
        # Simple duplication fallback: repeat each frame n_mid+1 times
        out_pil: list[Image.Image] = []
        for frame in frames:
            for _ in range(n_mid + 1):
                out_pil.append(frame)
        return out_pil


# ── Stage 2: Real-ESRGAN upscale ─────────────────────────────────────────

def upscale_frames(
    frames: list[Image.Image],
    scale: int = 2,
    tile: int = 512,
    half: bool = True,
) -> list[Image.Image]:
    """Upscale frames with Real-ESRGAN.

    scale: 2 or 4.
    tile:  tile size for VRAM-limited inference (0 = no tiling).
    half:  use fp16 (faster, slightly lower quality).

    Falls back to Lanczos resize if Real-ESRGAN is not installed.
    """
    try:
        import cv2
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=scale,
        )
        model_url = (
            f"https://github.com/xinntao/Real-ESRGAN/releases/download/"
            f"v0.1.0/RealESRGAN_x{scale}plus.pth"
        )
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_url,
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=half,
        )

        print(f"[postprocess] Real-ESRGAN ×{scale}: {len(frames)} frames")
        out: list[Image.Image] = []
        for i, frame in enumerate(frames):
            img_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            enhanced, _ = upsampler.enhance(img_bgr, outscale=scale)
            out.append(Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)))
            if (i + 1) % 10 == 0 or i == len(frames) - 1:
                print(f"  upscaled {i+1}/{len(frames)}", flush=True)
        return out

    except ImportError:
        print("[postprocess] Real-ESRGAN not installed — falling back to Lanczos resize")
        w, h = frames[0].size
        new_w, new_h = w * scale, h * scale
        return [f.resize((new_w, new_h), Image.LANCZOS) for f in frames]


# ── Temporal denoising (flicker reduction) ───────────────────────────────

def reduce_flicker(
    frames: list[Image.Image],
    blend_alpha: float = 0.15,
) -> list[Image.Image]:
    """Reduce inter-frame flicker with a lightweight exponential moving average.

    blend_alpha: weight of the previous frame blended into the current one.
    0.0 = no blending (original), 0.3 = heavy smoothing.

    This is a CPU-only fallback for when optical-flow-based stabilization
    (e.g. FILM) is not available.  It slightly softens motion but
    significantly reduces background flicker and lighting jumps.
    """
    if blend_alpha <= 0.0 or len(frames) < 2:
        return frames

    out: list[Image.Image] = [frames[0]]
    prev = np.array(frames[0], dtype=np.float32)

    for frame in frames[1:]:
        curr = np.array(frame, dtype=np.float32)
        blended = prev * blend_alpha + curr * (1.0 - blend_alpha)
        blended_u8 = blended.clip(0, 255).astype(np.uint8)
        out.append(Image.fromarray(blended_u8))
        prev = blended  # carry forward the blended result

    return out


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",  required=True, help="Input MP4 path")
    p.add_argument("--output", required=True, help="Output MP4 path")
    p.add_argument("--interpolate", action="store_true",
                   help="Apply RIFE frame interpolation")
    p.add_argument("--target-fps", type=float, default=24.0,
                   help="Target FPS after interpolation (default: 24)")
    p.add_argument("--upscale", action="store_true",
                   help="Apply Real-ESRGAN upscale")
    p.add_argument("--scale", type=int, default=2, choices=[2, 4],
                   help="Upscale factor (default: 2)")
    p.add_argument("--tile", type=int, default=512,
                   help="Real-ESRGAN tile size (0 = no tiling, default: 512)")
    p.add_argument("--reduce-flicker", action="store_true",
                   help="Apply lightweight EMA flicker reduction")
    p.add_argument("--flicker-alpha", type=float, default=0.15,
                   help="EMA blend weight for flicker reduction (default: 0.15)")
    p.add_argument("--device", default=None,
                   help="Torch device (default: auto-detect)")
    args = p.parse_args()

    if not args.interpolate and not args.upscale and not args.reduce_flicker:
        print("[error] specify at least one of --interpolate, --upscale, "
              "--reduce-flicker", file=sys.stderr)
        return 1

    frames, src_fps = read_video_frames(Path(args.input))
    out_fps = src_fps

    if args.reduce_flicker:
        print(f"[postprocess] flicker reduction (alpha={args.flicker_alpha})")
        frames = reduce_flicker(frames, blend_alpha=args.flicker_alpha)

    if args.interpolate:
        frames = interpolate_frames(
            frames,
            target_fps=args.target_fps,
            source_fps=src_fps,
            device=args.device,
        )
        out_fps = args.target_fps

    if args.upscale:
        frames = upscale_frames(frames, scale=args.scale, tile=args.tile)

    write_video_frames(frames, Path(args.output), out_fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
