"""RIFE temporal interpolation for avatar video output.

RIFE (Real-Time Intermediate Flow Estimation) inserts synthetic intermediate
frames between existing frames using optical flow estimation.  Applied after
diffusion rendering to increase the output frame rate (25fps → 50fps or
25fps → 100fps) and produce smoother slow-motion playback.

For sign language, 2× interpolation (25→50fps) is the recommended default:
    - Preserves hand motion clarity at normal playback speed
    - Enables 2× slow-motion review without additional rendering cost
    - Adds ~0.1s/frame overhead (negligible vs diffusion cost)

RIFE model versions:
    4.6  — recommended; best quality/speed tradeoff
    4.14 — highest quality; ~20% slower
    4.0  — fastest; lower quality on fast hand motion

Installation:
    The RIFE weights and inference code are not on PyPI.  Two options:
    Option A (recommended): practical-RIFE
        pip install practical-rife
    Option B: clone hzwer/Practical-RIFE and add to PYTHONPATH
        git clone https://github.com/hzwer/Practical-RIFE third_party/RIFE

This module tries Option A first, then Option B, then raises a clear error.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from avatar.config import RIFEConfig


def _require_rife():
    """Try to import RIFE inference module.  Returns the model class."""
    # Option A: practical-rife package
    try:
        from rife.model.RIFE_HDv3 import Model
        return Model, "practical-rife"
    except ImportError:
        pass

    # Option B: third_party/RIFE clone
    import sys
    rife_path = Path(__file__).resolve().parents[3] / "third_party" / "RIFE"
    if rife_path.exists():
        sys.path.insert(0, str(rife_path))
        try:
            from model.RIFE_HDv3 import Model
            return Model, "third_party"
        except ImportError:
            pass

    raise ImportError(
        "RIFE is not installed.\n"
        "Option A: pip install practical-rife\n"
        "Option B: git clone https://github.com/hzwer/Practical-RIFE "
        "third_party/RIFE\n"
        "Then download weights from the RIFE releases page."
    )


class RIFEInterpolator:
    """Temporal frame interpolator using RIFE optical flow.

    Usage:
        interp = RIFEInterpolator(cfg)
        interp.load()
        upsampled = interp.interpolate(frames, scale_factor=2)
        interp.unload()

    Or as a context manager:
        with RIFEInterpolator(cfg) as interp:
            upsampled = interp.interpolate(frames)
    """

    def __init__(
        self,
        cfg: Optional[RIFEConfig] = None,
        device: str = "cuda",
    ) -> None:
        self.cfg = cfg or RIFEConfig()
        self.device = device
        self._model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        print(f"[RIFE] Loading model v{self.cfg.model_version}…")
        t0 = time.perf_counter()
        try:
            import torch
            Model, source = _require_rife()
            self._model = Model()
            # Load weights from the standard RIFE weights path.
            weights_dir = (
                Path(__file__).resolve().parents[3]
                / "third_party" / "RIFE" / "train_log"
            )
            if weights_dir.exists():
                self._model.load_model(str(weights_dir), -1)
            else:
                # Try HuggingFace hub download.
                try:
                    from huggingface_hub import snapshot_download
                    weights_dir = snapshot_download(
                        repo_id=f"AlexWortega/RIFE",
                        local_dir=str(
                            Path(__file__).resolve().parents[3]
                            / "third_party" / "RIFE" / "train_log"
                        ),
                    )
                    self._model.load_model(str(weights_dir), -1)
                except Exception as e:
                    raise RuntimeError(
                        f"RIFE weights not found at {weights_dir}.\n"
                        f"Download from https://github.com/hzwer/Practical-RIFE/releases\n"
                        f"and place in third_party/RIFE/train_log/"
                    ) from e
            self._model.eval()
            if self.device == "cuda":
                import torch
                self._model = self._model.cuda()
            elapsed = time.perf_counter() - t0
            print(f"[RIFE] Loaded ({source}) in {elapsed:.1f}s")
            self._loaded = True
        except Exception as e:
            print(f"[RIFE] Load failed: {e}")
            raise

    def unload(self) -> None:
        if not self._loaded:
            return
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._loaded = False
        print("[RIFE] Unloaded.")

    def interpolate(
        self,
        frames: list[np.ndarray],
        scale_factor: Optional[int] = None,
    ) -> list[np.ndarray]:
        """Insert intermediate frames between each pair of input frames.

        Parameters
        ----------
        frames : list of (H, W, 3) uint8 RGB arrays
        scale_factor : int, optional
            2 = double frame count (25fps → 50fps)
            4 = quadruple frame count (25fps → 100fps)
            Defaults to cfg.scale_factor.

        Returns
        -------
        list of (H, W, 3) uint8 RGB arrays
            Length = len(frames) * scale_factor - (scale_factor - 1)
        """
        if not self._loaded:
            raise RuntimeError("Call load() before interpolate()")

        sf = scale_factor or self.cfg.scale_factor
        if sf == 1:
            return frames
        if len(frames) < 2:
            return frames

        import torch

        print(f"[RIFE] Interpolating {len(frames)} frames × {sf}…")
        t0 = time.perf_counter()

        result: list[np.ndarray] = []
        n_inserted = sf - 1   # intermediate frames between each pair

        for i in range(len(frames) - 1):
            result.append(frames[i])
            intermediates = self._interpolate_pair(frames[i], frames[i + 1], n_inserted)
            result.extend(intermediates)
        result.append(frames[-1])

        elapsed = time.perf_counter() - t0
        print(
            f"[RIFE] Done: {len(frames)} → {len(result)} frames in {elapsed:.1f}s "
            f"({elapsed / max(len(frames), 1) * 1000:.1f}ms/input-frame)"
        )
        return result

    def _interpolate_pair(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        n_intermediate: int,
    ) -> list[np.ndarray]:
        """Generate n_intermediate frames between frame_a and frame_b."""
        import torch

        def _to_tensor(f: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(f).float() / 255.0
            return t.permute(2, 0, 1).unsqueeze(0)   # (1, C, H, W)

        def _to_numpy(t: torch.Tensor) -> np.ndarray:
            arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
            return (arr * 255).clip(0, 255).astype(np.uint8)

        ta = _to_tensor(frame_a)
        tb = _to_tensor(frame_b)
        if self.device == "cuda":
            ta = ta.cuda()
            tb = tb.cuda()

        intermediates: list[np.ndarray] = []
        # For n_intermediate=1: timestep=0.5
        # For n_intermediate=3: timesteps=0.25, 0.5, 0.75
        timesteps = [(i + 1) / (n_intermediate + 1) for i in range(n_intermediate)]

        with torch.no_grad():
            for ts in timesteps:
                # RIFE inference: model.inference(img0, img1, timestep)
                mid = self._model.inference(ta, tb, timestep=ts)
                intermediates.append(_to_numpy(mid))

        return intermediates

    def __enter__(self) -> "RIFEInterpolator":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()


# ---------------------------------------------------------------------------
# Fallback: pure numpy linear interpolation (no RIFE weights needed)
# ---------------------------------------------------------------------------

class LinearInterpolator:
    """Frame-rate doubler using linear blending.

    Not as smooth as RIFE (no optical flow), but requires no model weights.
    Useful for testing the pipeline end-to-end before RIFE is installed.
    """

    def interpolate(
        self,
        frames: list[np.ndarray],
        scale_factor: int = 2,
    ) -> list[np.ndarray]:
        if scale_factor == 1 or len(frames) < 2:
            return frames
        result: list[np.ndarray] = []
        n_intermediate = scale_factor - 1
        for i in range(len(frames) - 1):
            result.append(frames[i])
            fa = frames[i].astype(np.float32)
            fb = frames[i + 1].astype(np.float32)
            for j in range(1, n_intermediate + 1):
                alpha = j / (n_intermediate + 1)
                mid = ((1 - alpha) * fa + alpha * fb).clip(0, 255).astype(np.uint8)
                result.append(mid)
        result.append(frames[-1])
        return result
