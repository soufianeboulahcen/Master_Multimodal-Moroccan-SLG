"""Abstract base class for diffusion rendering backends.

All backends (AnimateDiff, MimicMotion, ...) implement this interface so the
pipeline orchestrator can swap them without changing any calling code.

Contract:
    - load()   : load model weights into VRAM; idempotent
    - unload() : release VRAM; idempotent
    - render() : consume pose maps + identity conditioning → list of frames
    - is_loaded: property indicating whether weights are in VRAM
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class RenderResult:
    """Output of one backend render() call.

    Attributes
    ----------
    frames : list of (H, W, 3) uint8 RGB arrays
        Generated video frames in display order.
    fps : float
        Frame rate of the output (may differ from input if the backend
        performs internal temporal upsampling).
    metadata : dict
        Backend-specific diagnostics (inference time, seed, etc.).
    """
    frames: list[np.ndarray]
    fps: float
    metadata: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def save_frames(self, out_dir: str | Path, prefix: str = "frame_") -> list[Path]:
        """Save all frames as PNG files."""
        try:
            import cv2
        except ImportError as e:
            raise ImportError("opencv-python required to save frames") from e
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for i, frame in enumerate(self.frames):
            p = out_dir / f"{prefix}{i:06d}.png"
            cv2.imwrite(str(p), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            paths.append(p)
        return paths

    def to_video(
        self,
        out_path: str | Path,
        fps: Optional[float] = None,
        codec: str = "libx264",
        crf: int = 18,
    ) -> Path:
        """Encode frames to an MP4 file using ffmpeg via imageio."""
        try:
            import imageio
        except ImportError as e:
            raise ImportError("imageio[ffmpeg] required to encode video") from e
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        effective_fps = fps or self.fps
        writer = imageio.get_writer(
            str(out_path),
            fps=effective_fps,
            codec=codec,
            quality=None,
            output_params=["-crf", str(crf)],
        )
        for frame in self.frames:
            writer.append_data(frame)
        writer.close()
        return out_path


class RenderBackend(ABC):
    """Abstract diffusion rendering backend.

    Subclasses implement load(), unload(), and render().
    The pipeline calls them in that order.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load model weights into VRAM.  Must set self._loaded = True."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release VRAM.  Must set self._loaded = False."""
        ...

    @abstractmethod
    def render(
        self,
        body_map_paths: list[Path],
        hand_map_paths: list[Path],
        identity_conditioning: dict,
        prompt: str,
        seed: int,
    ) -> RenderResult:
        """Generate video frames from pose maps and identity conditioning.

        Parameters
        ----------
        body_map_paths : list of Path
            Paths to body ControlNet conditioning PNGs (one per frame).
        hand_map_paths : list of Path
            Paths to hand ControlNet conditioning PNGs (one per frame).
        identity_conditioning : dict
            Output of IdentityEncoder.encode().
        prompt : str
            Positive text prompt describing the desired appearance.
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        RenderResult
            Generated frames + metadata.
        """
        ...

    def __enter__(self) -> "RenderBackend":
        self.load()
        return self

    def __exit__(self, *_: Any) -> None:
        self.unload()

    @staticmethod
    def _load_image(path: Path) -> Any:
        """Load a PNG as a PIL Image (RGB)."""
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError("Pillow required") from e
        return Image.open(path).convert("RGB")

    @staticmethod
    def _load_images(paths: list[Path]) -> list[Any]:
        return [RenderBackend._load_image(p) for p in paths]
