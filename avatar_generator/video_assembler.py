"""Assemble generated PIL Image frames into an MP4 video or a PNG grid.

Two output modes:
    video  — H.264 MP4 via imageio-ffmpeg
    frames — individual PNG files in a directory

The assembler is intentionally separate from the diffusion engine so frames
can be saved incrementally during long generation runs (crash recovery).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from avatar_generator.config import GeneratorConfig


class VideoAssembler:
    """Assembles PIL Image frames into video or PNG outputs."""

    def __init__(self, cfg: Optional[GeneratorConfig] = None) -> None:
        self.cfg = cfg or GeneratorConfig()

    # ------------------------------------------------------------------
    # MP4 video
    # ------------------------------------------------------------------

    def to_video(
        self,
        frames: list,
        output_path: str | Path,
        fps: Optional[int] = None,
    ) -> Path:
        """Encode frames to an MP4 file.

        Parameters
        ----------
        frames : list of PIL Images or (H, W, 3) uint8 numpy arrays
        output_path : destination .mp4 path
        fps : frame rate (defaults to cfg.output_fps)

        Returns
        -------
        Path to the written file.
        """
        try:
            import imageio
        except ImportError as e:
            raise ImportError(
                "imageio[ffmpeg] is required for video encoding.\n"
                "Install: pip install imageio[ffmpeg]"
            ) from e

        import numpy as np

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        effective_fps = fps or self.cfg.output_fps

        print(f"[VideoAssembler] Encoding {len(frames)} frames → {output_path}")
        t0 = time.perf_counter()

        writer = imageio.get_writer(
            str(output_path),
            fps=effective_fps,
            codec=self.cfg.output_codec,
            quality=None,
            output_params=["-crf", str(self.cfg.output_crf), "-pix_fmt", "yuv420p"],
        )
        for frame in frames:
            if hasattr(frame, "numpy"):
                arr = np.array(frame)
            elif hasattr(frame, "convert"):
                arr = np.array(frame.convert("RGB"))
            else:
                arr = np.asarray(frame)
            writer.append_data(arr)
        writer.close()

        elapsed = time.perf_counter() - t0
        size_mb = output_path.stat().st_size / 1e6
        print(f"[VideoAssembler] Done: {size_mb:.1f} MB in {elapsed:.1f}s → {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # PNG frames
    # ------------------------------------------------------------------

    def save_frames(
        self,
        frames: list,
        out_dir: str | Path,
        prefix: str = "frame_",
    ) -> list[Path]:
        """Save each frame as a numbered PNG file.

        Returns list of written paths.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for i, frame in enumerate(frames):
            p = out_dir / f"{prefix}{i:06d}.png"
            if hasattr(frame, "save"):
                frame.save(str(p))
            else:
                import cv2
                import numpy as np
                arr = np.asarray(frame)
                cv2.imwrite(str(p), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            paths.append(p)
        return paths

    # ------------------------------------------------------------------
    # Contact sheet (debug grid)
    # ------------------------------------------------------------------

    def save_contact_sheet(
        self,
        frames: list,
        output_path: str | Path,
        cols: int = 4,
        thumb_size: tuple[int, int] = (256, 256),
    ) -> Path:
        """Save a grid of thumbnail frames for quick visual inspection."""
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError("Pillow required for contact sheet") from e

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        thumbs = []
        for frame in frames:
            if not hasattr(frame, "resize"):
                import numpy as np
                frame = Image.fromarray(np.asarray(frame))
            thumbs.append(frame.resize(thumb_size, Image.LANCZOS))

        rows = (len(thumbs) + cols - 1) // cols
        W, H = thumb_size
        grid = Image.new("RGB", (cols * W, rows * H), color=(20, 20, 20))
        for i, thumb in enumerate(thumbs):
            r, c = divmod(i, cols)
            grid.paste(thumb, (c * W, r * H))

        grid.save(str(output_path))
        print(f"[VideoAssembler] Contact sheet ({len(thumbs)} frames) → {output_path}")
        return output_path
