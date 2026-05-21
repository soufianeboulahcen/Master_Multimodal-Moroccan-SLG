"""AvatarGenerator — top-level orchestrator.

Connects PoseLoader → PoseMapRenderer → DiffusionEngine → VideoAssembler
into a single callable interface.

Usage (image mode — one frame):
    gen = AvatarGenerator(cfg)
    gen.load()
    img = gen.generate_image(
        pose_source="outputs/openpose_json/walking_keypoints",
        frame_index=0,
        reference_image="signer.jpg",
    )
    img.save("avatar_frame.png")
    gen.unload()

Usage (video mode — full clip):
    with AvatarGenerator(cfg) as gen:
        gen.generate(
            pose_source="outputs/openpose_json/walking_keypoints",
            output_path="outputs/avatar_generator/walking.mp4",
            reference_image="signer.jpg",
        )

Usage (batch — multiple clips, weights loaded once):
    with AvatarGenerator(cfg) as gen:
        for src, out in jobs:
            gen.generate(pose_source=src, output_path=out)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from avatar_generator.config import GeneratorConfig
from avatar_generator.pose_loader import PoseLoader
from avatar_generator.pose_renderer import PoseMapRenderer
from avatar_generator.diffusion_engine import DiffusionEngine
from avatar_generator.video_assembler import VideoAssembler


class AvatarGenerator:
    """End-to-end avatar generation: pose sequence → images/video."""

    def __init__(self, cfg: Optional[GeneratorConfig] = None) -> None:
        self.cfg = cfg or GeneratorConfig()
        self._loader = PoseLoader(self.cfg)
        self._renderer = PoseMapRenderer(self.cfg)
        self._engine = DiffusionEngine(self.cfg)
        self._assembler = VideoAssembler(self.cfg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load diffusion model weights into VRAM.  Idempotent."""
        self._engine.load()

    def unload(self) -> None:
        """Release VRAM.  Idempotent."""
        self._engine.unload()

    def __enter__(self) -> "AvatarGenerator":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def generate(
        self,
        pose_source: str | Path,
        output_path: str | Path,
        reference_image: Optional[str | Path] = None,
        prompt: Optional[str] = None,
        max_frames: Optional[int] = None,
        batch_size: int = 1,
        mode: str = "video",
    ) -> Path:
        """Generate avatar video or frame directory from a pose sequence.

        Parameters
        ----------
        pose_source : str | Path
            .skels file  OR  directory of *_keypoints.json files.
        output_path : str | Path
            Destination .mp4 (mode="video") or directory (mode="frames").
        reference_image : str | Path, optional
            Reference signer photo for IP-Adapter identity conditioning.
        prompt : str, optional
            Positive text prompt.  Defaults to cfg.default_prompt.
        max_frames : int, optional
            Truncate the pose sequence to this many frames.  Useful for
            quick tests without rendering a full clip.
        batch_size : int
            Frames per diffusion call.  1 is safest; increase for throughput.
        mode : "video" | "frames"
            Output format.

        Returns
        -------
        Path to the output file or directory.
        """
        t_start = time.perf_counter()
        pose_source = Path(pose_source)
        output_path = Path(output_path)

        # Step 1: Load pose sequence.
        print(f"\n[AvatarGenerator] Loading poses from {pose_source.name}")
        frames = self._loader.load(pose_source)
        if max_frames is not None:
            frames = frames[:max_frames]
        print(f"  {len(frames)} frames loaded")

        # Step 2: Render pose maps.
        print("[AvatarGenerator] Rendering pose maps…")
        pose_maps = self._renderer.render_sequence(frames)

        # Step 3: Load reference image if provided.
        ref_pil = None
        if reference_image is not None:
            ref_pil = self._load_reference(Path(reference_image))
            print(f"  reference image: {Path(reference_image).name}")

        # Step 4: Diffusion generation.
        print(f"[AvatarGenerator] Generating {len(pose_maps)} frames…")
        generated = self._engine.generate_frames(
            pose_maps=pose_maps,
            prompt=prompt,
            reference_image=ref_pil,
            batch_size=batch_size,
        )

        # Step 5: Assemble output.
        if mode == "video":
            result = self._assembler.to_video(generated, output_path)
        elif mode == "frames":
            paths = self._assembler.save_frames(generated, output_path)
            result = output_path
            print(f"[AvatarGenerator] {len(paths)} frames saved to {output_path}")
        else:
            raise ValueError(f"mode must be 'video' or 'frames', got {mode!r}")

        # Optionally also save individual frames alongside the video.
        if self.cfg.save_frames and mode == "video":
            frames_dir = output_path.parent / self.cfg.frames_subdir
            self._assembler.save_frames(generated, frames_dir)

        elapsed = time.perf_counter() - t_start
        print(f"[AvatarGenerator] Done in {elapsed:.1f}s → {result}")
        return result

    def generate_image(
        self,
        pose_source: str | Path,
        frame_index: int = 0,
        reference_image: Optional[str | Path] = None,
        prompt: Optional[str] = None,
    ) -> "PIL.Image.Image":
        """Generate a single avatar image from one frame of a pose sequence.

        Parameters
        ----------
        pose_source : str | Path
            .skels file or JSON directory.
        frame_index : int
            Which frame to render (0-based).
        reference_image : str | Path, optional
            Reference signer photo.
        prompt : str, optional
            Positive text prompt.

        Returns
        -------
        PIL Image (RGB, cfg.width × cfg.height).
        """
        frames = self._loader.load(Path(pose_source))
        if frame_index >= len(frames):
            raise IndexError(
                f"frame_index={frame_index} out of range for "
                f"{len(frames)}-frame sequence"
            )
        pose_map = self._renderer.render(frames[frame_index])
        ref_pil = self._load_reference(Path(reference_image)) if reference_image else None
        return self._engine.generate_single(pose_map, prompt, ref_pil)

    def generate_contact_sheet(
        self,
        pose_source: str | Path,
        output_path: str | Path,
        n_frames: int = 9,
        reference_image: Optional[str | Path] = None,
        prompt: Optional[str] = None,
    ) -> Path:
        """Generate a grid of N evenly-spaced frames for quick preview."""
        import numpy as np

        frames = self._loader.load(Path(pose_source))
        indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int).tolist()
        selected = [frames[i] for i in indices]
        pose_maps = self._renderer.render_sequence(selected)
        ref_pil = self._load_reference(Path(reference_image)) if reference_image else None
        generated = self._engine.generate_frames(pose_maps, prompt, ref_pil)
        return self._assembler.save_contact_sheet(generated, output_path)

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        jobs: list[dict],
        reference_image: Optional[str | Path] = None,
        prompt: Optional[str] = None,
        batch_size: int = 1,
    ) -> list[Path]:
        """Run multiple generate() calls with weights loaded once.

        Parameters
        ----------
        jobs : list of dicts with keys:
            pose_source  (required)
            output_path  (required)
            reference_image  (optional, overrides the shared reference_image)
            prompt  (optional)
            max_frames  (optional)
            mode  (optional, default "video")

        Returns
        -------
        list of output Paths (one per job).
        """
        results: list[Path] = []
        for i, job in enumerate(jobs):
            print(f"\n[AvatarGenerator] Batch job {i + 1}/{len(jobs)}")
            try:
                r = self.generate(
                    pose_source=job["pose_source"],
                    output_path=job["output_path"],
                    reference_image=job.get("reference_image", reference_image),
                    prompt=job.get("prompt", prompt),
                    max_frames=job.get("max_frames"),
                    batch_size=batch_size,
                    mode=job.get("mode", "video"),
                )
                results.append(r)
            except Exception as e:
                print(f"  ❌ Job {i + 1} failed: {e}")
                results.append(None)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_reference(path: Path):
        """Load a reference image as a PIL Image (RGB)."""
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError("Pillow required") from e
        if not path.exists():
            raise FileNotFoundError(f"Reference image not found: {path}")
        return Image.open(path).convert("RGB")
