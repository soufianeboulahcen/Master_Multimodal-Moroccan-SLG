"""Avatar rendering pipeline orchestrator.

Connects all subsystems in the correct order:

    1. PoseMapRenderer   — OpenPose JSON → RGB ControlNet conditioning maps
    2. IdentityEncoder   — reference image → identity conditioning tensors
    3. RenderBackend     — pose maps + identity → raw video frames
    4. RIFEInterpolator  — raw frames → temporally upsampled frames
    5. RenderResult      — frames → MP4 video

The pipeline is stateless between calls: each render() invocation is
independent.  Model weights are loaded once and reused across calls within
the same session (call unload() to release VRAM between sessions).

Usage:
    # Minimal: render from an existing OpenPose JSON directory
    cfg = AvatarConfig()
    pipe = AvatarPipeline(cfg)
    pipe.load()
    result = pipe.render(
        json_dir="outputs/openpose_json/أَنَا_keypoints",
        reference_image="outputs/avatar/reference_images/signer.jpg",
        output_path="outputs/avatar/أَنَا/أَنَا_avatar.mp4",
    )
    pipe.unload()

    # Or as a context manager (auto-unload):
    with AvatarPipeline(cfg) as pipe:
        result = pipe.render(...)

    # Batch mode (load once, render many):
    with AvatarPipeline(cfg) as pipe:
        for sign, json_dir in sign_dirs.items():
            pipe.render(json_dir=json_dir, reference_image=ref, ...)
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

from avatar.config import AvatarConfig, BackendType
from avatar.conditioning.pose_to_controlnet import PoseMapRenderer
from avatar.conditioning.identity_encoder import IdentityEncoder
from avatar.backends.base import RenderBackend, RenderResult
from avatar.interpolation.rife import RIFEInterpolator, LinearInterpolator


class AvatarPipeline:
    """End-to-end avatar video generation pipeline.

    Manages the lifecycle of all sub-components and orchestrates the
    rendering workflow from OpenPose JSON to final MP4.
    """

    def __init__(self, cfg: Optional[AvatarConfig] = None) -> None:
        self.cfg = cfg or AvatarConfig()
        self._pose_renderer = PoseMapRenderer(self.cfg.pose_map)
        self._identity_encoder = IdentityEncoder(self.cfg.identity, self.cfg.device)
        self._backend: Optional[RenderBackend] = None
        self._interpolator: Optional[RIFEInterpolator] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all model weights into VRAM.  Idempotent."""
        if self._loaded:
            return

        # Select and load the diffusion backend.
        self._backend = self._build_backend()
        self._backend.load()

        # Load RIFE if interpolation is enabled.
        if self.cfg.interpolate:
            self._interpolator = RIFEInterpolator(self.cfg.rife, self.cfg.device)
            try:
                self._interpolator.load()
            except Exception as e:
                print(f"[AvatarPipeline] RIFE load failed ({e}); falling back to linear interpolation")
                self._interpolator = None   # LinearInterpolator used as fallback

        self._loaded = True
        print("[AvatarPipeline] Ready.")

    def unload(self) -> None:
        """Release all VRAM.  Idempotent."""
        if not self._loaded:
            return
        if self._backend is not None:
            self._backend.unload()
        if self._interpolator is not None and hasattr(self._interpolator, "unload"):
            self._interpolator.unload()
        self._loaded = False
        print("[AvatarPipeline] Unloaded.")

    def __enter__(self) -> "AvatarPipeline":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()

    # ------------------------------------------------------------------
    # Main render entry point
    # ------------------------------------------------------------------

    def render(
        self,
        json_dir: str | Path,
        reference_image: str | Path,
        output_path: str | Path,
        prompt: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> RenderResult:
        """Render a sign from OpenPose JSON to a photorealistic avatar video.

        Parameters
        ----------
        json_dir : str | Path
            Directory containing per-frame *_keypoints.json files.
            Produced by mosl/pose/export_openpose_json.py or the procedural
            generator in scripts/generate_openpose/.
        reference_image : str | Path
            Path to a reference photo of the target signer.
            Used for identity conditioning (InstantID / IP-Adapter).
        output_path : str | Path
            Destination MP4 file path.
        prompt : str, optional
            Positive text prompt.  Defaults to a sign-language-appropriate
            description.
        seed : int, optional
            RNG seed.  Defaults to cfg.seed.
        verbose : bool
            Print progress to stdout.

        Returns
        -------
        RenderResult
            Contains the generated frames and metadata.
        """
        if not self._loaded:
            raise RuntimeError("Call load() (or use as context manager) before render()")

        json_dir = Path(json_dir)
        reference_image = Path(reference_image)
        output_path = Path(output_path)
        effective_seed = seed if seed is not None else self.cfg.seed
        effective_prompt = prompt or self._default_prompt()

        t_pipeline_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Step 1: Render pose maps
        # ------------------------------------------------------------------
        if verbose:
            print(f"\n[AvatarPipeline] Step 1/4 — Rendering pose maps from {json_dir.name}")

        with tempfile.TemporaryDirectory(prefix="avatar_pose_") as tmp:
            tmp_path = Path(tmp)
            body_dir = tmp_path / "body"
            hand_dir = tmp_path / "hand"

            body_paths, hand_paths = self._pose_renderer.render_and_save(
                json_dir, body_dir, hand_dir, verbose=verbose
            )

            if verbose:
                print(f"  {len(body_paths)} body maps + {len(hand_paths)} hand maps")

            # ------------------------------------------------------------------
            # Step 2: Encode identity
            # ------------------------------------------------------------------
            if verbose:
                print(f"[AvatarPipeline] Step 2/4 — Encoding identity from {reference_image.name}")

            identity_conditioning = self._identity_encoder.encode(reference_image)
            # Pass json_dir through for MimicMotion's DWPose conversion.
            identity_conditioning["_json_dir"] = str(json_dir)

            # ------------------------------------------------------------------
            # Step 3: Diffusion rendering
            # ------------------------------------------------------------------
            if verbose:
                print(f"[AvatarPipeline] Step 3/4 — Diffusion rendering ({self.cfg.backend.value})")

            result = self._backend.render(
                body_map_paths=body_paths,
                hand_map_paths=hand_paths,
                identity_conditioning=identity_conditioning,
                prompt=effective_prompt,
                seed=effective_seed,
            )

            # ------------------------------------------------------------------
            # Step 4: Temporal interpolation
            # ------------------------------------------------------------------
            if self.cfg.interpolate:
                if verbose:
                    print(
                        f"[AvatarPipeline] Step 4/4 — RIFE interpolation "
                        f"(×{self.cfg.rife.scale_factor})"
                    )
                if self._interpolator is not None:
                    result.frames = self._interpolator.interpolate(
                        result.frames, scale_factor=self.cfg.rife.scale_factor
                    )
                    result.fps = result.fps * self.cfg.rife.scale_factor
                else:
                    # Linear fallback.
                    lin = LinearInterpolator()
                    result.frames = lin.interpolate(
                        result.frames, scale_factor=self.cfg.rife.scale_factor
                    )
                    result.fps = result.fps * self.cfg.rife.scale_factor
                    if verbose:
                        print("  (used linear interpolation fallback)")
            else:
                if verbose:
                    print("[AvatarPipeline] Step 4/4 — Interpolation skipped")

            # ------------------------------------------------------------------
            # Encode to video
            # ------------------------------------------------------------------
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_video(
                output_path,
                fps=result.fps,
                codec=self.cfg.output_codec,
                crf=self.cfg.output_crf,
            )

        total_elapsed = time.perf_counter() - t_pipeline_start
        result.metadata["pipeline_total_time_s"] = total_elapsed
        result.metadata["output_path"] = str(output_path)

        if verbose:
            print(
                f"\n[AvatarPipeline] Done in {total_elapsed:.1f}s → {output_path}"
            )

        return result

    # ------------------------------------------------------------------
    # Batch rendering
    # ------------------------------------------------------------------

    def render_batch(
        self,
        jobs: list[dict],
        verbose: bool = True,
    ) -> list[RenderResult]:
        """Render multiple signs in sequence, reusing loaded weights.

        Parameters
        ----------
        jobs : list of dict
            Each dict must have keys: json_dir, reference_image, output_path.
            Optional keys: prompt, seed.

        Returns
        -------
        list of RenderResult
        """
        if not self._loaded:
            raise RuntimeError("Call load() before render_batch()")

        results: list[RenderResult] = []
        for i, job in enumerate(jobs):
            if verbose:
                print(f"\n[AvatarPipeline] Batch job {i + 1}/{len(jobs)}: {job.get('output_path', '?')}")
            try:
                r = self.render(
                    json_dir=job["json_dir"],
                    reference_image=job["reference_image"],
                    output_path=job["output_path"],
                    prompt=job.get("prompt"),
                    seed=job.get("seed"),
                    verbose=verbose,
                )
                results.append(r)
            except Exception as e:
                print(f"[AvatarPipeline] Job {i + 1} failed: {e}")
                results.append(RenderResult(frames=[], fps=25.0, metadata={"error": str(e)}))
        return results

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _build_backend(self) -> RenderBackend:
        from avatar.backends.animatediff import AnimateDiffBackend
        from avatar.backends.mimicmotion import MimicMotionBackend

        backend_type = self.cfg.backend

        if backend_type == BackendType.AUTO:
            # Default to AnimateDiff (better for hand-critical sign language).
            backend_type = BackendType.ANIMATEDIFF

        if backend_type == BackendType.ANIMATEDIFF:
            return AnimateDiffBackend(
                animatediff_cfg=self.cfg.animatediff,
                pose_cfg=self.cfg.pose_map,
                device=self.cfg.device,
            )
        elif backend_type == BackendType.MIMICMOTION:
            return MimicMotionBackend(
                mimicmotion_cfg=self.cfg.mimicmotion,
                pose_cfg=self.cfg.pose_map,
                device=self.cfg.device,
            )
        else:
            raise ValueError(f"Unknown backend: {backend_type}")

    @staticmethod
    def _default_prompt() -> str:
        return (
            "a person performing Moroccan Sign Language, photorealistic, "
            "high quality, detailed hands and fingers, clear facial expression, "
            "studio lighting, neutral background, 4k resolution"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: AvatarConfig) -> "AvatarPipeline":
        return cls(cfg)

    @classmethod
    def from_json(cls, config_path: str | Path) -> "AvatarPipeline":
        return cls(AvatarConfig.from_json(config_path))
