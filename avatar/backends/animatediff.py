"""AnimateDiff rendering backend.

Generates photorealistic video frames from OpenPose conditioning maps using
AnimateDiff + SDXL + dual ControlNet (body + hand branches) + IP-Adapter or
InstantID for identity preservation.

Architecture:
    Base model  : SDXL 1.0 (stabilityai/stable-diffusion-xl-base-1.0)
    Motion      : AnimateDiff SDXL motion adapter (guoyww/animatediff-motion-adapter-sdxl-beta)
    ControlNet  : thibaud/controlnet-openpose-sdxl-1.0 (two instances: body + hand)
    Identity    : IP-Adapter-Plus-Face SDXL or InstantID (injected post-load)

Sliding window strategy for long clips:
    AnimateDiff processes context_frames (default 16) at a time.  For clips
    longer than context_frames, we use a sliding window with context_overlap
    frames of overlap between adjacent windows.  The overlapping frames are
    blended in latent space using a linear ramp to avoid hard seams.

    Window schedule for a 100-frame clip with context=16, overlap=4:
        window 0: frames  0–15
        window 1: frames 12–27   (4-frame overlap with window 0)
        window 2: frames 24–39
        ...
    Blending: the first `overlap` frames of each window are blended with the
    last `overlap` frames of the previous window using a linear alpha ramp.

Memory management:
    The full stack (SDXL + 2× ControlNet + IP-Adapter) requires ~28–36 GB
    VRAM.  On the DGX Spark GB10 (128 GB unified) this is comfortable.
    xformers memory-efficient attention is enabled by default.
    VAE slicing is enabled to reduce peak VRAM during decode.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from avatar.backends.base import RenderBackend, RenderResult
from avatar.config import AnimateDiffConfig, PoseMapConfig


class AnimateDiffBackend(RenderBackend):
    """AnimateDiff + SDXL + dual ControlNet rendering backend."""

    def __init__(
        self,
        animatediff_cfg: Optional[AnimateDiffConfig] = None,
        pose_cfg: Optional[PoseMapConfig] = None,
        device: str = "cuda",
    ) -> None:
        super().__init__(device)
        self.cfg = animatediff_cfg or AnimateDiffConfig()
        self.pose_cfg = pose_cfg or PoseMapConfig()
        self._pipe = None
        self._controlnet_body = None
        self._controlnet_hand = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load AnimateDiff + SDXL + ControlNet weights into VRAM."""
        if self._loaded:
            return

        print("[AnimateDiff] Loading model stack…")
        t0 = time.perf_counter()

        try:
            import torch
            from diffusers import (
                AnimateDiffSDXLPipeline,
                ControlNetModel,
                MotionAdapter,
                DDIMScheduler,
            )
        except ImportError as e:
            raise ImportError(
                "diffusers>=0.27.0 is required for AnimateDiff rendering.\n"
                "Install: pip install diffusers transformers accelerate"
            ) from e

        dtype = getattr(torch, self.cfg.torch_dtype)

        # Load motion adapter.
        print(f"  motion adapter: {self.cfg.motion_adapter_id}")
        adapter = MotionAdapter.from_pretrained(
            self.cfg.motion_adapter_id, torch_dtype=dtype
        )

        # Load dual ControlNet instances.
        print(f"  controlnet (body): {self.cfg.controlnet_openpose_id}")
        self._controlnet_body = ControlNetModel.from_pretrained(
            self.cfg.controlnet_openpose_id, torch_dtype=dtype
        )
        print(f"  controlnet (hand): {self.cfg.controlnet_openpose_id}")
        self._controlnet_hand = ControlNetModel.from_pretrained(
            self.cfg.controlnet_openpose_id, torch_dtype=dtype
        )

        # Build AnimateDiff SDXL pipeline with both ControlNets.
        print(f"  base model: {self.cfg.base_model_id}")
        self._pipe = AnimateDiffSDXLPipeline.from_pretrained(
            self.cfg.base_model_id,
            motion_adapter=adapter,
            controlnet=[self._controlnet_body, self._controlnet_hand],
            torch_dtype=dtype,
        ).to(self.device)

        # Memory optimisations.
        if self.cfg.enable_xformers:
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
                print("  xformers: enabled")
            except Exception:
                print("  xformers: not available, skipping")

        if self.cfg.enable_vae_slicing:
            self._pipe.enable_vae_slicing()
            print("  VAE slicing: enabled")

        if self.cfg.enable_model_cpu_offload:
            self._pipe.enable_model_cpu_offload()
            print("  CPU offload: enabled")

        elapsed = time.perf_counter() - t0
        print(f"[AnimateDiff] Loaded in {elapsed:.1f}s")
        self._loaded = True

    def unload(self) -> None:
        """Release VRAM."""
        if not self._loaded:
            return
        try:
            import torch
        except ImportError:
            pass
        self._pipe = None
        self._controlnet_body = None
        self._controlnet_hand = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._loaded = False
        print("[AnimateDiff] Unloaded.")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self,
        body_map_paths: list[Path],
        hand_map_paths: list[Path],
        identity_conditioning: dict,
        prompt: str = (
            "a person signing in Moroccan Sign Language, photorealistic, "
            "high quality, detailed hands, studio lighting, 4k"
        ),
        seed: int = 42,
    ) -> RenderResult:
        """Generate frames using sliding-window AnimateDiff inference."""
        if not self._loaded:
            raise RuntimeError("Call load() before render()")

        import torch

        n_frames = len(body_map_paths)
        if n_frames == 0:
            raise ValueError("body_map_paths is empty")
        if len(hand_map_paths) != n_frames:
            raise ValueError(
                f"body_map_paths ({n_frames}) and hand_map_paths "
                f"({len(hand_map_paths)}) must have the same length"
            )

        # Inject identity conditioning into the pipeline.
        from avatar.conditioning.identity_encoder import IdentityEncoder
        from avatar.config import IdentityConfig
        id_cfg = IdentityConfig()
        id_encoder = IdentityEncoder(id_cfg, self.device)
        id_encoder.inject_into_pipeline(self._pipe, identity_conditioning)

        # Determine reference image for IP-Adapter.
        ref_image = identity_conditioning.get("ref_image")

        # Build window schedule.
        windows = self._build_windows(n_frames)
        print(f"[AnimateDiff] Rendering {n_frames} frames in {len(windows)} window(s)")

        all_latents: list[Optional[np.ndarray]] = [None] * n_frames
        generator = torch.Generator(device=self.device).manual_seed(seed)

        t_total = time.perf_counter()
        for w_idx, (start, end) in enumerate(windows):
            w_body = self._load_images(body_map_paths[start:end])
            w_hand = self._load_images(hand_map_paths[start:end])
            w_len = end - start

            print(f"  window {w_idx + 1}/{len(windows)}: frames {start}–{end - 1}")
            t0 = time.perf_counter()

            pipe_kwargs = dict(
                prompt=prompt,
                negative_prompt=self.cfg.negative_prompt,
                num_frames=w_len,
                num_inference_steps=self.cfg.num_inference_steps,
                guidance_scale=self.cfg.guidance_scale,
                controlnet_conditioning_scale=[
                    self.pose_cfg.body_controlnet_weight,
                    self.pose_cfg.hand_controlnet_weight,
                ],
                image=[[img for img in w_body], [img for img in w_hand]],
                generator=generator,
                output_type="np",   # return numpy arrays directly
            )
            if ref_image is not None:
                pipe_kwargs["ip_adapter_image"] = ref_image

            output = self._pipe(**pipe_kwargs)
            # output.frames: (1, T, H, W, C) float32 in [0, 1]
            window_frames_f = output.frames[0]   # (T, H, W, C)
            window_frames = (window_frames_f * 255).clip(0, 255).astype(np.uint8)

            elapsed_w = time.perf_counter() - t0
            print(f"    {elapsed_w:.1f}s  ({elapsed_w / w_len:.2f}s/frame)")

            # Place frames into the output buffer, blending the overlap region.
            for local_i, global_i in enumerate(range(start, end)):
                if all_latents[global_i] is None:
                    all_latents[global_i] = window_frames[local_i]
                else:
                    # Blend: linear ramp over the overlap region.
                    overlap = self.cfg.context_overlap
                    alpha = local_i / max(overlap, 1)
                    alpha = min(alpha, 1.0)
                    prev = all_latents[global_i].astype(np.float32)
                    curr = window_frames[local_i].astype(np.float32)
                    blended = (1 - alpha) * prev + alpha * curr
                    all_latents[global_i] = blended.clip(0, 255).astype(np.uint8)

        total_elapsed = time.perf_counter() - t_total
        frames = [f for f in all_latents if f is not None]
        print(
            f"[AnimateDiff] Done: {len(frames)} frames in {total_elapsed:.1f}s "
            f"({total_elapsed / max(len(frames), 1):.2f}s/frame)"
        )

        return RenderResult(
            frames=frames,
            fps=25.0,
            metadata={
                "backend": "animatediff",
                "n_windows": len(windows),
                "total_time_s": total_elapsed,
                "seed": seed,
                "prompt": prompt,
            },
        )

    # ------------------------------------------------------------------
    # Window schedule
    # ------------------------------------------------------------------

    def _build_windows(self, n_frames: int) -> list[tuple[int, int]]:
        """Build (start, end) index pairs for the sliding window schedule."""
        ctx = self.cfg.context_frames
        overlap = self.cfg.context_overlap
        stride = ctx - overlap

        if n_frames <= ctx:
            return [(0, n_frames)]

        windows: list[tuple[int, int]] = []
        start = 0
        while start < n_frames:
            end = min(start + ctx, n_frames)
            windows.append((start, end))
            if end == n_frames:
                break
            start += stride
        return windows
