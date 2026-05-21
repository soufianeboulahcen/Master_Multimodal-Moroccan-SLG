"""Diffusion inference engine: SDXL + ControlNet-OpenPose + IP-Adapter.

Wraps the HuggingFace diffusers pipeline into a simple interface:

    engine = DiffusionEngine(cfg)
    engine.load()
    frames = engine.generate_frames(pose_maps, prompt, reference_image)
    engine.unload()

Architecture:
    Base model  : SDXL 1.0
    ControlNet  : ControlNet-OpenPose SDXL (single branch)
    Identity    : IP-Adapter-Plus-Face SDXL (optional, skipped if no reference)

Frame generation is done one frame at a time (image mode) or in batches.
For temporal consistency across frames, the same generator seed is used and
the prompt is held constant.  Full video-diffusion consistency (AnimateDiff)
is handled by the avatar/ module; this engine is the simpler single-frame path.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from avatar_generator.config import GeneratorConfig


class DiffusionEngine:
    """SDXL + ControlNet-OpenPose inference engine."""

    def __init__(self, cfg: Optional[GeneratorConfig] = None) -> None:
        self.cfg = cfg or GeneratorConfig()
        self._pipe = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load SDXL + ControlNet (+ IP-Adapter if identity_strength > 0)."""
        if self._loaded:
            return

        print("[DiffusionEngine] Loading model stack…")
        t0 = time.perf_counter()

        try:
            import torch
            from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
        except ImportError as e:
            raise ImportError(
                "diffusers>=0.27.0 is required.\n"
                "Install: pip install diffusers transformers accelerate safetensors"
            ) from e

        cfg = self.cfg
        dtype = getattr(torch, cfg.torch_dtype)

        print(f"  controlnet: {cfg.controlnet_id}")
        controlnet = ControlNetModel.from_pretrained(
            cfg.controlnet_id, torch_dtype=dtype
        )

        print(f"  base model: {cfg.base_model_id}")
        # Use a high-quality VAE for better colour fidelity.
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype
        )
        self._pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            cfg.base_model_id,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=dtype,
        ).to(cfg.device)

        # Memory optimisations.
        if cfg.enable_xformers:
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
                print("  xformers: enabled")
            except Exception:
                print("  xformers: not available")

        if cfg.enable_vae_slicing:
            self._pipe.enable_vae_slicing()

        if cfg.cpu_offload:
            self._pipe.enable_model_cpu_offload()
            print("  CPU offload: enabled")

        # Load IP-Adapter if identity conditioning is requested.
        if cfg.identity_strength > 0 and cfg.ip_adapter_model_id:
            try:
                print(f"  IP-Adapter: {cfg.ip_adapter_model_id}")
                self._pipe.load_ip_adapter(
                    cfg.ip_adapter_model_id,
                    subfolder=cfg.ip_adapter_subfolder,
                    weight_name=cfg.ip_adapter_weight_name,
                )
                self._pipe.set_ip_adapter_scale(cfg.identity_strength)
                print(f"  IP-Adapter scale: {cfg.identity_strength}")
            except Exception as e:
                print(f"  IP-Adapter load warning: {e} — identity conditioning disabled")

        elapsed = time.perf_counter() - t0
        print(f"[DiffusionEngine] Loaded in {elapsed:.1f}s")
        self._loaded = True

    def unload(self) -> None:
        """Release VRAM."""
        if not self._loaded:
            return
        self._pipe = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._loaded = False
        print("[DiffusionEngine] Unloaded.")

    def __enter__(self) -> "DiffusionEngine":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate_frames(
        self,
        pose_maps: list["PIL.Image.Image"],
        prompt: Optional[str] = None,
        reference_image: Optional["PIL.Image.Image"] = None,
        batch_size: int = 1,
    ) -> list["PIL.Image.Image"]:
        """Generate one avatar frame per pose map.

        Parameters
        ----------
        pose_maps : list of PIL Images
            ControlNet conditioning images (one per frame).
        prompt : str, optional
            Positive text prompt.  Defaults to cfg.default_prompt.
        reference_image : PIL Image, optional
            Reference signer photo for IP-Adapter identity conditioning.
            Ignored if cfg.identity_strength == 0.
        batch_size : int
            Number of frames to generate per diffusion call.
            batch_size=1 is safest for VRAM; increase for throughput.

        Returns
        -------
        list of PIL Images — generated avatar frames.
        """
        if not self._loaded:
            raise RuntimeError("Call load() before generate_frames()")

        import torch

        cfg = self.cfg
        effective_prompt = prompt or cfg.default_prompt
        generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed)

        results: list["PIL.Image.Image"] = []
        n = len(pose_maps)
        t_total = time.perf_counter()

        for batch_start in range(0, n, batch_size):
            batch_maps = pose_maps[batch_start: batch_start + batch_size]
            b = len(batch_maps)

            pipe_kwargs = dict(
                prompt=[effective_prompt] * b,
                negative_prompt=[cfg.negative_prompt] * b,
                image=batch_maps,
                controlnet_conditioning_scale=cfg.controlnet_conditioning_scale,
                num_inference_steps=cfg.num_steps,
                guidance_scale=cfg.guidance_scale,
                width=cfg.width,
                height=cfg.height,
                generator=generator,
                output_type="pil",
            )

            # IP-Adapter: pass reference image if loaded and available.
            if reference_image is not None and cfg.identity_strength > 0:
                pipe_kwargs["ip_adapter_image"] = [reference_image] * b

            output = self._pipe(**pipe_kwargs)
            results.extend(output.images)

            elapsed = time.perf_counter() - t_total
            done = batch_start + b
            fps_so_far = done / elapsed if elapsed > 0 else 0
            print(
                f"  [{done}/{n}] {elapsed:.1f}s elapsed  "
                f"({elapsed / done:.2f}s/frame)"
            )

        total = time.perf_counter() - t_total
        print(
            f"[DiffusionEngine] {n} frames in {total:.1f}s "
            f"({total / max(n, 1):.2f}s/frame)"
        )
        return results

    def generate_single(
        self,
        pose_map: "PIL.Image.Image",
        prompt: Optional[str] = None,
        reference_image: Optional["PIL.Image.Image"] = None,
    ) -> "PIL.Image.Image":
        """Generate a single avatar frame.  Convenience wrapper."""
        return self.generate_frames([pose_map], prompt, reference_image)[0]
