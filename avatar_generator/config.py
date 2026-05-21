"""GeneratorConfig — all knobs for avatar_generator in one place.

Load from YAML:
    cfg = GeneratorConfig.from_yaml("avatar_generator/config.yaml")

Save to YAML:
    cfg.save("my_config.yaml")

Override at runtime:
    cfg = GeneratorConfig(width=1024, height=1024, num_steps=30)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GeneratorConfig:
    # ── Model IDs (HuggingFace Hub) ──────────────────────────────────
    base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_id: str = "thibaud/controlnet-openpose-sdxl-1.0"
    # IP-Adapter for identity conditioning (set to "" to disable)
    ip_adapter_model_id: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "sdxl_models"
    ip_adapter_weight_name: str = "ip-adapter-plus-face_sdxl_vit-h.bin"

    # ── Output resolution ────────────────────────────────────────────
    width: int = 768
    height: int = 768

    # ── Diffusion sampling ───────────────────────────────────────────
    num_steps: int = 25
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 0.85
    # Identity conditioning strength (0 = disabled, 0.4 = recommended)
    identity_strength: float = 0.40
    negative_prompt: str = (
        "blurry, low quality, deformed hands, extra fingers, missing fingers, "
        "bad anatomy, watermark, text, ugly, cartoon, anime"
    )

    # ── Prompts ──────────────────────────────────────────────────────
    default_prompt: str = (
        "a person performing sign language, photorealistic, high quality, "
        "detailed hands and fingers, studio lighting, neutral background, 4k"
    )

    # ── Pose map rendering ───────────────────────────────────────────
    # Source resolution of OpenPose JSON coordinates
    pose_source_width: int = 1280
    pose_source_height: int = 720
    # Render hands at 2x then downsample for anti-aliased finger lines
    hand_supersample: bool = True
    pose_confidence_threshold: float = 0.1

    # ── Video assembly ───────────────────────────────────────────────
    output_fps: int = 25
    output_codec: str = "libx264"
    output_crf: int = 18          # H.264 quality (18 = near-lossless)

    # ── Hardware ─────────────────────────────────────────────────────
    device: str = "cuda"
    torch_dtype: str = "float16"  # "float16" or "bfloat16"
    enable_xformers: bool = True
    enable_vae_slicing: bool = True
    # Offload to CPU between steps — slower but saves VRAM on small GPUs
    cpu_offload: bool = False

    # ── Reproducibility ──────────────────────────────────────────────
    seed: int = 42

    # ── Output ───────────────────────────────────────────────────────
    save_frames: bool = False     # also save individual PNG frames
    frames_subdir: str = "frames"

    # ── .skels format ────────────────────────────────────────────────
    # Number of joints × 3 coords per frame in .skels files
    skels_joints: int = 50        # 50 joints × xyz = 150 floats/frame
    # Canvas size used when projecting 3D skels to 2D pixel coords
    skels_canvas_width: int = 1280
    skels_canvas_height: int = 720

    # ────────────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import yaml
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(asdict(self), f, allow_unicode=True, sort_keys=False)
        except ImportError:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GeneratorConfig":
        path = Path(path)
        try:
            import yaml
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except ImportError:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str | Path) -> "GeneratorConfig":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
