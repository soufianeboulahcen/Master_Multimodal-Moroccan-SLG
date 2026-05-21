"""Avatar rendering configuration.

All hyperparameters for the avatar rendering subsystem in one place.
Mirrors the pattern established by SignLLMConfig in mosl/model/signllm.py.

Defaults are tuned for the DGX Spark (GB10, 128 GB unified memory, CUDA 13)
running the NGC PyTorch 26.04-py3 base image.  Override via AvatarConfig(...)
or load from a JSON file with AvatarConfig.from_json().
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class BackendType(str, Enum):
    """Diffusion rendering backend selection."""
    ANIMATEDIFF = "animatediff"
    MIMICMOTION = "mimicmotion"
    AUTO = "auto"           # select based on sign type heuristic


class IdentityBackend(str, Enum):
    """Identity conditioning method."""
    INSTANTID = "instantid"
    IP_ADAPTER = "ip_adapter"
    NONE = "none"           # no identity conditioning (skeleton-only output)


@dataclass
class PoseMapConfig:
    """Controls how OpenPose JSON is rendered into ControlNet conditioning images."""

    # Output resolution for pose maps fed to ControlNet.
    # Must match the diffusion model's expected input resolution.
    width: int = 768
    height: int = 768

    # Source resolution of the OpenPose JSON coordinates.
    # Matches the procedural generator and SignLLM export (1280×720).
    source_width: int = 1280
    source_height: int = 720

    # ControlNet branch weights.  Two branches are used simultaneously:
    #   body_branch: full-body skeleton (lower weight — context)
    #   hand_branch: hands only (higher weight — sign-critical detail)
    body_controlnet_weight: float = 0.65
    hand_controlnet_weight: float = 0.90

    # Render hands at 2× resolution then downsample for anti-aliased finger lines.
    hand_supersample: bool = True
    hand_supersample_factor: int = 2

    # Keypoint colours follow the DWPose convention that ControlNet was trained on.
    # Overriding these risks misalignment with the model's training distribution.
    use_dwpose_colours: bool = True

    # Minimum confidence threshold below which a keypoint is treated as missing.
    confidence_threshold: float = 0.1


@dataclass
class AnimateDiffConfig:
    """AnimateDiff-specific rendering parameters."""

    # Base diffusion model.  SDXL gives best quality; SD1.5 is faster.
    base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # AnimateDiff motion adapter for SDXL.
    motion_adapter_id: str = "guoyww/animatediff-motion-adapter-sdxl-beta"

    # ControlNet model for OpenPose conditioning.
    controlnet_openpose_id: str = "thibaud/controlnet-openpose-sdxl-1.0"

    # Sliding context window size (frames).  AnimateDiff processes this many
    # frames at once; longer clips are handled via overlapping windows.
    context_frames: int = 16
    context_overlap: int = 4       # frames shared between adjacent windows

    # Diffusion sampling parameters.
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    negative_prompt: str = (
        "blurry, low quality, deformed hands, extra fingers, missing fingers, "
        "anatomically incorrect, watermark, text, ugly, bad anatomy"
    )

    # Memory optimisation flags (safe to enable on DGX Spark).
    enable_xformers: bool = True
    enable_vae_slicing: bool = True
    enable_model_cpu_offload: bool = False  # keep on GPU for batch throughput

    # Inference dtype.  fp16 halves VRAM at negligible quality cost.
    torch_dtype: str = "float16"


@dataclass
class MimicMotionConfig:
    """MimicMotion-specific rendering parameters."""

    model_id: str = "tencent/MimicMotion"

    # SVD (Stable Video Diffusion) base model.
    svd_model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt"

    num_inference_steps: int = 25
    guidance_scale: float = 3.0
    noise_aug_strength: float = 0.02

    # MimicMotion processes fixed-length chunks; longer clips are stitched.
    chunk_size: int = 16
    chunk_overlap: int = 6

    torch_dtype: str = "float16"


@dataclass
class IdentityConfig:
    """Identity preservation configuration (InstantID or IP-Adapter)."""

    backend: IdentityBackend = IdentityBackend.INSTANTID

    # InstantID model paths.
    instantid_model_id: str = "InstantX/InstantID"
    instantid_controlnet_id: str = "InstantX/InstantID"   # face ControlNet

    # IP-Adapter model paths (fallback).
    ip_adapter_model_id: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "sdxl_models"
    ip_adapter_weight_name: str = "ip-adapter-plus-face_sdxl_vit-h.bin"

    # InsightFace face detection model (used by both InstantID and IP-Adapter-Face).
    insightface_model: str = "buffalo_l"

    # Conditioning strength.  Higher = more identity-faithful, less pose-faithful.
    # 0.3–0.5 is the recommended range for sign language (pose must dominate).
    identity_strength: float = 0.40

    # Cache face embeddings to disk so the same reference is never re-encoded.
    cache_embeddings: bool = True
    embedding_cache_dir: str = "outputs/avatar/.embedding_cache"


@dataclass
class RIFEConfig:
    """RIFE temporal interpolation configuration."""

    # RIFE model variant.  4.x series is recommended.
    model_version: str = "4.6"

    # Upsampling factor: 2 = 25fps→50fps, 4 = 25fps→100fps.
    scale_factor: int = 2

    # UHD mode for high-resolution inputs (>1080p).  Not needed at 768×768.
    uhd: bool = False

    # Run RIFE on CPU if GPU VRAM is exhausted after diffusion.
    device: str = "cuda"


@dataclass
class AvatarConfig:
    """Top-level configuration for the avatar rendering pipeline.

    Usage:
        cfg = AvatarConfig()                          # all defaults
        cfg = AvatarConfig(backend=BackendType.MIMICMOTION)
        cfg = AvatarConfig.from_json("avatar_cfg.json")
        cfg.save("avatar_cfg.json")
    """

    # Backend selection.
    backend: BackendType = BackendType.ANIMATEDIFF

    # Sub-configs.
    pose_map: PoseMapConfig = field(default_factory=PoseMapConfig)
    animatediff: AnimateDiffConfig = field(default_factory=AnimateDiffConfig)
    mimicmotion: MimicMotionConfig = field(default_factory=MimicMotionConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    rife: RIFEConfig = field(default_factory=RIFEConfig)

    # Output settings.
    output_fps: int = 25
    output_codec: str = "libx264"
    output_crf: int = 18           # H.264 quality (lower = better, 18 is near-lossless)

    # Device.
    device: str = "cuda"
    seed: int = 42

    # Whether to run RIFE interpolation after diffusion rendering.
    interpolate: bool = True

    # Whether to apply hand super-resolution post-processing.
    hand_superres: bool = True

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str | Path) -> "AvatarConfig":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls()
        # Shallow-merge top-level keys; nested dataclasses reconstructed manually.
        cfg.backend = BackendType(data.get("backend", cfg.backend))
        cfg.output_fps = data.get("output_fps", cfg.output_fps)
        cfg.output_codec = data.get("output_codec", cfg.output_codec)
        cfg.output_crf = data.get("output_crf", cfg.output_crf)
        cfg.device = data.get("device", cfg.device)
        cfg.seed = data.get("seed", cfg.seed)
        cfg.interpolate = data.get("interpolate", cfg.interpolate)
        cfg.hand_superres = data.get("hand_superres", cfg.hand_superres)
        if "pose_map" in data:
            cfg.pose_map = PoseMapConfig(**data["pose_map"])
        if "animatediff" in data:
            cfg.animatediff = AnimateDiffConfig(**data["animatediff"])
        if "mimicmotion" in data:
            cfg.mimicmotion = MimicMotionConfig(**data["mimicmotion"])
        if "identity" in data:
            d = data["identity"].copy()
            if "backend" in d:
                d["backend"] = IdentityBackend(d["backend"])
            cfg.identity = IdentityConfig(**d)
        if "rife" in data:
            cfg.rife = RIFEConfig(**data["rife"])
        return cfg
