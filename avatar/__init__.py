"""Avatar rendering subsystem.

Converts SignLLM skeleton outputs (OpenPose JSON) into photorealistic
avatar videos using diffusion-based rendering.

Integration seam:
    .skels → mosl/pose/export_openpose_json.py → OpenPose JSON
           → avatar/conditioning/pose_to_controlnet.py → RGB pose maps
           → avatar/backends/ (AnimateDiff / MimicMotion)
           → avatar/interpolation/rife.py
           → outputs/avatar/<sign>/<sign>_avatar.mp4

The existing SignLLM pipeline is never modified.
"""
from avatar.pipeline import AvatarPipeline
from avatar.config import AvatarConfig, BackendType

__all__ = ["AvatarPipeline", "AvatarConfig", "BackendType"]
