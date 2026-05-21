"""avatar_generator — standalone avatar image/video generation module.

Converts SignLLM skeleton outputs into photorealistic human avatar frames
without modifying any existing pipeline.

Inputs accepted:
    .skels files  (SignLLM compressed pose format)
    OpenPose JSON directories  (per-frame *_keypoints.json)

Output:
    PNG frames  or  MP4 video clip

Minimal usage:
    from avatar_generator import AvatarGenerator, GeneratorConfig
    gen = AvatarGenerator(GeneratorConfig())
    gen.load()
    gen.generate(
        pose_source="outputs/openpose_json/walking_keypoints",
        output_path="outputs/avatar_generator/walking.mp4",
    )
    gen.unload()
"""
from avatar_generator.generator import AvatarGenerator
from avatar_generator.config import GeneratorConfig

__all__ = ["AvatarGenerator", "GeneratorConfig"]
