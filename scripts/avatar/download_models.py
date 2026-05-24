"""Download all model weights needed for the avatar generation pipeline.

Run once before the first generation. Downloads to HuggingFace cache
(~/.cache/huggingface/hub) by default.

Usage:
    python scripts/avatar/download_models.py
    python scripts/avatar/download_models.py --skip-flux   # SDXL stack only
    python scripts/avatar/download_models.py --list        # show what will be downloaded
"""
from __future__ import annotations

import argparse
import sys

MODELS = {
    "sdxl_base": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "desc": "SDXL 1.0 base UNet (~6.9 GB)",
        "required": True,
    },
    "controlnet_openpose_sdxl": {
        "repo": "thibaud/controlnet-openpose-sdxl-1.0",
        "desc": "ControlNet OpenPose for SDXL (~2.5 GB)",
        "required": True,
    },
    "animatediff_sdxl": {
        "repo": "guoyww/animatediff-motion-adapter-sdxl-beta",
        "desc": "AnimateDiff motion adapter for SDXL (~0.5 GB)",
        "required": True,
    },
    "ip_adapter_sdxl_face": {
        "repo": "h94/IP-Adapter",
        "desc": "IP-Adapter face model for SDXL (~1.0 GB)",
        "required": True,
        "subfolder": "sdxl_models",
        "filename": "ip-adapter-plus-face_sdxl_vit-h.bin",
    },
    "clip_vit_large": {
        "repo": "openai/clip-vit-large-patch14-336",
        "desc": "CLIP ViT-L/14@336 for IP-Adapter encoding (~1.7 GB)",
        "required": True,
    },
    "svd_xt": {
        "repo": "stabilityai/stable-video-diffusion-img2vid-xt",
        "desc": "Stable Video Diffusion XT (~9.5 GB) — optional",
        "required": False,
    },
    "controlnet_sd15_openpose": {
        "repo": "lllyasviel/control_v11p_sd15_openpose",
        "desc": "ControlNet OpenPose for SD1.5 (~1.4 GB) — lightweight fallback",
        "required": False,
    },
}


def download_model(key: str, info: dict) -> None:
    from huggingface_hub import snapshot_download, hf_hub_download
    print(f"\n[download] {key}: {info['desc']}")
    try:
        if "filename" in info:
            path = hf_hub_download(
                repo_id=info["repo"],
                subfolder=info.get("subfolder"),
                filename=info["filename"],
            )
        else:
            path = snapshot_download(repo_id=info["repo"])
        print(f"[download] ✓ {key} → {path}")
    except Exception as e:
        print(f"[download] ✗ {key} failed: {e}", file=sys.stderr)
        if info.get("required"):
            raise


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="List models without downloading")
    p.add_argument("--skip-optional", action="store_true",
                   help="Skip non-required models (SVD, SD1.5 fallback)")
    p.add_argument("--only", nargs="+", choices=list(MODELS.keys()),
                   help="Download only specific models")
    args = p.parse_args()

    if args.list:
        print("Models to download:")
        for key, info in MODELS.items():
            req = "required" if info["required"] else "optional"
            print(f"  [{req:8s}] {key}: {info['desc']}")
        return 0

    to_download = args.only or list(MODELS.keys())
    for key in to_download:
        info = MODELS[key]
        if args.skip_optional and not info["required"]:
            print(f"[download] skipping optional: {key}")
            continue
        download_model(key, info)

    print("\n[download] all done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
