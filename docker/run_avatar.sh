#!/usr/bin/env bash
# Run a command inside the avatar generation image (pfe-avatar:latest).
#
# Usage:
#   docker/run_avatar.sh                                    # interactive bash
#   docker/run_avatar.sh python scripts/avatar/download_models.py
#   docker/run_avatar.sh python scripts/avatar/pose_to_controlnet_map.py \
#       --source procedural --motion walking
#   docker/run_avatar.sh python scripts/avatar/generate_avatar_video.py \
#       --pose-maps outputs/avatar/pose_maps/walking/ \
#       --reference assets/avatar_reference.jpg \
#       --prompt "a Moroccan man performing sign language" \
#       --out outputs/avatar/videos/walking.mp4
#
# Differences from docker/run.sh:
#   - Uses pfe-avatar:latest (diffusion stack) instead of pfe-pose:latest
#   - shm-size increased to 16g (AnimateDiff loads multiple large models)
#   - HuggingFace cache bind-mounted from host to avoid re-downloading weights
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE="${PFE_AVATAR_IMAGE:-pfe-avatar:latest}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    echo "image $IMAGE not found — building from docker/Dockerfile.diffusion..." >&2
    # Build on top of pfe-pose:latest (build it first if needed)
    if ! docker image inspect pfe-pose:latest > /dev/null 2>&1; then
        echo "building base image pfe-pose:latest first..." >&2
        docker build -t pfe-pose:latest "$PROJECT_ROOT/docker"
    fi
    docker build -t "$IMAGE" -f "$PROJECT_ROOT/docker/Dockerfile.diffusion" "$PROJECT_ROOT"
fi

if [ $# -eq 0 ]; then
    set -- bash
fi

TTY_FLAG=()
[ -t 1 ] && TTY_FLAG=(-t)

exec docker run --rm -i "${TTY_FLAG[@]}" \
    --gpus all \
    --shm-size=16g \
    -u "$(id -u):$(id -g)" \
    -v "$PROJECT_ROOT:/workspace/PFE-SOUFIAN" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -w /workspace/PFE-SOUFIAN \
    "$IMAGE" \
    "$@"
