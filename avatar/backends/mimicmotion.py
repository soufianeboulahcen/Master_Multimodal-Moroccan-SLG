"""MimicMotion rendering backend.

Generates photorealistic full-body video using MimicMotion (Tencent), which
is built on Stable Video Diffusion (SVD).  MimicMotion conditions on a
reference image and a DWPose keypoint sequence to produce temporally coherent
video with strong identity preservation.

When to use MimicMotion vs AnimateDiff:
    MimicMotion  — full-body showcase renders, high temporal coherence,
                   strong identity from a single reference frame.
    AnimateDiff  — hand-critical sign language motion, dual ControlNet
                   for fine hand detail, more flexible prompt control.

MimicMotion expects DWPose-format keypoints, not CMU OpenPose.  This backend
handles the conversion from our OpenPose v1.3 JSON format automatically via
_openpose_to_dwpose().

Chunk strategy:
    SVD processes fixed-length chunks (default 16 frames).  Longer clips are
    split into overlapping chunks and stitched using the same linear-blend
    approach as AnimateDiff.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from avatar.backends.base import RenderBackend, RenderResult
from avatar.config import MimicMotionConfig, PoseMapConfig


# DWPose body keypoint order (17 joints, COCO format used by MimicMotion).
# Maps from our COCO-18 indices to DWPose COCO-17 (no neck joint).
_COCO18_TO_COCO17 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 2, 3, 4]


def _openpose_to_dwpose(openpose_json_path: Path) -> dict:
    """Convert one OpenPose v1.3 JSON frame to DWPose-compatible dict.

    DWPose format expected by MimicMotion:
        bodies.candidate  : (17, 2) float — COCO-17 body keypoints, normalised [0,1]
        bodies.subset     : (1, 17) float — confidence scores
        hands             : (2, 21, 2) float — [left, right] hand keypoints
        faces             : (1, 68, 2) float — face keypoints (zeros if absent)
    """
    import json as _json
    with open(openpose_json_path, encoding="utf-8") as f:
        data = _json.load(f)

    people = data.get("people", [{}])
    person = people[0] if people else {}

    body_flat = person.get("pose_keypoints_2d", [0.0] * 75)
    lhand_flat = person.get("hand_left_keypoints_2d", [0.0] * 33)
    rhand_flat = person.get("hand_right_keypoints_2d", [0.0] * 33)

    # Parse body: (25, 3) → take COCO-18 subset → remap to COCO-17.
    body_arr = np.array(body_flat, dtype=np.float32).reshape(-1, 3)
    if body_arr.shape[0] < 18:
        pad = np.zeros((18 - body_arr.shape[0], 3), dtype=np.float32)
        body_arr = np.concatenate([body_arr, pad], axis=0)
    body_18 = body_arr[:18]
    # Normalise to [0, 1] (source is 1280×720).
    body_18[:, 0] /= 1280.0
    body_18[:, 1] /= 720.0
    # Remap to COCO-17.
    body_17 = body_18[_COCO18_TO_COCO17]
    candidate = body_17[:, :2]   # (17, 2)
    subset = body_17[:, 2:3].T   # (1, 17)

    # Parse hands: (11, 3) → expand to (21, 2) via linear extrapolation.
    from avatar.conditioning.pose_to_controlnet import _parse_kpts, _extrapolate_hand_21
    lhand_11 = _parse_kpts(lhand_flat, 11)
    rhand_11 = _parse_kpts(rhand_flat, 11)
    lhand_21 = _extrapolate_hand_21(lhand_11)[:, :2]
    rhand_21 = _extrapolate_hand_21(rhand_11)[:, :2]
    # Normalise.
    lhand_21[:, 0] /= 1280.0
    lhand_21[:, 1] /= 720.0
    rhand_21[:, 0] /= 1280.0
    rhand_21[:, 1] /= 720.0
    hands = np.stack([lhand_21, rhand_21], axis=0)   # (2, 21, 2)

    return {
        "bodies": {"candidate": candidate, "subset": subset},
        "hands": hands,
        "faces": np.zeros((1, 68, 2), dtype=np.float32),
    }


class MimicMotionBackend(RenderBackend):
    """MimicMotion (SVD-based) full-body rendering backend."""

    def __init__(
        self,
        mimicmotion_cfg: Optional[MimicMotionConfig] = None,
        pose_cfg: Optional[PoseMapConfig] = None,
        device: str = "cuda",
    ) -> None:
        super().__init__(device)
        self.cfg = mimicmotion_cfg or MimicMotionConfig()
        self.pose_cfg = pose_cfg or PoseMapConfig()
        self._pipe = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return

        print("[MimicMotion] Loading model stack…")
        t0 = time.perf_counter()

        try:
            import torch
        except ImportError as e:
            raise ImportError("torch is required") from e

        dtype = getattr(torch, self.cfg.torch_dtype)

        # MimicMotion is distributed as a HuggingFace model.
        # The pipeline class is in the MimicMotion repo; we import it
        # from third_party if available, otherwise from the installed package.
        try:
            from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
        except ImportError:
            try:
                from diffusers import MimicMotionPipeline  # future diffusers integration
            except ImportError as e:
                raise ImportError(
                    "MimicMotion is not installed.\n"
                    "Install: follow https://github.com/tencent/MimicMotion\n"
                    "or: pip install git+https://github.com/tencent/MimicMotion.git"
                ) from e

        self._pipe = MimicMotionPipeline.from_pretrained(
            self.cfg.model_id, torch_dtype=dtype
        ).to(self.device)

        elapsed = time.perf_counter() - t0
        print(f"[MimicMotion] Loaded in {elapsed:.1f}s")
        self._loaded = True

    def unload(self) -> None:
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
        print("[MimicMotion] Unloaded.")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self,
        body_map_paths: list[Path],
        hand_map_paths: list[Path],
        identity_conditioning: dict,
        prompt: str = "",   # MimicMotion is image-conditioned, not text-conditioned
        seed: int = 42,
    ) -> RenderResult:
        """Generate frames using MimicMotion chunk inference."""
        if not self._loaded:
            raise RuntimeError("Call load() before render()")

        import torch

        n_frames = len(body_map_paths)
        if n_frames == 0:
            raise ValueError("body_map_paths is empty")

        ref_image = identity_conditioning.get("ref_image")
        if ref_image is None:
            raise ValueError(
                "MimicMotion requires a reference image in identity_conditioning['ref_image']"
            )

        # MimicMotion needs the original JSON files for DWPose conversion.
        # body_map_paths are PNGs; we derive the JSON dir from the naming convention.
        # Convention: body_map_paths are in <out_dir>/body/body_XXXXXX.png
        # JSON files are in the json_dir passed to PoseMapRenderer.
        # We store the json_dir in the metadata dict passed through the pipeline.
        # If not available, fall back to using the body maps as ControlNet input.
        json_dir = identity_conditioning.get("_json_dir")

        chunks = self._build_chunks(n_frames)
        print(f"[MimicMotion] Rendering {n_frames} frames in {len(chunks)} chunk(s)")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        all_frames: list[Optional[np.ndarray]] = [None] * n_frames
        t_total = time.perf_counter()

        for c_idx, (start, end) in enumerate(chunks):
            c_len = end - start
            print(f"  chunk {c_idx + 1}/{len(chunks)}: frames {start}–{end - 1}")
            t0 = time.perf_counter()

            # Build DWPose sequence for this chunk if JSON dir is available.
            if json_dir is not None:
                json_dir_path = Path(json_dir)
                json_files = sorted(json_dir_path.glob("*_keypoints.json"))
                chunk_pose = [
                    _openpose_to_dwpose(json_files[i])
                    for i in range(start, min(end, len(json_files)))
                ]
            else:
                chunk_pose = None

            pipe_kwargs = dict(
                ref_image=ref_image,
                num_frames=c_len,
                num_inference_steps=self.cfg.num_inference_steps,
                guidance_scale=self.cfg.guidance_scale,
                noise_aug_strength=self.cfg.noise_aug_strength,
                generator=generator,
                output_type="np",
            )
            if chunk_pose is not None:
                pipe_kwargs["pose_sequence"] = chunk_pose

            output = self._pipe(**pipe_kwargs)
            chunk_frames_f = output.frames[0]   # (T, H, W, C)
            chunk_frames = (chunk_frames_f * 255).clip(0, 255).astype(np.uint8)

            elapsed_c = time.perf_counter() - t0
            print(f"    {elapsed_c:.1f}s  ({elapsed_c / c_len:.2f}s/frame)")

            overlap = self.cfg.chunk_overlap
            for local_i, global_i in enumerate(range(start, end)):
                if all_frames[global_i] is None:
                    all_frames[global_i] = chunk_frames[local_i]
                else:
                    alpha = min(local_i / max(overlap, 1), 1.0)
                    prev = all_frames[global_i].astype(np.float32)
                    curr = chunk_frames[local_i].astype(np.float32)
                    all_frames[global_i] = (
                        ((1 - alpha) * prev + alpha * curr).clip(0, 255).astype(np.uint8)
                    )

        total_elapsed = time.perf_counter() - t_total
        frames = [f for f in all_frames if f is not None]
        print(
            f"[MimicMotion] Done: {len(frames)} frames in {total_elapsed:.1f}s "
            f"({total_elapsed / max(len(frames), 1):.2f}s/frame)"
        )

        return RenderResult(
            frames=frames,
            fps=25.0,
            metadata={
                "backend": "mimicmotion",
                "n_chunks": len(chunks),
                "total_time_s": total_elapsed,
                "seed": seed,
            },
        )

    def _build_chunks(self, n_frames: int) -> list[tuple[int, int]]:
        size = self.cfg.chunk_size
        overlap = self.cfg.chunk_overlap
        stride = size - overlap
        if n_frames <= size:
            return [(0, n_frames)]
        chunks: list[tuple[int, int]] = []
        start = 0
        while start < n_frames:
            end = min(start + size, n_frames)
            chunks.append((start, end))
            if end == n_frames:
                break
            start += stride
        return chunks
